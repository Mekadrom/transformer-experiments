from collections import OrderedDict
from dataloader import SequenceLoader
from positional_encodings.torch_encodings import PositionalEncoding2D
from rotary_embedding_torch import RotaryEmbedding
from tqdm import tqdm
from transformer_provider import TransformerModelProvider

import argparse
import codecs
import math
import os
import torch
import youtokentome
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import sacrebleu

def get_lr(step, d_model, warmup_steps):
    """
    The LR schedule. This version below is twice the definition in the paper, as used in the official T2T repository.

    :param step: training step number
    :param d_model: size of vectors throughout the transformer model
    :param warmup_steps: number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official T2T repo
    :return: updated learning rate
    """
    return 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))

def get_buffered_positional_encoding(args, d_model, maxlen=100, num_dims=1):
    """
    Computes positional encoding as defined in the paper.

    :param d_model: size of vectors throughout the transformer model
    :param max_length: maximum sequence length up to which positional encodings must be calculated
    :return: positional encoding, a tensor of size (1, max_length, d_model)
    """
    if num_dims == 1:
        positional_encoding = torch.zeros((maxlen, d_model)) # (max_length, d_model)
        for i in range(maxlen):
            for k in range(d_model):
                if k % 2 == 0:
                    positional_encoding[i, k] = math.sin(i / math.pow(10000, k / d_model))
                else:
                    positional_encoding[i, k] = math.cos(i / math.pow(10000, (k - 1) / d_model))
        positional_encoding = positional_encoding.unsqueeze(0) # (1, max_length, d_model)
    elif num_dims == 2:
        positional_encoding_2d = PositionalEncoding2D(args.positional_encoding_dim).to(args.device)
        positional_encoding = torch.zeros((1, maxlen, maxlen, args.positional_encoding_dim))
        positional_encoding = positional_encoding_2d(positional_encoding.to(args.device))
    return positional_encoding  # (1, max_length, d_model) or (1, max_length, max_length, d_model)

def load_tokenizers(run_dir):
    src_tokenizer_file = os.path.join(run_dir, 'src_tokenizer.model')
    tgt_tokenizer_file = os.path.join(run_dir, 'tgt_tokenizer.model')

    if not os.path.exists(src_tokenizer_file):
        raise Exception(f"Source tokenizer file {src_tokenizer_file} does not exist")

    print(f"Loading source tokenizer from {src_tokenizer_file}")
    src_bpe_model = youtokentome.BPE(model=src_tokenizer_file)

    if os.path.exists(tgt_tokenizer_file):
        print(f"Loading target tokenizer from {tgt_tokenizer_file}")
        tgt_bpe_model = youtokentome.BPE(model=tgt_tokenizer_file)
    else:
        print(f"Sharing tokenizer between source and target languages")
        tgt_bpe_model = src_bpe_model

    return src_bpe_model, tgt_bpe_model

def get_positional_encoding(args):
    if args.positional_encoding_type == 'sinusoidal' or args.positional_encoding_type == 'buffer':
        if args.qkv_config == 'kv+pos':
            positional_encoding = get_buffered_positional_encoding(
                args,
                d_model=args.d_model,
                maxlen=args.maxlen+1,
                num_dims=2
            ).to(args.device)
        else:
            positional_encoding = get_buffered_positional_encoding(
                args,
                d_model=args.d_model,
                maxlen=args.maxlen+1,
            ).to(args.device)
    elif args.positional_encoding_type == 'rotary':
        positional_encoding = RotaryEmbedding(dim=args.positional_encoding_dim)
    return positional_encoding

def load_checkpoint_or_generate_new(args, run_dir, src_bpe_model, tgt_bpe_model, checkpoint_model_name='transformer_checkpoint.pth.tar'):
    print('Initializing model...')

    if os.path.exists(os.path.join(run_dir, checkpoint_model_name)):
        checkpoint = torch.load(os.path.join(run_dir, checkpoint_model_name))
        if hasattr(args, 'start_epoch') and args.start_epoch == 0:
            args.start_epoch = checkpoint['epoch'] + 1
            print('\nLoaded checkpoint from epoch %d.\n' % args.start_epoch)

        model = checkpoint['model']

        if 'optimizer' in checkpoint:
            optimizer = checkpoint['optimizer']
        else:
            optimizer = None

        if 'positional_encoding' in checkpoint:
            positional_encoding = checkpoint['positional_encoding']
            if (type(positional_encoding) == 'RotaryEmbedding' and args.positional_encoding_type != 'RotaryEmbedding') or (type(positional_encoding) != 'RotaryEmbedding' and args.positional_encoding_type == 'RotaryEmbedding'):
                print("WARNING: positional encoding type mismatch between args and saved model. Using positional encoding from args instead.")
                positional_encoding = get_positional_encoding(args)
        else:
            positional_encoding = None
    else:
        print("Starting from scratch...")
        positional_encoding = get_positional_encoding(args)
        model = TransformerModelProvider().provide(args, src_bpe_model.vocab_size(), tgt_bpe_model.vocab_size(), positional_encoding=positional_encoding,  tie_embeddings=tgt_bpe_model==src_bpe_model)
        optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=[args.beta1, args.beta2], eps=args.epsilon)

    return model, optimizer, positional_encoding

def print_model(model):
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')
    print(f'The model has {sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad):,} non-zero trainable parameters')
    print(f"Model structure: \n {model}")

def load_data(tokens_in_batch, run_dir, src_bpe_model, tgt_bpe_model):
    print('Loading training data SequenceLoader...')
    train_loader = SequenceLoader(
        src_bpe_model=src_bpe_model,
        tgt_bpe_model=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix="tgt",
        split="train",
        tokens_in_batch=tokens_in_batch
    )

    print('Loading validation data SequenceLoader...')
    val_loader = SequenceLoader(
        src_bpe_model=src_bpe_model,
        tgt_bpe_model=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix="tgt",
        split="val",
        tokens_in_batch=tokens_in_batch
    )

    print('Loading test data SequenceLoader...')
    test_loader = SequenceLoader(
        src_bpe_model=src_bpe_model,
        tgt_bpe_model=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix="tgt",
        split="test",
        tokens_in_batch=tokens_in_batch
    )
    return train_loader, val_loader, test_loader

def save_checkpoint(epoch, model, optimizer, positional_encoding, prefix=''):
    """
    Checkpoint saver. Each save overwrites previous save.

    :param epoch: epoch number (0-indexed)
    :param model: transformer model
    :param optimizer: optimized
    :param prefix: checkpoint filename prefix
    """
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer, 'positional_encoding': positional_encoding}
    filename = prefix + 'transformer_checkpoint.pth.tar'
    torch.save(state, filename)

def change_lr(optimizer, new_lr):
    """
    Scale learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be changed
    :param new_lr: new learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def beam_search_translate(src, model, src_bpe_model, tgt_bpe_model, device, beam_size=4, length_norm_coefficient=0.6):
    """
    Translates a source language sequence to the target language, with beam search decoding.

    :param source_sequence: the source language sequence, either a string or tensor of bpe-indices
    :param beam_size: beam size
    :param length_norm_coefficient: co-efficient for normalizing decoded sequences' scores by their lengths
    :return: the best hypothesis, and all candidate hypotheses
    """
    with torch.no_grad():
        # Beam size
        k = beam_size

        # Minimum number of hypotheses to complete
        n_completed_hypotheses = min(k, 10)

        # Vocab size
        tgt_vocab_size = tgt_bpe_model.vocab_size()

        # If the source sequence is a string, convert to a tensor of IDs
        if isinstance(src, str):
            encoder_sequences = src_bpe_model.encode(
                src,
                output_type=youtokentome.OutputType.ID,
                bos=False,
                eos=False
            )
            encoder_sequences = torch.LongTensor(encoder_sequences).unsqueeze(0) # (1, source_sequence_length)
        else:
            encoder_sequences = src
        encoder_sequences = encoder_sequences.to(device) # (1, source_sequence_length)
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences.size(1)]).to(device) # (1)

        # Encode
        encoder_sequences = model.encoder(encoder_sequences=encoder_sequences, encoder_sequence_lengths=encoder_sequence_lengths) # (1, source_sequence_length, d_model)

        # Our hypothesis to begin with is just <BOS>
        hypotheses = torch.LongTensor([[tgt_bpe_model.subword_to_id('<BOS>')]]).to(device) # (1, 1)
        hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device) # (1)

        # Tensor to store hypotheses' scores; now it's just 0
        hypotheses_scores = torch.zeros(1).to(device) # (1)

        # Lists to store completed hypotheses and their scores
        completed_hypotheses = list()
        completed_hypotheses_scores = list()

        # Start decoding
        step = 1

        # Assume "s" is the number of incomplete hypotheses currently in the bag; a number less than or equal to "k"
        # At this point, s is 1, because we only have 1 hypothesis to work with, i.e. "<BOS>"
        while True:
            s = hypotheses.size(0)
            decoder_sequences = model.decoder(
                decoder_sequences=hypotheses,
                decoder_sequence_lengths=hypotheses_lengths,
                encoder_sequences=encoder_sequences.repeat(s, 1, 1),
                encoder_sequence_lengths=encoder_sequence_lengths.repeat(s) # (s, step, tgt_vocab_size)
            )

            # Scores at this step
            scores = decoder_sequences[:, -1, :] # (s, tgt_vocab_size)
            scores = F.log_softmax(scores, dim=-1) # (s, tgt_vocab_size)

            # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
            scores = hypotheses_scores.unsqueeze(1) + scores # (s, tgt_vocab_size)

            # Unroll and find top k scores, and their unrolled indices
            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True) # (k)

            # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
            prev_word_indices = unrolled_indices // tgt_vocab_size # (k)
            next_word_indices = unrolled_indices % tgt_vocab_size # (k)

            # Construct the the new top k hypotheses from these indices
            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1) # (k, step + 1)

            # Which of these new hypotheses are complete (reached <EOS>)?
            complete = next_word_indices == tgt_bpe_model.subword_to_id('<EOS>') # (k), bool

            # Set aside completed hypotheses and their scores normalized by their lengths
            # For the length normalization formula, see
            # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
            completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

            # Stop if we have completed enough hypotheses
            if len(completed_hypotheses) >= n_completed_hypotheses:
                break

            # Else, continue with incomplete hypotheses
            hypotheses = top_k_hypotheses[~complete] # (s, step + 1)
            hypotheses_scores = top_k_hypotheses_scores[~complete] # (s)
            hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device) # (s)

            # Stop if things have been going on for too long
            if step > 100:
                break
            step += 1

        # If there is not a single completed hypothesis, use partial hypotheses
        if len(completed_hypotheses) == 0:
            completed_hypotheses = hypotheses.tolist()
            completed_hypotheses_scores = hypotheses_scores.tolist()

        # Decode the hypotheses
        all_hypotheses = list()
        for i, h in enumerate(tgt_bpe_model.decode(completed_hypotheses, ignore_ids=[0, 2, 3])):
            all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

        # Find the best scoring completed hypothesis
        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["hypothesis"]

        return best_hypothesis, all_hypotheses

def average_checkpoints(positional_encoding, source_folder, model_name_prefix='step', model_name_suffix='.pth.tar'):
    source_folder = f"runs/{source_folder}" if not source_folder.startswith('runs/') else source_folder
    
    # Get list of checkpoint names
    checkpoint_names = [f for f in os.listdir(source_folder) if f.startswith(model_name_prefix) and f.endswith(model_name_suffix)]
    assert len(checkpoint_names) > 0, "Did not find any checkpoints!"

    # Average parameters from checkpoints
    averaged_params = OrderedDict()
    for c in checkpoint_names:
        checkpoint = torch.load(os.path.join(source_folder, c))['model']
        checkpoint_params = checkpoint.state_dict()
        checkpoint_param_names = checkpoint_params.keys()
        for param_name in checkpoint_param_names:
            if param_name not in averaged_params:
                averaged_params[param_name] = checkpoint_params[param_name].clone() * 1 / len(checkpoint_names)
            else:
                averaged_params[param_name] += checkpoint_params[param_name] * 1 / len(checkpoint_names)

    # Use one of the checkpoints as a surrogate to load the averaged parameters into
    averaged_checkpoint = torch.load(os.path.join(source_folder, checkpoint_names[0]))['model']
    for param_name in averaged_checkpoint.state_dict().keys():
        assert param_name in averaged_params
    averaged_checkpoint.load_state_dict(averaged_params)

    # Save averaged checkpoint
    torch.save({'model': averaged_checkpoint, 'positional_encoding': positional_encoding}, f"{source_folder}/averaged_transformer_checkpoint.pth.tar")

def sacrebleu_evaluate(args, run_dir, src_bpe_model, tgt_bpe_model, model, device, sacrebleu_in_python):
    """
    Returns None when command line sacrebleu is used
    """

    before_nanos = time.time_ns()

    bleu_score = None

    test_loader = SequenceLoader(src_bpe_model=src_bpe_model,
                                tgt_bpe_model=tgt_bpe_model,
                                data_folder=os.path.join('..', "data"),
                                source_suffix="src",
                                target_suffix="tgt",
                                split="test",
                                tokens_in_batch=None)
    test_loader.create_batches()

    # Evaluate
    with torch.no_grad():

        hypotheses = list()
        references = list()
        for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(tqdm(test_loader, total=test_loader.n_batches)):
            hypotheses.append(beam_search_translate(src=source_sequence, src_bpe_model=src_bpe_model, tgt_bpe_model=tgt_bpe_model, device=device, model=model, beam_size=4, length_norm_coefficient=0.6)[0])
            references.extend(tgt_bpe_model.decode(target_sequence.tolist(), ignore_ids=[0, 2, 3]))

        if sacrebleu_in_python:
            print("\n13a tokenization, cased:\n")
            bleu_score = sacrebleu.corpus_bleu(hypotheses, [references])
            print(bleu_score)
            print("\n13a tokenization, caseless:\n")
            print(sacrebleu.corpus_bleu(hypotheses, [references], lowercase=True))
            print("\nInternational tokenization, cased:\n")
            print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl'))
            print("\nInternational tokenization, caseless:\n")
            print(sacrebleu.corpus_bleu(hypotheses, [references], tokenize='intl', lowercase=True))
            print("\n")
        else:
            cat_command = "cat" if os.name == "posix" else "type"

            with codecs.open(os.path.join(run_dir, "translated_test.tgt"), "w", encoding="utf-8") as f:
                f.write("\n".join(hypotheses))

            print("\n13a tokenization, cased:\n")
            os.system(f"{cat_command} translated_test.tgt | sacrebleu -t wmt14/full -l en-de")
            print("\n13a tokenization, caseless:\n")
            os.system(f"{cat_command} translated_test.tgt | sacrebleu -t wmt14/full -l en-de -lc")
            print("\nInternational tokenization, cased:\n")
            os.system(f"{cat_command} translated_test.tgt | sacrebleu -t wmt14/full -l en-de -tok intl")
            print("\nInternational tokenization, caseless:\n")
            os.system(f"{cat_command} translated_test.tgt | sacrebleu -t wmt14/full -l en-de -tok intl -lc")
            print("\n")
        print(
            "The first value (13a tokenization, cased) is how the BLEU score is officially calculated by WMT (mteval-v13a.pl). \nThis is probably not how it is calculated in the 'Attention Is All You Need' paper, however.\nSee https://github.com/tensorflow/tensor2tensor/issues/317#issuecomment-380970191 for more details.\n")
        
    after_nanos = time.time_ns()

    print(f"Time taken for sacrebleu evaluation: {(after_nanos - before_nanos) / 1e9} seconds")

    return bleu_score

def prune_structured(layer, is_encoder_layer, prune_heads_amount, prune_heads_norm, prune_ffn_amount, prune_ffn_norm, prune_type):
    # todo: implement this
    print("Pruning structured not supported yet")

    # if prune_type in ['heads', 'all']:
    #     prune.ln_structured(layer[0], name='weight', amount=prune_heads_amount, n=prune_heads_norm, dim=0)

    # if is_encoder_layer:
    #     if prune_type in ['ffn', 'all']:
    #         prune.ln_structured(layer[1], name='weight', amount=prune_ffn_amount, n=prune_ffn_norm, dim=0)
    # else:
    #     if prune_type in ['heads', 'all']:
    #         prune.ln_structured(layer[1], name='weight', amount=prune_heads_amount, n=prune_heads_norm, dim=0)

    #     if prune_type in ['ffn', 'all']:
    #         prune.ln_structured(layer[2], name='weight', amount=prune_ffn_amount, n=prune_ffn_norm, dim=0)

def prune_unstructured(layer, is_encoder_layer, prune_heads_amount, prune_ffn_amount, prune_type):
    if prune_type in ['heads', 'all']:
        prune.l1_unstructured(layer[0], name='weight', amount=prune_heads_amount, dim=0)

    if is_encoder_layer:
        if prune_type in ['ffn', 'all']:
            prune.l1_unstructured(layer[1], name='weight', amount=prune_ffn_amount, dim=0)
    else:
        if prune_type in ['heads', 'all']:
            prune.l1_unstructured(layer[1], name='weight', amount=prune_heads_amount, dim=0)

        if prune_type in ['ffn', 'all']:
            prune.l1_unstructured(layer[2], name='weight', amount=prune_ffn_amount, dim=0)

def prune_model(model, prune_heads_amount, prune_heads_norm, prune_ffn_amount, prune_ffn_norm, is_prune_structured, prune_type):
    """
    Prune the model.

    :param args: command-line arguments
    :param model: transformer model
    """

    for encoder_layer in model.encoder.encoder_layers:
        if is_prune_structured:
            prune_structured(encoder_layer, True, prune_heads_amount, prune_heads_norm, prune_ffn_amount, prune_ffn_norm, prune_type)
        else:
            prune_unstructured(encoder_layer, True, prune_heads_amount, prune_heads_norm, prune_ffn_amount, prune_ffn_norm, prune_type)

    for decoder_layer in model.decoder.decoder_layers:
        if is_prune_structured:
            prune_structured(decoder_layer, False, prune_heads_amount, prune_heads_norm, prune_ffn_amount, prune_ffn_norm, prune_type)
        else:
            prune_unstructured(decoder_layer, False, prune_heads_amount, prune_heads_norm, prune_ffn_amount, prune_ffn_norm, prune_type)

def create_activation_function(activation_function_name):
    if activation_function_name == 'relu':
        return nn.ReLU()
    elif activation_function_name == 'gelu':
        return nn.GELU()
    elif activation_function_name == 'elu':
        return nn.ELU()
    elif activation_function_name == 'selu':
        return nn.SELU()
    elif activation_function_name == 'prelu':
        return nn.PReLU()
    elif activation_function_name == 'leaky_relu':
        return nn.LeakyReLU()
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")

def get_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--run_name', type=str, required=True)
    argparser.add_argument('--tokenizer_run_name', type=str, required=True)

    # ease of changing default
    default_d_model = 512

    argparser.add_argument('--d_model', type=int, default=default_d_model)
    argparser.add_argument('--n_heads', type=int, default=8)
    argparser.add_argument('--d_queries', type=int, default=64)
    argparser.add_argument('--d_values', type=int, default=64)
    argparser.add_argument('--qkv_config', type=str, default='qkv', choices=['qkv', 'kv+pos', 'kv'])
    argparser.add_argument('--d_inner', type=int, default=2048)
    argparser.add_argument('--n_encoder_layers', type=int, default=6)
    argparser.add_argument('--n_decoder_layers', type=int, default=6)
    argparser.add_argument('--dropout', type=float, default=0.1)

    argparser.add_argument('--maxlen', type=int, default=150)

    argparser.add_argument('--init_weights_from', type=str, default='glorot_uniform', choices=['glorot_uniform', 'glorot_normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal'])
    argparser.add_argument('--init_weights_gain', type=float, default=1.0)
    argparser.add_argument('--use_admin', action='store_true')

    argparser.add_argument('--positional_encoding_type', type=str, default='sinusoidal', choices=['rotary', 'buffer', 'sinusoidal', 'none']) # buffer and sinusoidal refer to the same thing
    argparser.add_argument('--positional_encoding_dim', type=int, default=64) # 64 makes sense for rotary, but not for kv+pos buffer/sinusoidal
    argparser.add_argument('--learnable_positional_encoding', action='store_true', default=False)

    # values configured like so: "LayerType:LayerDim,LayerType:LayerDim,..."
    # eg "MultiHeadAttention:512,LightweightConv1d:256,DynamicConv1d:256"
    # the dim values summed up have to be divisible by the n_heads
    argparser.add_argument('--encoder_layer_self_attn_config', type=str, default=None)
    argparser.add_argument('--decoder_layer_self_attn_config', type=str, default=None)

    argparser.add_argument('--encoder_param_sharing_type', type=str, default='none', choices=['none', 'sequence', 'cycle', 'cycle-rev', 'ffn-cycle-rev', 'heads-cycle-rev', 'all'])
    argparser.add_argument('--decoder_param_sharing_type', type=str, default='none', choices=['none', 'sequence', 'cycle', 'cycle-rev', 'ffn-cycle-rev', 'heads-cycle-rev', 'all'])
    argparser.add_argument('--m_encoder_independent_layers', type=int, default=0)
    argparser.add_argument('--m_decoder_independent_layers', type=int, default=0)
    argparser.add_argument('--activation_function', type=str, default='relu', choices=['relu', 'gelu', 'elu', 'pau'])

    argparser.add_argument('--tokens_in_batch', type=int, default=5000)
    argparser.add_argument('--target_tokens_per_batch', type=int, default=25000)
    argparser.add_argument('--n_steps', type=int, default=100000)
    argparser.add_argument('--warmup_steps', type=int, default=8000)
    argparser.add_argument('--beta1', type=float, default=0.9)
    argparser.add_argument('--beta2', type=float, default=0.98)
    argparser.add_argument('--epsilon', type=float, default=1e-9)
    argparser.add_argument('--label_smoothing', type=float, default=0.1)

    argparser.add_argument('--prune_mode', type=str, default='none', choices=['none', 'only-prune', 'train-prune', 'train-prune-retrain'])
    argparser.add_argument('--prune_type', type=str, default='all', choices=['heads', 'ffn', 'all'])
    argparser.add_argument('--prune_structured', action='store_true')
    argparser.add_argument('--prune_heads_amount', type=float, default=0.0)
    argparser.add_argument('--prune_heads_norm', type=int, default=2)
    argparser.add_argument('--prune_ffn_amount', type=float, default=0.0)
    argparser.add_argument('--prune_ffn_norm', type=int, default=2)
    argparser.add_argument('--n_prune_retrains', type=int, default=1)
    argparser.add_argument('--prune_retrain_n_steps', type=int, default=10000)
    argparser.add_argument('--prune_retrain_warmup_steps', type=float, default=800)

    argparser.add_argument('--distillation_teacher_run_name', type=str, default=None)

    # debug/technical args that are not hyperparameters
    argparser.add_argument('--start_epoch', type=int, default=0)
    argparser.add_argument('--print_frequency', type=int, default=20)
    argparser.add_argument('--device', type=str, default='cuda:0')
    argparser.add_argument('--save_initial_checkpoint', action='store_true')

    args, unk = argparser.parse_known_args()

    if len(unk) > 0:
        print(f"unknown arguments: {unk}")

    if args.positional_encoding_type == 'rotary' and args.qkv_config != 'qkv':
        print("rotary positional encoding only works with qkv_config=qkv")
        exit(1)

    args.__setattr__('batches_per_step', args.target_tokens_per_batch // args.tokens_in_batch)
    args.__setattr__('lr', get_lr(step=1, d_model=args.d_model, warmup_steps=args.warmup_steps))

    if args.encoder_layer_self_attn_config is None:
        args.__setattr__('encoder_layer_self_attn_config', f"MultiHeadAttention:{args.d_model}:{args.n_heads}")

    if args.decoder_layer_self_attn_config is None:
        args.__setattr__('decoder_layer_self_attn_config', "shared")

    if args.decoder_layer_self_attn_config == 'shared':
        args.__setattr__('decoder_layer_self_attn_config', args.encoder_layer_self_attn_config)

    return args, unk
