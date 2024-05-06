from collections import OrderedDict
from positional_encodings.torch_encodings import PositionalEncoding2D
from rotary_embedding_torch import RotaryEmbedding
from modules import swiglu
from tqdm import tqdm
from translation.dataloader import SequenceLoader
from translation.model_provider import TransformerModelProvider

import argparse
import codecs
import math
import os
import sacrebleu
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import yaml
import youtokentome

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
        positional_encoding = get_buffered_positional_encoding(
            args,
            d_model=args.d_model,
            maxlen=args.maxlen+1,
        ).to(args.device)
        positional_encoding.requires_grad = args.learnable_positional_encoding
    elif args.positional_encoding_type == 'rotary':
        positional_encoding = RotaryEmbedding(dim=args.positional_encoding_dim)
    return positional_encoding

def load_translation_checkpoint_or_generate_new(args, run_dir, src_bpe_model, tgt_bpe_model, checkpoint_model_name='transformer_checkpoint.pth.tar', vae_model=False):
    print('Initializing model...')

    if os.path.exists(os.path.join(run_dir, checkpoint_model_name)):
        checkpoint = torch.load(os.path.join(run_dir, checkpoint_model_name))
        if hasattr(args, 'start_epoch') and args.start_epoch == 0:
            args.start_epoch = checkpoint['epoch'] + 1
            print('\nLoaded checkpoint from epoch %d.\n' % args.start_epoch)

        if vae_model:
            model = TransformerModelProvider().provide_vae_transformer(args, src_bpe_model.vocab_size())
        else:
            model = TransformerModelProvider().provide_transformer(args, src_bpe_model.vocab_size(), tgt_bpe_model.vocab_size(), tie_embeddings=tgt_bpe_model==src_bpe_model)

        model.load_state_dict(checkpoint['model'].state_dict())

        if 'optimizer' in checkpoint:
            optimizer = checkpoint['optimizer']
        else:
            optimizer = None
    else:
        print("Starting from scratch...")
        if vae_model:
            model = TransformerModelProvider().provide_vae_transformer(args, src_bpe_model.vocab_size())
        else:
            model = TransformerModelProvider().provide_transformer(args, src_bpe_model.vocab_size(), tgt_bpe_model.vocab_size(), tie_embeddings=tgt_bpe_model==src_bpe_model)

        optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=args.lr, betas=[args.beta1, args.beta2], eps=args.epsilon)

    return model, optimizer

def print_model(model):
    print(f"Model structure: \n {model}")
    print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} total parameters')
    print(f'The model has {sum(torch.count_nonzero(p).item() for p in model.parameters() if p.requires_grad):,} non-zero total parameters')

    def tensor_in(tensor, tensor_list):
        for t in tensor_list:
            if tensor is t:
                return True
        return False

    already_counted = []
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad and not tensor_in(param, already_counted):
            # print(f"Layer {name} has {param.numel():,} parameters and {torch.count_nonzero(param).item():,} non-zero parameters")
            total_params += param.numel()
            already_counted.append(param)

    print(f'The model has {total_params:,} trainable parameters')

def load_data(tokens_in_batch, run_dir, src_bpe_model, tgt_bpe_model, vae_model=False):
    target_suffix = 'src' if vae_model else 'tgt'

    print('Loading training data SequenceLoader...')
    train_loader = SequenceLoader(
        src_bpe_model=src_bpe_model,
        tgt_bpe_model=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix=target_suffix,
        split="train",
        tokens_in_batch=tokens_in_batch
    )

    print('Loading validation data SequenceLoader...')
    val_loader = SequenceLoader(
        src_bpe_model=src_bpe_model,
        tgt_bpe_model=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix=target_suffix,
        split="val",
        tokens_in_batch=tokens_in_batch
    )

    print('Loading test data SequenceLoader...')
    test_loader = SequenceLoader(
        src_bpe_model=src_bpe_model,
        tgt_bpe_model=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix=target_suffix,
        split="test",
        tokens_in_batch=tokens_in_batch
    )
    return train_loader, val_loader, test_loader

def save_checkpoint(epoch, model, optimizer, prefix=''):
    """
    Checkpoint saver. Each save overwrites previous save.

    :param epoch: epoch number (0-indexed)
    :param model: transformer model
    :param optimizer: optimized
    :param prefix: checkpoint filename prefix
    """
    state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
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

def greedy_translate(args, src, model, src_bpe_model, tgt_bpe_model, device):
    with torch.no_grad():
        # If the source sequence is a string, convert to a tensor of IDs
        if isinstance(src, str):
            encoder_sequences = src_bpe_model.encode(
                src,
                output_type=youtokentome.OutputType.ID,
                bos=False,
                eos=False
            )
            encoder_sequences = torch.LongTensor(encoder_sequences).unsqueeze(0)

        else:
            encoder_sequences = src
        encoder_sequences = encoder_sequences.to(device)
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences.size(1)]).to(device)

        src_key_padding_mask = (encoder_sequences == 0).to(device)

        encoder_sequences, _ = model.encoder(encoder_sequences, encoder_sequence_lengths, src_key_padding_mask)

        if args.train_vae:
            # mu and logvar + reparemeterization trick to sample from the latent space, which replaces the encoder's output
            cls_token = encoder_sequences[:, 0, :]
            mu = model.mu(cls_token)
            logvar = model.logvar(cls_token)

            encoder_sequences = model.reparameterize(mu, logvar).unsqueeze(0)
            encoder_sequence_lengths = torch.ones(encoder_sequences.shape[:-1], dtype=torch.long, device=encoder_sequences.device)

        steps = 0
        decoded = torch.LongTensor([[tgt_bpe_model.subword_to_id('<BOS>')]]).to(device)
        while True:
            decoder_sequences, _ = model.decoder(decoded, torch.LongTensor([decoded.size(1)]).to(device), encoder_sequences, encoder_sequence_lengths)

            next_word_scores = decoder_sequences[:, -1, :]
            next_word_scores = F.log_softmax(next_word_scores, dim=-1)

            next_word = torch.argmax(next_word_scores, dim=-1)

            decoded = torch.cat([decoded, next_word.unsqueeze(1)], dim=1)

            if next_word.item() == tgt_bpe_model.subword_to_id('<EOS>'):
                return ' '.join(tgt_bpe_model.decode(decoded.tolist(), ignore_ids=[0, 2, 3]))

            steps += 1
            if steps >= args.maxlen:
                # gone on too long
                break

        return ' '.join(tgt_bpe_model.decode(decoded.tolist(), ignore_ids=[0, 2, 3]))

def beam_search_translate(args, src, model, src_bpe_model, tgt_bpe_model, device, beam_size=4, length_norm_coefficient=0.6):
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

        src_key_padding_mask = (encoder_sequences == 0).to(device) # (1, source_sequence_length)
        
        # Encode
        encoder_sequences, (t_mu, t_logvar), (q_mus, q_logvars), (k_mus, k_logvars), (v_mus, v_logvars), gating_variances = model.encoder(encoder_sequences, encoder_sequence_lengths, src_key_padding_mask) # (1, source_sequence_length, d_model)
        if args.train_vae:
            # mu and logvar + reparemeterization trick to sample from the latent space, which replaces the encoder's output
            cls_token = encoder_sequences[:, 0, :]
            mu = model.mu(cls_token)
            logvar = model.logvar(cls_token)

            encoder_sequences = model.reparameterize(mu, logvar)
            encoder_sequence_lengths = torch.ones(encoder_sequences.size(0), dtype=torch.long, device=encoder_sequences.device)

        # Our hypothesis to begin with is just <BOS>
        hypotheses = torch.LongTensor([[tgt_bpe_model.subword_to_id('<BOS>')]]).to(device) # (1, 1)
        hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device).unsqueeze(-1) # (1)

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

            tgt_key_padding_masks = torch.zeros(s, hypotheses.size(1)).to(device).bool()

            decoder_sequences, (t_mu, t_logvar), (s_q_mus, s_q_logvars), (s_k_mus, s_k_logvars), (s_v_mus, s_v_logvars), (c_q_mus, c_q_logvars), (c_k_mus, c_k_logvars), (c_v_mus, c_v_logvars), gating_variances = model.decoder(
                hypotheses,
                hypotheses_lengths,
                encoder_sequences.repeat(s, 1, 1),
                encoder_sequence_lengths.repeat(s).unsqueeze(-1), # (s, step, tgt_vocab_size)
                src_key_padding_mask.repeat(s, 1), # (s, 1)
                tgt_key_padding_masks
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
            hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device).unsqueeze(-1) # (s)

            # Stop if things have been going on for too long
            if step > args.maxlen:
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

def average_checkpoints(epoch, optimizer, source_folder, num_latest_checkpoints=None, model_name_prefix='step', model_name_suffix='_transformer_checkpoint.pth.tar'):
    source_folder = f"runs/{source_folder}" if not source_folder.startswith('runs/') else source_folder
    
    # Get list of checkpoint names
    checkpoint_names = [f for f in os.listdir(source_folder) if f.startswith(model_name_prefix) and f.endswith(model_name_suffix)]
    assert len(checkpoint_names) > 0, "Did not find any checkpoints!"

    # order the checkpoint names by step number
    checkpoint_names = sorted(checkpoint_names, key=lambda x: int(x[len(model_name_prefix):-len(model_name_suffix)]))

    if num_latest_checkpoints is not None:
        # only take X latest checkpoints
        checkpoint_names = checkpoint_names[-num_latest_checkpoints:]

    # Average parameters from checkpoints
    averaged_params = OrderedDict()
    for c in tqdm(checkpoint_names, desc="Averaging checkpoints"):
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
    torch.save({'epoch': epoch, 'model': averaged_checkpoint, 'optim': optimizer}, f"{source_folder}/averaged_transformer_checkpoint.pth.tar")

def sacrebleu_evaluate(args, run_dir, src_bpe_model, tgt_bpe_model, model, device, sacrebleu_in_python, test_loader=None, vae_model=False):
    """
    Returns None when command line sacrebleu is used
    """

    before_nanos = time.time_ns()

    bleu_score = None

    if test_loader is None:
        target_suffix = "src" if vae_model else "tgt"
        test_loader = SequenceLoader(src_bpe_model=src_bpe_model,
                                    tgt_bpe_model=tgt_bpe_model,
                                    data_folder=os.path.join('.', "data"),
                                    source_suffix="src",
                                    target_suffix=target_suffix,
                                    split="test",
                                    tokens_in_batch=None)
        test_loader.create_batches()

    # Evaluate
    with torch.no_grad():
        hypotheses = list()
        references = list()
        for i, (source_sequence, target_sequence, source_sequence_length, target_sequence_length) in enumerate(tqdm(test_loader, total=test_loader.n_batches)):
            hypotheses.append(beam_search_translate(args, src=source_sequence, src_bpe_model=src_bpe_model, tgt_bpe_model=tgt_bpe_model, device=device, model=model, beam_size=4, length_norm_coefficient=0.6)[0])
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

def create_activation_function(d_in, activation_function_name):
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
    elif activation_function_name == 'swiglu':
        return swiglu.SwiGLU(d_in)
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")

class YamlDict(dict):
    def __init__(self, *args, **kwargs):
        super(YamlDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, name):
        return self.__getitem__(name) if name in self else super().__getattribute__(name)

def load_yaml(file_path, ovr_args):
    file_path_dir = os.path.dirname(file_path)
    with open(os.path.join(file_path_dir, 'defaults.yaml'), 'r') as default_config:
        with open(file_path, 'r') as f:
            y = yaml.safe_load(default_config)
            y.update(yaml.safe_load(f))
            y.update(ovr_args)
            return YamlDict(y)

def get_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--run_name', type=str, required=True)
    argparser.add_argument('--config_file_path', type=str, required=True)

    argsparser_args, unk = argparser.parse_known_args()

    # convert unk list to dict
    unk = {unk[i][2:]: unk[i + 1] for i in range(0, len(unk), 2)}

    if len(unk) > 0:
        print(f"unknown arguments: {unk}")

    args = load_yaml(argsparser_args.config_file_path, unk)
    args.__setattr__('run_name', argsparser_args.run_name)

    print(f"args: {args}")

    if args.n_gqa_groups == 0 or args.n_heads == 0:
        print("it is not recommended to not have any multi-head attention layers")
        exit(1)

    args.__setattr__('batches_per_step', args.target_tokens_per_batch // args.tokens_in_batch)
    args.__setattr__('lr', get_lr(step=1, d_model=args.d_model, warmup_steps=args.warmup_steps))

    torch.set_printoptions(profile='full')

    torch.autograd.set_detect_anomaly(args.detect_nans)
    cudnn.benchmark = bool(args.cudnn_benchmark)

    return args, unk
