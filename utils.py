from collections import OrderedDict
from dataloader import SequenceLoader
from datasets import load_dataset
from positional_encodings.torch_encodings import PositionalEncoding2D
from modules.swiglu import SwiGLU
from multigpu_translation_training_wrapper import MultiGPUTranslationWrapper
from torch import nn
from torch.backends import cudnn
from tqdm import tqdm
from typing import Optional, Tuple, Union

import argparse
import codecs
import math
import numpy as np
import os
import random
import sacrebleu
import time
import torch
import torch.nn.functional as F
import yaml_dict
import youtokentome as yttm

lang_tag_token_ids = {
    'en': 0,
    'de': 1,
    'cs': 2,
    'fi': 3,
    'fr': 4,
    'gu': 5,
    'kk': 6,
    'lt': 7,
    'ru': 8,
    'et': 9,
    'tr': 10,
    'lv': 11,
    'ro': 12,
    'hi': 13
}

def lang_code_to_id(lang_code):
    return lang_tag_token_ids[lang_code] + 4

def vocab_size(args, bpe_model: yttm.BPE):
    if hasattr(args, 'multilang') and bool(args.multilang):
        return bpe_model.vocab_size() + len(lang_tag_token_ids)
    return bpe_model.vocab_size()

def encode(multilang: bool, tokenizer: yttm.BPE, sentences: Union[str, list[str]], *args, **kwargs):
    if isinstance(sentences, str):
        sentences = [sentences]
    else:
        sentences = sentences

    assert isinstance(sentences, list), f"Expected sentences to be a list of strings, but got {type(sentences)}"
    assert isinstance(sentences[0], str), f"Expected sentences to be a list of strings, but got list[{type(sentences[0])}]"

    if not multilang:
        return tokenizer.encode(sentences, *args, **kwargs)

    tokenized = []
    for s in sentences:
        splits = s.split('__')
        if len(splits) != 2:
            raise Exception(f"Expected string to be in the format 'lang_code__string', but got \"{s}\"")
        lang_code, s = splits
        tokens = tokenizer.encode(s, *args, **kwargs)
        offset_tokens = [lang_code_to_id(lang_code)]
        for token in tokens:
            if token < 4:
                offset_tokens.append(token)
            else:
                offset_tokens.append(token + len(lang_tag_token_ids))
        tokenized.append(offset_tokens)
    return tokenized

def decode(multilang: bool, tokenizer: yttm.BPE, ids: Union[list[int], list[list[int]]], *args, **kwargs):
    if isinstance(ids[0], int):
        ids = [ids]

    if isinstance(ids, (torch.Tensor, torch.LongTensor)):
        if len(ids.size()) == 2:
            ids = ids.tolist()
        elif len(ids.size()) == 3:
            ids = ids.tolist()
            ids = [tensor.tolist() for tensor in ids]
        else:
            raise Exception(f"Expected ids to be a list of lists of integers, but got {ids.size()}")

    assert isinstance(ids, list), f"Expected ids to be a list of lists of integers, but got {type(ids)}"
    assert isinstance(ids[0], list), f"Expected ids to be a list of lists of integers, but got list[{type(ids[0])}]"
    assert isinstance(ids[0][0], int), f"Expected ids to be a list of lists of integers, but got list[list[{type(ids[0][0])}]]"

    if not multilang:
        sentences = tokenizer.decode(ids, *args, **kwargs)
        tokens = []
        for token_set in ids:
            token_set = [tokenizer.id_to_subword(token) for token in token_set]
            tokens.append(token_set)
        return sentences, tokens

    sentences = []
    tokens = []
    for seq in ids:
        lang_code = f"{list(lang_tag_token_ids.keys())[seq[0] - 4]}"

        seq = seq[1:]
        seq = [(token - len(lang_tag_token_ids) if token >= len(lang_tag_token_ids) else token) for token in seq]
        seq = [token.item() if isinstance(token, torch.Tensor) else token for token in seq]

        sentence = tokenizer.decode(seq, *args, **kwargs)[0]
        token_set = [f"<{lang_code}>"] + [tokenizer.id_to_subword(token) for token in seq]
        sentence = f"{lang_code}__{sentence}"
        sentences.append(sentence)
        tokens.append(token_set)
    return sentences, tokens

def sanitize_model(model):
    if hasattr(model, '_orig_mod'):
        return sanitize_model(model._orig_mod)
    
    return model

def get_lr(step, d_model, warmup_steps):
    return 2. * math.pow(d_model, -0.5) * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))

def change_lr(optimizer: torch.optim.Optimizer, new_lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def get_buffered_positional_encoding(args, d_model, device, maxlen=100, num_dims=1):
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
        positional_encoding_2d = PositionalEncoding2D(args.positional_encoding_dim).to(device)
        positional_encoding = torch.zeros((1, maxlen, maxlen, args.positional_encoding_dim))
        positional_encoding = positional_encoding_2d(positional_encoding.to(device))
    return positional_encoding  # (1, max_length, d_model) or (1, max_length, max_length, d_model)

def get_tensor_positional_encoding(args, device):
    positional_encoding = get_buffered_positional_encoding(
        args,
        d_model=args.d_model,
        device=device,
        maxlen=args.maxlen + 1,
    ).to(device)
    positional_encoding.requires_grad = bool(args.learnable_positional_encoding)
    return positional_encoding

def load_tokenizers(run_dir) -> Tuple[yttm.BPE, yttm.BPE]:
    src_tokenizer_file = os.path.join(run_dir, 'src_tokenizer.model')
    tgt_tokenizer_file = os.path.join(run_dir, 'tgt_tokenizer.model')

    if not os.path.exists(src_tokenizer_file):
        raise Exception(f"Source tokenizer file {src_tokenizer_file} does not exist")

    print(f"Loading source tokenizer from {src_tokenizer_file}")
    src_bpe_model = yttm.BPE(model=src_tokenizer_file)

    if os.path.exists(tgt_tokenizer_file):
        print(f"Loading target tokenizer from {tgt_tokenizer_file}")
        tgt_bpe_model = yttm.BPE(model=tgt_tokenizer_file)
    else:
        print(f"Sharing tokenizer between source and target languages")
        tgt_bpe_model = src_bpe_model

    return src_bpe_model, tgt_bpe_model

def get_optimizer(optimizer_name, model: nn.Module, lr, beta1, beta2, epsilon, weight_decay):
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(params=[p for p in model.parameters() if p.requires_grad], lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)
    else:
        return torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay)

def print_model(model: nn.Module):
    print(f"Model structure: \n {model}")
    print(f'The model has {count_parameters(model):,} total parameters')
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

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_translation_data(args, tokens_in_batch, run_dir, src_bpe_model, tgt_bpe_model, pad_to_length=None) -> Tuple[SequenceLoader, SequenceLoader, SequenceLoader]:
    print('Loading training data SequenceLoader...')

    train_loader = SequenceLoader(
        args=args,
        src_tokenizer=src_bpe_model,
        tgt_tokenizer=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix="tgt",
        split="train",
        tokens_in_batch=tokens_in_batch,
        pad_to_length=pad_to_length
    )

    print('Loading validation data SequenceLoader...')
    val_loader = SequenceLoader(
        args=args,
        src_tokenizer=src_bpe_model,
        tgt_tokenizer=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix="tgt",
        split="val",
        tokens_in_batch=tokens_in_batch,
        pad_to_length=pad_to_length
    )

    print('Loading test data SequenceLoader...')
    test_loader = SequenceLoader(
        args=args,
        src_tokenizer=src_bpe_model,
        tgt_tokenizer=tgt_bpe_model,
        data_folder=os.path.join(run_dir),
        source_suffix="src",
        target_suffix="tgt",
        split="test",
        tokens_in_batch=tokens_in_batch,
        pad_to_length=pad_to_length
    )
    return train_loader, val_loader, test_loader

def save_checkpoint(epoch, model: nn.Module, optimizer: torch.optim.Optimizer, prefix=''):
    if isinstance(model, MultiGPUTranslationWrapper):
        state = model.save_checkpoint()
    else:
        state = {'epoch': epoch, 'model': model, 'optimizer': optimizer}
    torch.save(state, f"{prefix}transformer_checkpoint.pth.tar")

def average_checkpoints(epoch, optimizer, source_folder, num_latest_checkpoints=None, model_name_prefix='step', model_name_suffix='_transformer_checkpoint.pth.tar'):
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

def greedy_translate(args, src, model: nn.Module, src_tokenizer: yttm.BPE, tgt_tokenizer: yttm.BPE):
    with torch.no_grad():
        # If the source sequence is a string, convert to a tensor of IDs
        if isinstance(src, str):
            encoder_sequences = encode(args, src_tokenizer, src)
            encoder_sequences = torch.LongTensor(encoder_sequences).unsqueeze(0)

        else:
            encoder_sequences = src
        encoder_sequences = encoder_sequences.to(args.encoder_device)
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences.size(1)]).to(args.encoder_device)

        src_key_padding_mask = (encoder_sequences == 0).to(args.encoder_device)

        encoder_sequences, _ = model.encoder(encoder_sequences, encoder_sequence_lengths, src_key_padding_mask)

        steps = 0
        decoded = torch.LongTensor([[tgt_tokenizer.subword_to_id('<BOS>')]]).to(args.decoder_device)
        while True:
            decoder_sequences, _ = model.decoder(decoded, torch.LongTensor([decoded.size(1)]).to(args.decoder_device), encoder_sequences, encoder_sequence_lengths)

            next_word_scores = decoder_sequences[:, -1, :]
            next_word_scores = F.log_softmax(next_word_scores, dim=-1)

            next_word = torch.argmax(next_word_scores, dim=-1)

            decoded = torch.cat([decoded, next_word.unsqueeze(1)], dim=1)

            if next_word.item() == tgt_tokenizer.subword_to_id('<EOS>'):
                return ' '.join(tgt_tokenizer.decode(decoded.tolist(), ignore_ids=[0, 2, 3]))

            steps += 1
            if steps >= args.maxlen:
                # gone on too long
                break

        return ' '.join(tgt_tokenizer.decode(decoded.tolist(), ignore_ids=[0, 2, 3]))

def beam_search_translate(args, src, start_token, model: nn.Module, src_tokenizer: yttm.BPE, tgt_tokenizer: yttm.BPE, beam_size=4, length_norm_coefficient=0.6):
    """
    Translates a source language sequence to the target language, with beam search decoding.

    :param source_sequence: the source language sequence, either a string or tensor of bpe-indices
    :param beam_size: beam size
    :param length_norm_coefficient: co-efficient for normalizing decoded sequences' scores by their lengths
    :return: the best hypothesis, and all candidate hypotheses
    """
    encoder_sequences: Optional[torch.Tensor] = None
    with torch.no_grad():
        # Beam size
        k = beam_size

        # Minimum number of hypotheses to complete
        n_completed_hypotheses = min(k, 10)

        # Vocab size
        tgt_vocab_size = vocab_size(args, tgt_tokenizer)

        # If the source sequence is a string, convert to a tensor of IDs
        if isinstance(src, str):
            encoder_sequences = encode(bool(args.multilang), src_tokenizer, src)
            encoder_sequences = torch.LongTensor(encoder_sequences)
        else:
            encoder_sequences = src
        encoder_sequences = encoder_sequences.to(args.encoder_device) # (1, source_sequence_length)

        src_key_padding_mask = (encoder_sequences == 0).to(args.encoder_device) # (1, source_sequence_length)
        
        encoder_sequences, _ = model.encoder(encoder_sequences, src_key_padding_mask) # (1, source_sequence_length, d_model)

        # hypotheses begin with just lang code tag for the target language
        hypotheses = torch.LongTensor([[start_token]]) # (1, 1)

        # tensor to store hypotheses' scores; now it's just 0
        hypotheses_scores = torch.zeros(1).to(args.decoder_device) # (1)

        # lists to store completed hypotheses and their scores
        completed_hypotheses = []
        completed_hypotheses_scores = []

        step = 1
        # assume "s" is the number of incomplete hypotheses currently in the bag; a number less than or equal to "k"
        # at this point, s is 1, because we only have 1 hypothesis to work with, i.e. "<BOS>"
        while True:
            s = hypotheses.size(0)
            hypotheses = hypotheses.to(args.encoder_device)

            tgt_key_padding_masks = torch.zeros(s, hypotheses.size(1)).to(args.decoder_device).bool()

            decoder_sequences, _ = model.decoder(
                hypotheses,
                encoder_sequences.repeat(s, 1, 1),
                src_key_padding_mask.repeat(s, 1), # (s, 1)
                tgt_key_padding_masks
            )

            hypotheses = hypotheses.to(args.decoder_device)

            # Scores at this step
            scores = decoder_sequences[:, -1, :] # (s, tgt_vocab_size)
            scores = F.log_softmax(scores, dim=-1) # (s, tgt_vocab_size)

            # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
            scores = hypotheses_scores.unsqueeze(1) + scores # (s, tgt_vocab_size)

            # Unroll and find top k scores, and their unrolled indices
            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True) # (k)
            top_k_hypotheses_scores = top_k_hypotheses_scores.to(args.decoder_device)
            unrolled_indices = unrolled_indices.to(args.decoder_device)

            # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
            prev_word_indices = unrolled_indices // tgt_vocab_size # (k)
            next_word_indices = unrolled_indices % tgt_vocab_size # (k)

            # print(f"hypotheses: {hypotheses.shape}")
            # print(f"prev_word_indices: {prev_word_indices.shape}, {prev_word_indices.min()}, {prev_word_indices.max()}")
            # print(f"next_word_indices: {next_word_indices.shape}, {next_word_indices.min()}, {next_word_indices.max()}")

            # Construct the the new top k hypotheses from these indices
            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)], dim=1) # (k, step + 1)

            # Which of these new hypotheses are complete (reached <EOS>)?
            complete = (next_word_indices == tgt_tokenizer.subword_to_id('<EOS>')).to(args.decoder_device) # (k), bool

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
        for i, h in enumerate(decode(bool(args.multilang), tgt_tokenizer, completed_hypotheses, ignore_ids=[0, 2, 3])[0]):
            all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

        # Find the best scoring completed hypothesis
        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["hypothesis"]

        return best_hypothesis, all_hypotheses

def sacrebleu_evaluate(args, run_dir, src_bpe_model, tgt_bpe_model, model, sacrebleu_in_python, test_loader=None):
    """
    Returns None when command line sacrebleu is used
    """

    start_token = tgt_bpe_model.subword_to_id('<BOS>')
    if hasattr(args, 'multilang') and bool(args.multilang):
        start_token = lang_code_to_id(lang_code)

    before_nanos = time.time_ns()

    bleu_score = None

    if test_loader is None:
        test_loader = SequenceLoader(
            args=args,
            src_tokenizer=src_bpe_model,
            tgt_tokenizer=tgt_bpe_model,
            data_folder=os.path.join('.', "data"),
            source_suffix="src",
            target_suffix="tgt",
            split="test",
            tokens_in_batch=None
        )

    # Evaluate
    with torch.no_grad():
        hypotheses = list()
        references = list()
        for i, (source_sequence, target_sequence, _, _) in enumerate(tqdm(test_loader, total=test_loader.n_batches)):
            lang_code, target_sequence = target_sequence.split('__')
            hypotheses.append(beam_search_translate(args, src=source_sequence, start_token=start_token, src_tokenizer=src_bpe_model, tgt_tokenizer=tgt_bpe_model, model=model, beam_size=4, length_norm_coefficient=0.6)[0])
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

def get_activation_function(activation_function_name):
    if activation_function_name == 'relu':
        return nn.ReLU
    elif activation_function_name == 'gelu':
        return nn.GELU
    elif activation_function_name == 'elu':
        return nn.ELU
    elif activation_function_name == 'selu':
        return nn.SELU
    elif activation_function_name == 'prelu':
        return nn.PReLU
    elif activation_function_name == 'leaky_relu':
        return nn.LeakyReLU
    elif activation_function_name == 'silu':
        return nn.SiLU
    elif activation_function_name == 'none':
        return nn.Identity
    else:
        raise Exception(f"Unknown activation function {activation_function_name}")

def create_activation_function(d_in, activation_function_name):
    if activation_function_name == 'swiglu':
        return SwiGLU(d_in)
    return get_activation_function(activation_function_name)()

def get_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--run_name', type=str, required=True)
    argparser.add_argument('--config', type=str, required=True)

    argsparser_args, unk = argparser.parse_known_args()

    # convert unk list to dict
    unk = {unk[i][2:]: unk[i + 1] for i in range(0, len(unk), 2)}

    if len(unk) > 0:
        print(f"unknown arguments: {unk}")

    args = yaml_dict.load_yaml(argsparser_args.config, unk)

    setattr(args, 'run_name', argsparser_args.run_name)

    print(f"args: {args}")

    if args.n_gqa_groups == 0 or args.n_heads == 0:
        print("it is not recommended to not have any multi-head attention layers")
        exit(1)

    if hasattr(args, 'tokens_in_batch'):
        setattr(args, 'tokens_in_batch', int(args.tokens_in_batch))
        setattr(args, 'batches_per_step', int(args.target_tokens_per_batch) // args.tokens_in_batch)
    setattr(args, 'lr', get_lr(step=1, d_model=args.d_model, warmup_steps=args.warmup_steps))

    torch.set_printoptions(profile='full')

    torch.autograd.set_detect_anomaly(args.detect_nans)
    cudnn.benchmark = bool(args.cudnn_benchmark)

    if 'seed' in args and args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

    return args, unk
