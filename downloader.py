from datasets import load_dataset
from tqdm import tqdm

import argparse
import codecs
import itertools
import os
import re
import youtokentome as yttm
import unicodedata
import utils

allowed_ranges = [
    (0x0000, 0x007F),  # Basic Latin (English, numbers, punctuation)
    (0x00A0, 0x00FF),  # Latin-1 Supplement (French, German, etc.)
    (0x0100, 0x017F),  # Latin Extended-A (Czech, Estonian, Lithuanian, Latvian)
    (0x0180, 0x024F),  # Latin Extended-B (Romanian, Vietnamese)
    (0x0250, 0x02AF),  # IPA Extensions (for various languages)
    (0x0300, 0x036F),  # Combining Diacritical Marks
    (0x0370, 0x03FF),  # Greek and Coptic (for loanwords)
    (0x0400, 0x04FF),  # Cyrillic (Russian, Kazakh)
    (0x0500, 0x052F),  # Cyrillic Supplement
    (0x1E00, 0x1EFF),  # Latin Extended Additional (Vietnamese)
    (0x2000, 0x206F),  # General Punctuation
    (0x2070, 0x209F),  # Superscripts and Subscripts
    (0x20A0, 0x20CF),  # Currency Symbols
    (0x2100, 0x214F),  # Letterlike Symbols
    (0x2150, 0x218F),  # Number Forms
    (0x2C60, 0x2C7F),  # Latin Extended-C
    (0xA720, 0xA7FF),  # Latin Extended-D
    (0xAB30, 0xAB6F),  # Latin Extended-E
    (0x0A80, 0x0AFF),  # Gujarati
    (0x0900, 0x097F),  # Devanagari (Hindi)
]

BLACKLISTED_CHARS = [
    '/',
    '\\'
]

all_valid_bytes = set(itertools.chain.from_iterable(range(start, end + 1) for start, end in allowed_ranges))

[all_valid_bytes.remove(ord(c)) for c in BLACKLISTED_CHARS]

argparser = argparse.ArgumentParser()

argparser.add_argument('--run_name', type=str, required=True)
argparser.add_argument('--dataset', type=str, default='wmt14', help='Dataset to download')
argparser.add_argument('--src_vocab_size', type=int, default=32000, help='Source vocabulary size')
argparser.add_argument('--tgt_vocab_size', type=int, default=32000, help='Target vocabulary size')
argparser.add_argument('--share_vocab', action='store_true', help='Share vocabulary between source and target languages')
argparser.add_argument('--min_length', type=int, default=3, help='Minimum number of tokens in an example')
argparser.add_argument('--max_length', type=int, default=192, help='Maximum number of tokens in an example')

args = argparser.parse_args()

run_dir = os.path.join('runs', args.run_name)
os.makedirs(run_dir, exist_ok=True)

os.makedirs('data', exist_ok=True)
os.makedirs(os.path.join('data', 'cache'), exist_ok=True)
os.makedirs(os.path.join('data', 'aggregate'), exist_ok=True)

def fix_example_line(line):
    return re.sub(r'\s+', ' ', line).strip()

def is_valid_byte(byte):
    return byte in all_valid_bytes

def is_valid_string(input_string):
    # Remove any combining characters (diacritics) for better matching
    normalized_string = ''.join(c for c in unicodedata.normalize('NFD', input_string) if unicodedata.category(c) != 'Mn')

    # Check if all characters in the normalized string are in the allowed ranges
    return all(is_valid_byte(ord(char)) for char in normalized_string) and not '----' in input_string and not '....' in input_string and not '##' in input_string and not '__' in input_string

def save_to_file(data, src_lang, tgt_lang, src_filename, tgt_filename, collation_fn=lambda x: x):
    src_filepath = os.path.join('data', 'aggregate', src_filename)
    tgt_filepath = os.path.join('data', 'aggregate', tgt_filename)
    tok_src_filepath = os.path.join('data', 'aggregate', src_filename + '.tok')
    tok_tgt_filepath = os.path.join('data', 'aggregate', tgt_filename + '.tok')
    with codecs.open(src_filepath, 'a', encoding='utf-8') as src_datafile, codecs.open(tgt_filepath, 'a', encoding='utf-8') as tgt_datafile:
        with codecs.open(tok_src_filepath, 'a', encoding='utf-8') as tok_src_datafile, codecs.open(tok_tgt_filepath, 'a', encoding='utf-8') as tok_tgt_datafile:
            for example in tqdm(data, unit=' examples', total=len(data)):
                example = collation_fn(example)

                src_example = example['translation'][src_lang]
                tgt_example = example['translation'][tgt_lang]
                if is_valid_string(src_example) and is_valid_string(tgt_example):
                    src_example = fix_example_line(src_example)
                    tgt_example = fix_example_line(tgt_example)

                    src_datafile.write(f"{src_lang}__{src_example}\n")
                    tgt_datafile.write(f"{tgt_lang}__{tgt_example}\n")
                    tok_src_datafile.write(src_example + '\n')
                    tok_tgt_datafile.write(tgt_example + '\n')

def save_datasets(dataset, src_lang, tgt_lang):
    save_to_file(dataset['train'], src_lang, tgt_lang, 'train.src', 'train.tgt')

    if 'validation' in dataset:
        val = 'validation'
    if 'valid' in dataset:
        val = 'valid'
    if 'val' in dataset:
        val = 'val'
        
    save_to_file(dataset[val], src_lang, tgt_lang, 'val.src', 'val.tgt')

    if 'test' in dataset:
        save_to_file(dataset['test'], src_lang, tgt_lang, 'test.src', 'test.tgt')

def download_dataset(path, src_lang, tgt_lang, name):
    dataset = load_dataset(path, name, cache_dir=os.path.join('data', 'cache'))
    save_datasets(dataset, src_lang, tgt_lang)

def train_tokenizer(share_vocab=False):
    src_filepath = os.path.join('data', 'aggregate', 'train.src')
    tgt_filepath = os.path.join('data', 'aggregate', 'train.tgt')
    if share_vocab:
        src_bpe = yttm.BPE.train(data=src_filepath, vocab_size=args.src_vocab_size, model=os.path.join(run_dir, 'src_tokenizer.model'))
        tgt_bpe = src_bpe
    else:
        src_bpe = yttm.BPE.train(data=src_filepath, vocab_size=args.src_vocab_size, model=os.path.join(run_dir, 'src_tokenizer.model'))
        tgt_bpe = yttm.BPE.train(data=tgt_filepath, vocab_size=args.tgt_vocab_size, model=os.path.join(run_dir, 'tgt_tokenizer.model'))
    return src_bpe, tgt_bpe

def filter_dataset(src_tokenizer, tgt_tokenizer, src_filename, tgt_filename):
    src_filepath = os.path.join('data', 'aggregate', src_filename)
    tgt_filepath = os.path.join('data', 'aggregate', tgt_filename)
    with codecs.open(src_filepath, 'r', encoding='utf-8') as src_datafile_in, codecs.open(tgt_filepath, 'r', encoding='utf-8') as tgt_datafile_in:
        with codecs.open(os.path.join(run_dir, src_filename), 'w', encoding='utf-8') as src_datafile_out, codecs.open(os.path.join(run_dir, tgt_filename), 'w', encoding='utf-8') as tgt_datafile_out:
            src = src_datafile_in.readlines()
            tgt = tgt_datafile_in.readlines()
            for s, t in zip(src, tgt):
                src_tokens = utils.encode(args, src_tokenizer, s.strip(), output_type=yttm.OutputType.ID)
                tgt_tokens = utils.encode(args, tgt_tokenizer, t.strip(), output_type=yttm.OutputType.ID)
                if args.min_length < len(src_tokens) < args.max_length and args.min_length < len(tgt_tokens) < args.max_length:
                    src_datafile_out.write(s)
                    tgt_datafile_out.write(t)

if args.dataset == 'wmt14':
    download_dataset('wmt/wmt14', 'de', 'en', 'de-en')
    src_bpe, tgt_bpe = train_tokenizer(share_vocab=args.share_vocab)
elif args.dataset == 'mixed':
    download_dataset("wmt/wmt19", "cs", "en", "cs-en")
    download_dataset("wmt/wmt19", "de", "en", "de-en")
    download_dataset("wmt/wmt19", "fi", "en", "fi-en")
    download_dataset("wmt/wmt19", "fr", "de", "fr-de")
    download_dataset("wmt/wmt19", "gu", "en", "gu-en")
    download_dataset("wmt/wmt19", "kk", "en", "kk-en")
    download_dataset("wmt/wmt19", "lt", "en", "lt-en")
    download_dataset("wmt/wmt19", "ru", "en", "ru-en")

    download_dataset("wmt/wmt18", "et", "en", "et-en")
    download_dataset("wmt/wmt18", "tr", "en", "tr-en")

    download_dataset("wmt/wmt17", "lv", "en", "lv-en")

    download_dataset("wmt/wmt16", "ro", "en", "ro-en")

    download_dataset("wmt/wmt15", "fr", "en", "fr-en")

    download_dataset("wmt/wmt14", "hi", "en", "hi-en")

    src_bpe, tgt_bpe = train_tokenizer(share_vocab=True)
else:
    raise ValueError(f"Unknown dataset {args.dataset}")

filter_dataset(src_bpe, tgt_bpe, 'train.src', 'train.tgt')
filter_dataset(src_bpe, tgt_bpe, 'val.src', 'val.tgt')
filter_dataset(src_bpe, tgt_bpe, 'test.src', 'test.tgt')
