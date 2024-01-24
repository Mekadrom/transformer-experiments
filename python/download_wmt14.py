from datasets import load_dataset
from tqdm import tqdm

import os

dataset = load_dataset('wmt14', 'de-en', cache_dir='data/actual')

allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,!?ßäöüÄÖÜ"\';:()[]{}<>+-*/=@#$%&|~`^_\n')

def is_valid(s):
    return all(c in allowed_chars for c in s)

def save_to_file(data, src_filename, tgt_filename):
    with open(os.path.join('data', 'actual', src_filename), 'w', encoding='utf-8') as src_file, open(os.path.join('data', 'actual', tgt_filename), 'w', encoding='utf-8') as tgt_file:
        for example in tqdm(data):
            en = example['translation']['en']
            de = example['translation']['de']

            if is_valid(en) and is_valid(de):
                en = en.replace("' s ", "'s ")

                src_file.write(en + '\n')
                tgt_file.write(de + '\n')

# Save train, validation, and test sets
save_to_file(dataset['train'], 'train.src', 'train.tgt')
save_to_file(dataset['validation'], 'valid.src', 'valid.tgt')
save_to_file(dataset['test'], 'test.src', 'test.tgt')