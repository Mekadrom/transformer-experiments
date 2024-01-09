from datasets import load_dataset

dataset = load_dataset('wmt14', 'de-en', cache_dir='data/actual')

def save_to_file(data, src_filename, tgt_filename):
    with open(src_filename, 'w', encoding='utf-8') as src_file, open(tgt_filename, 'w', encoding='utf-8') as tgt_file:
        for example in data:
            src_file.write(example['translation']['en'] + '\n')
            tgt_file.write(example['translation']['de'] + '\n')

# Save train, validation, and test sets
save_to_file(dataset['train'], 'train.src', 'train.tgt')
save_to_file(dataset['validation'], 'valid.src', 'valid.tgt')
save_to_file(dataset['test'], 'test.src', 'test.tgt')