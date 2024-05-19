from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from tokenizers.pre_tokenizers import Whitespace

import os
import utils

def train_tokenizer():
    args, unk = utils.get_args()

    train_dataset, _, _ = utils.load_llm_dataset(args.train_dataset, splits=('train'))

    print(f"Training LLM tokenizer on dataset: {type(train_dataset)}")

    def batch_iterator(batch_size=512):
        for i in range(0, len(train_dataset), batch_size):
            yield train_dataset[i: i + batch_size]["content"]

    tokenizer = ByteLevelBPETokenizer()
    normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Strip()])
    pre_tokenizer = Whitespace()

    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer

    tokenizer.train_from_iterator(batch_iterator(), vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save(os.path.join('llm', 'runs', args.run_name, 'bpe.json'))

if __name__ == '__main__':
    train_tokenizer()
