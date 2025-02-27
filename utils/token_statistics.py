from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

import argparse

def colored_text(text, color_code):
    return f"\033[{color_code}m{text}\033[0m"

def check_word_in_dataset(path, name, words):
    dataset = load_dataset(path, name, split='train')

    for example in dataset:
        for word in words:
            if word in example['text']:
                print(f"Found '{colored_text(word, 31)}' in the following example:")
                print(example['text'].replace(word, colored_text(word, 31)))
                print()

def collect_dataset_token_statistics(path, name, tokenizer_name):
    dataset = load_dataset(path, name, split='train')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    token_counts = {}
    example_lengths = []
    for example in tqdm(dataset):
        tokens = tokenizer.encode(example['text'])
        example_lengths.append(len(tokens))
        for token_id in tokens:
            token = tokenizer.convert_ids_to_tokens(token_id)
            if token not in token_counts:
                token_counts[token] = 0
            token_counts[token] += 1

    return token_counts, example_lengths

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--dataset_name', type=str, required=True)
    argparser.add_argument('--dataset_config_name', type=str, required=True)
    argparser.add_argument('--tokenizer_name', type=str, required=True)

    args = argparser.parse_args()

    # token_counts, example_lengths = collect_dataset_token_statistics(args.dataset_name, args.dataset_config_name, args.tokenizer_name)

    # print(f"Found {len(token_counts)} unique tokens in the dataset.")
    # print(f"The dataset contains {sum(token_counts.values())} tokens in total using the {args.tokenizer_name} tokenizer.")
    # print(f"Top 10 most common tokens:")
    # for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
    #     print(f"{token}: {count}")

    # print(f"Top 10 least common tokens:")
    # for token, count in sorted(token_counts.items(), key=lambda x: x[1])[:10]:
    #     print(f"{token}: {count}")

    # print(f"Average example length: {sum(example_lengths) / len(example_lengths)}")
    # print(f"Max example length: {max(example_lengths)}")
    # print(f"Min example length: {min(example_lengths)}")

    # while True:
    #     token = input("Enter a token to check its frequency: ")
    #     if token in token_counts:
    #         print(f"Token '{token}' appears {token_counts[token]} times in the dataset.")
    #     else:
    #         print(f"Token '{token}' not found in the dataset.")
    #     print()

    print(len(load_dataset(args.dataset_name, args.dataset_config_name, split='train')))

    # check_word_in_dataset(args.dataset_name, args.dataset_config_name, ['Higgs', 'boson'])
