from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizer

import os
import torch

def create_causal_lm_dataloaders(tokenizer: PreTrainedTokenizer, dataset_name, dataset_config_name, batch_size, sequence_length, streaming=True):
    def preprocess_function(examples):
        examples = [example for example in examples["text"] if len(example) > 0]
        model_inputs = tokenizer(
            examples,
            max_length=sequence_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
        )
        return model_inputs

    dataset = load_dataset(dataset_name, dataset_config_name, streaming=streaming)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
    )

    dataloaders = {}
    for split in ["train", "validation", "test"]:
        if split in dataset:
            tokenized_split = dataset[split].map(
                preprocess_function,
                batched=True,
                batch_size=1000,
                remove_columns=["text"]
            )

            dataloaders[split] = DataLoader(
                tokenized_split,
                batch_size=batch_size,
                collate_fn=data_collator,
                num_workers=min(4, os.cpu_count() or 1),
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=3,
            )

    return (
        dataloaders["train"] if "train" in dataloaders else None,
        dataloaders["validation"] if "validation" in dataloaders else None,
        dataloaders["test"] if "test" in dataloaders else None
    )

# Concatenate examples function
def concatenate_examples(tokenizer, sequence_length, examples):
    # Tokenize all examples first
    all_tokens = []
    for text in examples["text"]:
        if text and len(text.strip()) > 0:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens + [tokenizer.eos_token_id])  # Add EOS between examples
    
    # Create chunks of sequence_length tokens
    concatenated_chunks = []
    for i in range(0, len(all_tokens), sequence_length):
        chunk = all_tokens[i:i + sequence_length]
        if len(chunk) == sequence_length:  # Only use complete chunks
            concatenated_chunks.append(chunk)
    
    # Convert to tensor format
    result = {
        "input_ids": concatenated_chunks,
        "attention_mask": [[1] * len(chunk) for chunk in concatenated_chunks]
    }
    return result

# Custom collator that doesn't add padding (as our sequences are already fixed length)
def collate_fn(batch):
    batch_input_ids = torch.stack([torch.tensor(item["input_ids"]) for item in batch])
    batch_attention_mask = torch.stack([torch.tensor(item["attention_mask"]) for item in batch])[:, :-1].contiguous()  # Remove last token mask
    batch_labels = batch_input_ids.clone()[:, 1:].contiguous()  # Shift input_ids to the right by 1
    batch_input_ids = batch_input_ids[:, :-1].contiguous()  # Remove last token from input_ids
    return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask, "labels": batch_labels}

def create_causal_lm_dataloaders_concatenated(tokenizer, dataset_name, dataset_config_name, batch_size, sequence_length, streaming=True):
    dataset = load_dataset(dataset_name, dataset_config_name, streaming=streaming)
    
    # Process data and create dataloaders
    dataloaders = {}
    for split in ["train", "validation", "test"]:
        if split in dataset:
            # Process in batches to maintain streaming benefits
            tokenized_split = dataset[split].map(
                partial(concatenate_examples, tokenizer, sequence_length),
                batched=True,
                batch_size=1000,  # Process 1000 examples at a time
                remove_columns=["text"]
            )
            
            dataloaders[split] = DataLoader(
                tokenized_split,
                batch_size=batch_size,
                collate_fn=collate_fn,
                num_workers=max(1, min(os.cpu_count() - 2, 8)),  # More dynamic scaling
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=3
            )
    
    return (
        dataloaders["train"] if "train" in dataloaders else None,
        dataloaders["validation"] if "validation" in dataloaders else None,
        dataloaders["test"] if "test" in dataloaders else None
    )

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    torch.set_printoptions(profile="full")

    train_loader, validation_loader, test_loader = create_causal_lm_dataloaders_concatenated(tokenizer, "wikitext", "wikitext-103-raw-v1", batch_size=8, sequence_length=1024)

    for batch in train_loader:
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        attention_mask = batch["attention_mask"]

        print(f"input_ids: {input_ids[0]}")
        print(f"labels: {labels[0]}")
        print(f"attention_mask: {attention_mask[0]}")

        break