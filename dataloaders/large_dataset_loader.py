from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, PreTrainedTokenizer

import torch

def create_causal_lm_dataloaders(tokenizer: PreTrainedTokenizer, dataset_name, dataset_config_name, batch_size, sequence_length, streaming=False):
    def preprocess_function(examples):
        examples = [example for example in examples['text'] if example.strip() != ""]
        model_inputs = tokenizer(
            examples,
            max_length=sequence_length,
            truncation=True,
            return_tensors=None,
            return_attention_mask=True,
        )
        return model_inputs

    dataset = load_dataset(dataset_name, dataset_config_name, streaming=streaming)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        return_tensors="pt",
        padding=True
    )

    dataloaders = {}
    for split in ["train", "validation", "test"]:
        if split in dataset:
            tokenized_split = dataset[split].map(
                preprocess_function,
                batched=True,
                remove_columns=dataset[split].column_names if not streaming else None
            )
            
            dataloaders[split] = DataLoader(
                tokenized_split,
                batch_size=batch_size,
                collate_fn=data_collator,
                num_workers=4,
                pin_memory=True
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

    train_loader, validation_loader, test_loader = create_causal_lm_dataloaders(tokenizer, "wikitext", "wikitext-103-raw-v1", batch_size=8, sequence_length=1024)

    for batch in train_loader:
        input_ids = batch["input_ids"][:, :-1].contiguous()
        labels = batch["labels"][:, 1:].contiguous()
        attention_mask = batch["attention_mask"][:, 1:].contiguous()

        print(f"input_ids: {input_ids[0]}")
        print(f"labels: {labels[0]}")
        print(f"attention_mask: {attention_mask[0]}")

        break