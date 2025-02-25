from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

import single_text_file_dataset

def make_dataloader(dataset_name, dataset_config_name, tokenizer, batch_size, maxlen, split='train'):
    dataset = load_dataset(dataset_name, dataset_config_name, split=split)

    all_text = []
    for example in tqdm(dataset, desc=f"Loading {split} dataset"):
        all_text.append(example['text'])

    dataset = single_text_file_dataset.SingleTextFileDataset(tokenizer, all_text, seq_length=maxlen, pad_token_id=tokenizer.pad_token_id)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def create_causal_lm_dataloaders(dataset_name, dataset_config_name, tokenizer, batch_size, maxlen):
    train_loader = make_dataloader(dataset_name, dataset_config_name, tokenizer, batch_size, maxlen)
    validation_loader = make_dataloader(dataset_name, dataset_config_name, tokenizer, batch_size, maxlen, split='validation')
    test_loader = make_dataloader(dataset_name, dataset_config_name, tokenizer, batch_size, maxlen, split='test')
    return train_loader, validation_loader, test_loader
