from datasets import load_dataset
from torch.utils.data import DataLoader

import custom_datasets
import requests
import torch

class CustomDataLoader:
    def load_data(self, tokenizer, maxlen, batch_size):
        raise NotImplementedError
    
class TinyShakespeareDataLoader(CustomDataLoader):
    def load_data(self, tokenizer, maxlen, batch_size):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        text = response.text

        dataset = custom_datasets.SingleTextFileDataset(tokenizer.encode, text, seq_length=maxlen)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return train_loader, None, None

class DefaultDataLoader(CustomDataLoader):
    def __init__(self, path, name):
        self.path = path
        self.name = name

    def load_split(self, tokenizer, maxlen, batch_size, split):
        dataset = load_dataset(self.path, self.name)[split]
    
        # Concatenate all texts and tokenize
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=False)
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            remove_columns=dataset.column_names,
            batched=True
        )
        
        # Concatenate all tokenized texts
        concat_ids = []
        for example in tokenized_dataset["input_ids"]:
            concat_ids.extend(example)

        # Split into input and label chunks
        input_chunks = []
        label_chunks = []
        for i in range(0, len(concat_ids) - maxlen, maxlen):
            input_chunks.append(torch.tensor(concat_ids[i:i + maxlen]))
            label_chunks.append(torch.tensor(concat_ids[i + 1:i + 1 + maxlen]))

        dataset = torch.utils.data.TensorDataset(torch.stack(input_chunks), torch.stack(label_chunks))

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == 'train'
        )
        
        return dataloader

    def load_data(self, tokenizer, maxlen, batch_size):
        train_loader = self.load_split(tokenizer, maxlen, batch_size, "train")
        val_loader = self.load_split(tokenizer, maxlen, batch_size, "validation")
        test_loader = self.load_split(tokenizer, maxlen, batch_size, "test")
        return train_loader, val_loader, test_loader
