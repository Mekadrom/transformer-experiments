from torch.utils.data import Dataset
from tqdm import tqdm

import torch

class SingleTextFileDataset(Dataset):
    def __init__(self, tokenizer, examples, seq_length, pad_token_id):
        self.seq_length = seq_length
        self.pad_token_id = pad_token_id

        self.examples = []
        for example in tqdm(examples, desc="Tokenizing examples"):
            if example.strip() == "":
                continue
            seq = tokenizer.encode(example) + [tokenizer.eos_token_id]
            self.examples.extend(seq)

        print(f"Number of examples: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        input_ids = torch.tensor(self.examples[idx:idx+self.seq_length])
        labels = torch.tensor(self.examples[idx+1:idx+self.seq_length+1])
        return input_ids, labels
