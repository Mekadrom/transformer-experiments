from torch.utils.data import Dataset

import torch

class SingleTextFileDataset(Dataset):
    def __init__(self, tokenizer_encode_fn, text, seq_length):
        self.seq_length = seq_length
        
        self.encoded_text = tokenizer_encode_fn(text)
        
    def __len__(self):
        return len(self.encoded_text) - self.seq_length
        
    def __getitem__(self, idx):
        sequence = self.encoded_text[idx:idx + self.seq_length]
        target = self.encoded_text[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(sequence), torch.tensor(target)
