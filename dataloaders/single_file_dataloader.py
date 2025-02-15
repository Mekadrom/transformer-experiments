from torch.utils.data import DataLoader

import custom_datasets
import requests

class TinyShakespeareDataLoader:
    def load_data(self, tokenizer, maxlen, batch_size):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        text = response.text

        dataset = custom_datasets.SingleTextFileDataset(tokenizer.encode, text, seq_length=maxlen)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return train_loader, None, None
