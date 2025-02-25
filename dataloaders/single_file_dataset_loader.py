from torch.utils.data import DataLoader

import single_text_file_dataset
import requests

class SingleFileDatasetLoader:
    def load_data(self, tokenizer, maxlen, batch_size):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        text = response.text

        dataset = single_text_file_dataset.SingleTextFileDataset(tokenizer.encode, text, seq_length=maxlen, pad_token_id=tokenizer.pad_token_id)

        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return train_loader, None, None
