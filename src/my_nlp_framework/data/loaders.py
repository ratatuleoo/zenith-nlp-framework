# src/my_nlp_framework/data/loaders.py
import pandas as pd
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]
        tokens = self.tokenizer.tokenize(text)
        return tokens
