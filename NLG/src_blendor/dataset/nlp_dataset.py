import torch
from torch.utils.data import Dataset  

class NLPDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data['input_ids'])
    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for (key, value) in self.data.items()}
        return item