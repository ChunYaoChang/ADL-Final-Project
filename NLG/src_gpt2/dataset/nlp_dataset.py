import torch
from torch.utils.data import Dataset  

class NLPDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return torch.LongTensor(self.data[idx])
