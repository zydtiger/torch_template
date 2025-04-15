import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class CustomDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        # All columns except last
        self.features = self.data.iloc[:, :-1].values
        # Last column is label
        self.labels = self.data.iloc[:, -1].values

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.FloatTensor([self.labels[idx]])
        return features, label


def get_train_loader(batch_size=32):
    dataset = CustomDataset("./data/train.csv")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_val_loader(batch_size=32):
    dataset = CustomDataset("./data/validate.csv")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
