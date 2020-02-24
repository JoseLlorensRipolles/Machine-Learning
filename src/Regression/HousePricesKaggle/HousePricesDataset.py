import torch
from torch.utils.data import Dataset


class HousePricesDataset(Dataset):
    def __init__(self, train=True):

        self.data = list()
        self.targets = list()

        if train:
            self.data = torch.load('resources/tr_data.pt')
            self.targets = torch.load('resources/tr_targets.pt')
        else:
            self.data = torch.load('resources/ts_data.pt')
            self.targets = torch.load('resources/ts_targets.pt')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


if __name__ == '__main__':
    tr_set = HousePricesDataset(train=True)
    print(len(tr_set))
