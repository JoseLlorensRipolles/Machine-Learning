from torch.utils.data import Dataset
import torch
from torchvision.transforms import ToTensor
import numpy as np

class IrisDataset(Dataset):
    def __init__(self, train=True, random_seed=0):
        f = open('./resources/iris.data', 'r')
        self.samples = list()
        lines = f.read().splitlines()
        lines = list(filter(None, lines))
        np.random.seed(random_seed)
        np.random.shuffle(lines)
        len_dataset = len(lines)
        if train:
            data_lines = lines[0:int(len_dataset*0.8)]
        else:
            data_lines = lines[int(len_dataset*0.8):]

        for line in data_lines:
            if len(line) != 0:
                line_items = line.split(',')
                data = line_items[:4]
                data = list(map(float, data))
                target_text = line_items[4]

                if 'setosa' in target_text:
                    target = 0
                elif 'versicolor' in target_text:
                    target = 1
                else:
                    target = 2
                self.samples.append((torch.Tensor(data), target))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == '__main__':
    dataset = IrisDataset(train=False)
    print(len(dataset))
    print(dataset[0])
