import csv
import torch.nn.functional as F
import torch
import numpy as np
from torch.utils.data import Dataset

def col_to_one_hot(col, add_titles=True):
    category_name = col[0]
    data = col[1:]
    set_of_categories = sorted(set(data))
    category_to_int_map = dict([(y, x) for x, y in enumerate(set_of_categories)])
    int_col = [category_to_int_map[x] for x in data]
    one_hot_data = F.one_hot(torch.LongTensor(int_col))
    if add_titles:
        one_hot_matrix = np.vstack((list(set_of_categories), one_hot_data))
        return one_hot_matrix
    else:
        return one_hot_data


class AirBnBDataset(Dataset):
    def __init__(self, train=True, random_seed=0):

        self.data = list()
        self.targets = list()
        with open('AB_NYC_2019.csv', encoding="utf-8") as csv_file:

            csv_reader = csv.reader(csv_file, delimiter=',')
            data_matrix = []
            for i in range(1000):
                data_matrix.append(next(csv_reader))



            data_matrix = np.array(data_matrix)

            neighbourhood_col = data_matrix[:, 4]
            neighbourhood_one_hot = col_to_one_hot(neighbourhood_col)

            room_type_col = data_matrix[:, 8]
            room_type_one_hot = col_to_one_hot(room_type_col)


            data_matrix = np.delete(data_matrix, [0, 1, 2, 3, 4, 5, 8, 12], 1)
            data_matrix = np.hstack((data_matrix, neighbourhood_one_hot))
            data_matrix = np.hstack((data_matrix, room_type_one_hot))

            print(list(enumerate(data_matrix[0])))

            np.random.seed(random_seed)
            np.random.shuffle(data_matrix)

            if train:
                data_matrix = data_matrix[0:int(len(data_matrix)*0.8)]
            else:
                data_matrix = data_matrix[int(len(data_matrix) * 0.8):]

            self.targets = torch.Tensor(data_matrix[:, 2])
            self.data = torch.Tensor(np.delete(data_matrix, 2))

    def __len__(self):
        return self.data.size()

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]



