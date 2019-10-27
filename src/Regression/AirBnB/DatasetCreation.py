import csv
import torch.nn.functional as F
import torch
import numpy as np

with open('AB_NYC_2019.csv', encoding="utf-8") as csv_file:

    csv_reader = csv.reader(csv_file, delimiter=',')
    data_matrix = []
    for i in range(1000):
        data_matrix.append(next(csv_reader))

    print(list(enumerate(data_matrix[0])))

    data_matrix = np.array(data_matrix)


    neighbourhood = data_matrix[1:, 4]
    set_of_neighbourhoods = sorted(set(neighbourhood))
    neighbourhood_to_int_map = dict([(y, x) for x, y in enumerate(set_of_neighbourhoods)])
    neighbourhood = [neighbourhood_to_int_map[x] for x in neighbourhood]
    neighbourhood = F.one_hot(torch.LongTensor(neighbourhood))
    neighbourhood = np.vstack((list(set_of_neighbourhoods), neighbourhood))

    room_type_col = data_matrix[1:,  8]
    room_type_set = list(sorted(set(room_type_col)))
    room_type_to_int_map = dict([(y, x) for x, y in enumerate(room_type_set)])
    room_type_int_col = [room_type_to_int_map[x] for x in room_type_col]
    room_type_one_hot = F.one_hot(torch.LongTensor(room_type_int_col))
    room_type = np.vstack((room_type_set, room_type_one_hot))

    data_matrix = np.delete(data_matrix, [0, 1, 2, 3, 4, 5, 8, 12], 1)
    data_matrix = np.hstack((data_matrix, neighbourhood))
    data_matrix = np.hstack((data_matrix, room_type))

    print(data_matrix[:2])
