import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):

    def __init__(self, input_length):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_length, 1500)
        self.fc2 = nn.Linear(1500, 600)
        self.fc3 = nn.Linear(600, 200)
        self.fc4 = nn.Linear(200, 1)

        self.dropout = nn.Dropout(0.1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
