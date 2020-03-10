import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):

    def __init__(self, input_length):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_length, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 50)
        self.fc4 = nn.Linear(50, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
