import torch.nn as nn
import torch.nn.functional as F


class FFNN(nn.Module):

    def __init__(self):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(275, 300)
        self.fc2 = nn.Linear(300, 500)
        self.fc3 = nn.Linear(500, 500)
        self.fc4 = nn.Linear(500, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
