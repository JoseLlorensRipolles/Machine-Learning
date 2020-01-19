import time

import torch
import torch.nn.functional as F
from src.Regression.AirBnB.AirBnBDataset import AirBnBDataset
import src.Regression.AirBnB.Architecture as Architecture
from torch.utils.data import DataLoader


def train(device, model, opt, criterion, train_loader, epoch):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        opt.step()

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
               100. * batch_idx / len(train_loader), loss.item()))


def test(device, model, criterion, test_loader, i):
    model.eval()
    test_loss = 0
    correct = 0

    for batch_idx, (data, targets) in enumerate(test_loader):
        data, targets = data.to(device), targets.to(device)
        output = model(data)
        test_loss += criterion(output, targets)
        correct += torch.argmax(output, dim=1).eq(targets).sum().item()

def main():
    tr_set = AirBnBDataset(train=True)
    ts_set = AirBnBDataset(train=False)

    train_loader = DataLoader(tr_set, 2)
    test_loader = DataLoader(ts_set, 2)

    device = torch.device("cuda")
    model = Architecture.FFNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss(reduction='mean')

    for i in range(1000):
        train(device, model, opt, criterion, train_loader, i)
        test(device, model, criterion, test_loader, i)


if __name__ == '__main__':
    main()

