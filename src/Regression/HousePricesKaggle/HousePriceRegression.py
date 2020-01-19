import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.Regression.HousePricesKaggle.HousePricesDataset import HousePricesDataset
from src.Regression.HousePricesKaggle import Architecture
import matplotlib.pyplot as plt
import numpy as np


def train(device, model, opt, criterion, tr_loader, vl_loader, epoch, tr_losses, val_losses, best_validation):
    model.train()
    for batch_idx, (data, targets) in enumerate(tr_loader):
        data = data.to(device)
        targets = targets.to(device)
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        opt.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            val_loss = 0
            for val_data, val_targets in vl_loader:
                val_data = val_data.to(device)
                val_targets = val_targets.to(device)
                val_output = model(val_data)
                val_loss += criterion(val_output, val_targets).item()
                if val_loss < best_validation:
                    torch.save(model.state_dict(), "./resources/model")
                    best_validation = val_loss
            print('Train Epoch: {} \tLoss: {:.6f}\tVal: {:.6f}'.format(
                epoch, loss.item(), val_loss))
            tr_losses.append(loss.item())
            val_losses.append(val_loss)



def test(device, criterion, test_loader):
    model = Architecture.FFNN().to(device)
    model.load_state_dict(torch.load("./resources/model"))
    test_loss = 0
    correct = 0

    for batch_idx, (data, targets) in enumerate(test_loader):
        data, targets = data.to(device), targets.to(device)
        output = model(data)
    print(enumerate(output))

def main():
    set = HousePricesDataset(train=True)
    lengths = [int(len(set) * 0.8), int(len(set) * 0.2)]
    tr_set, val_set = torch.utils.data.random_split(set, lengths)
    ts_set = HousePricesDataset(train=False)

    tr_loader = DataLoader(tr_set, 2000)
    val_loader = DataLoader(val_set, 2000)
    ts_loader = DataLoader(ts_set, 2000)

    device = torch.device("cuda")
    model = Architecture.FFNN().to(device)

    opt = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    tr_losses = list()
    val_losses = list()
    best_validation = np.inf

    for i in range(50):
        train(device, model, opt, criterion, tr_loader, val_loader, i, tr_losses, val_losses, best_validation)
    test(device, criterion, ts_loader)
    plt.semilogy(tr_losses, label="Tr")
    plt.semilogy(val_losses, label="Val")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

