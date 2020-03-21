import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.Regression.HousePricesKaggle import Architecture
from src.Regression.HousePricesKaggle.HousePricesDataset import HousePricesDataset


def train(tr_loader):
    input_length = tr_loader.dataset.data.shape[1]
    device = torch.device("cuda")
    model = Architecture.FFNN(input_length).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()

    epoch = -1

    while epoch < 1200:
        epoch = epoch + 1
        running_tr_loss = 0

        model.train()
        for data, targets in tr_loader:
            data = data.to(device)
            targets = targets.to(device)
            opt.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            running_tr_loss += loss.item()
            loss.backward()
            opt.step()
        print("Epoch{}: \t {}".format(epoch, running_tr_loss))
    torch.save(model.state_dict(), "./resources/model")


def test(test_loader):
    input_length = test_loader.dataset.data.shape[1]
    device = torch.device("cuda")
    model = Architecture.FFNN(input_length).to(device)
    model.load_state_dict(torch.load("./resources/model"))
    data = HousePricesDataset(train=False).data
    data = data.to(device)
    output = model(data).cpu().detach().numpy()

    mean = 12.024057394918406
    std = 0.39931245219387496
    output = output*std+mean
    return np.expm1(output)




def main():
    set = HousePricesDataset(train=True)
    ts_set = HousePricesDataset(train=False)

    tr_loader = DataLoader(set, 2000, shuffle=True)
    ts_loader = DataLoader(ts_set, 2000)

    output = np.zeros((ts_set.data.shape[0], 1))
    for i in range(10):
        train(tr_loader)
        output += test(ts_loader)
    output = output/10
    submission = "Id,SalePrice\n"
    for idx, output in enumerate(output):
        submission += str(idx+1461)+","+str(output[0])+" \n"
    submission = submission[:-3]
    f = open("./resources/submission.csv", "w")
    f.write(submission)
    f.close()


if __name__ == '__main__':
    main()

