import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.Regression.HousePricesKaggle import Architecture
from src.Regression.HousePricesKaggle.HousePricesDataset import HousePricesDataset


def train(device, model, opt, criterion, tr_loader, vl_loader):
    tr_losses = list()
    val_losses = list()
    best_validation = np.inf
    best_validation_epoch = -1
    epoch = -1

    while epoch - best_validation_epoch < 1000:
        epoch = epoch + 1
        model.train()
        for data, targets in tr_loader:
            data = data.to(device)
            targets = targets.to(device)
            opt.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = 0
        for val_data, val_targets in vl_loader:
            val_data = val_data.to(device)
            val_targets = val_targets.to(device)
            val_output = model(val_data)
            val_loss += criterion(val_output, val_targets).item()
            if val_loss < best_validation:
                torch.save(model.state_dict(), "./resources/model")
                best_validation = val_loss
                best_validation_epoch = epoch
        print('Train Epoch: {} \tLoss: {:.6f}\tVal: {:.6f}'.format(
            epoch, loss.item()/100000000, val_loss/100000000))
        tr_losses.append(loss.item())
        val_losses.append(val_loss)

    return tr_losses, val_losses


def test(device, test_loader, input_length):
    model = Architecture.FFNN(input_length).to(device)
    model.load_state_dict(torch.load("./resources/model"))
    for data, targets in test_loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data).tolist()
    submission = "Id,SalePrice\n"
    for idx, output in enumerate(output):
        submission += str(idx+1461)+","+str(output[0])+" \n"
    submission = submission[:-3]
    f = open("./resources/submission.csv", "w")
    f.write(submission)
    f.close()



def main():
    set = HousePricesDataset(train=True)
    lengths = [int(len(set) * 0.8), int(len(set) * 0.2)]
    tr_set, val_set = torch.utils.data.random_split(set, lengths)
    ts_set = HousePricesDataset(train=False)

    tr_loader = DataLoader(tr_set, 2000, shuffle=True)
    vl_loader = DataLoader(val_set, 2000, shuffle=True)
    ts_loader = DataLoader(ts_set, 2000)
    input_length = set.data.shape[1]

    device = torch.device("cuda")
    model = Architecture.FFNN(input_length).to(device)
    # opt = torch.optim.Adam(model.parameters(), weight_decay=0.2)
    opt = torch.optim.SGD(model.parameters(), lr=0.00000001, momentum=0.9)
    criterion = torch.nn.MSELoss()
    tr_losses, val_losses = train(device, model, opt, criterion, tr_loader, vl_loader)

    test(device, ts_loader, input_length)
    print("Min val:", np.min(val_losses)/100000000)
    plt.semilogy(tr_losses, label="Tr")
    plt.semilogy(val_losses, label="Val")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()

