import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.Regression.HousePricesKaggle import Architecture
from src.Regression.HousePricesKaggle.HousePricesDataset import HousePricesDataset


def train(tr_loader, vl_loader):
    input_length = tr_loader.dataset.dataset.data.shape[1]
    device = torch.device("cuda")
    model = Architecture.FFNN(input_length).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()
    tr_losses = list()
    val_losses = list()
    best_validation = np.inf
    best_validation_epoch = -1
    epoch = -1

    while epoch - best_validation_epoch < 200:
        epoch = epoch + 1
        running_tr_loss = 0
        running_vl_loss = 0

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

        model.eval()
        for val_data, val_targets in vl_loader:
            val_data = val_data.to(device)
            val_targets = val_targets.to(device)
            val_output = model(val_data)
            running_vl_loss += criterion(val_output, val_targets).item()

        tr_loss = running_tr_loss / len(tr_loader)
        vl_loss = running_vl_loss / len(vl_loader)

        if vl_loss < best_validation:
            torch.save(model.state_dict(), "./resources/model")
            best_validation = vl_loss
            best_validation_epoch = epoch

        print('Train Epoch: {} \tLoss: {:.6f}\tVal: {:.6f}'.format(
            epoch, tr_loss, vl_loss))
        tr_losses.append(tr_loss)
        val_losses.append(vl_loss)

    print("Min val:", np.min(val_losses))
    plt.semilogy(tr_losses, label="Tr")
    plt.semilogy(val_losses, label="Val")
    plt.legend()
    plt.show()



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
    lengths = [int(len(set) * 0.8), int(len(set) * 0.2)]
    tr_set, val_set = torch.utils.data.random_split(set, lengths)
    ts_set = HousePricesDataset(train=False)

    tr_loader = DataLoader(tr_set, 2000, shuffle=True)
    vl_loader = DataLoader(val_set, 2000, shuffle=True)
    ts_loader = DataLoader(ts_set, 2000)

    output = np.zeros((ts_set.data.shape[0], 1))
    for i in range(5):
        train(tr_loader, vl_loader)
        output += test(ts_loader)
    output = output/5
    submission = "Id,SalePrice\n"
    for idx, output in enumerate(output):
        submission += str(idx+1461)+","+str(output[0])+" \n"
    submission = submission[:-3]
    f = open("./resources/submission.csv", "w")
    f.write(submission)
    f.close()


if __name__ == '__main__':
    main()

