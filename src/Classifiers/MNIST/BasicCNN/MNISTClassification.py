from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import Classifiers.MNIST.BasicCNN.Architecture as Architecture
import torch
import torchvision
from torch.utils.data.dataloader import DataLoader


def train(device, model, optimizer, scheduler, criterion, tr_loader, val_loader):
    tr_losses = []
    val_losses = []
    best_val = 9999
    best_val_epoch = -1
    epoch = -1
    while epoch - best_val_epoch < 5:
        epoch += 1
        running_tr_loss = 0
        running_val_loss = 0

        model.train()
        for data, targets in tr_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss = criterion(output, targets)
            running_tr_loss += loss.item()
            loss.backward()
            optimizer.step()

        model.eval()
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss = criterion(output, targets)
            running_val_loss += loss.item()

        tr_loss = running_tr_loss/len(tr_loader)
        val_loss = running_val_loss/len(val_loader)

        if val_loss < best_val:
            torch.save(model.state_dict(), "./model")
            best_val = val_loss
            best_val_epoch = epoch

        print('Train Epoch: {} \tLoss: {:.6f}\tVal: {:.6f}'.format(
            epoch, tr_loss, val_loss))
        tr_losses.append(tr_loss)
        val_losses.append(val_loss)
        scheduler.step()

    plt.plot(tr_losses)
    plt.plot(val_losses)
    plt.show()


def test(device,  ts_loader):
    model = Architecture.CNN().to(device)
    model.load_state_dict(torch.load("./model"))
    correct = 0

    model.eval()
    for batch_idx, (data, targets) in enumerate(ts_loader):
        data = data.to(device)
        targets = targets.to(device)
        output = model(data)
        correct += torch.argmax(output, dim=1).eq(targets).sum().item()

    print('Accuracy: ', correct/len(ts_loader.dataset))


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = Architecture.CNN().to(device)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=0.0000001,  momentum=0.2)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tr_val_set = torchvision.datasets.MNIST("../resources", train=True, download=True, transform=transforms)
    tr_set, val_set = torch.utils.data.random_split(tr_val_set, [55000, 5000])
    tr_loader = DataLoader(tr_set, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=128, shuffle=True, drop_last=True)

    ts_set = torchvision.datasets.MNIST("../resources", train=False, download=True, transform=transforms)
    ts_loader = DataLoader(ts_set, batch_size=1000, shuffle=True)

    train(device, model, optimizer, scheduler, criterion, tr_loader, val_loader)
    test(device, ts_loader)

