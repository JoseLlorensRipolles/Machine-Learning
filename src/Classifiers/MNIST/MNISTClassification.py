from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import src.Classifiers.MNIST.Architecture as Architecture
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
        epoch_tr_loss = 0
        epoch_val_loss = 0

        for data, targets in tr_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss = criterion(output, targets)
            epoch_tr_loss += loss.item()
            loss.backward()
            optimizer.step()

        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss = criterion(output, targets)
            epoch_val_loss += loss.item()

        if epoch_val_loss < best_val:
            torch.save(model.state_dict(), "./resources/model")
            best_val = epoch_val_loss
            best_val_epoch = epoch

        print('Train Epoch: {} \tLoss: {:.6f}\tVal: {:.6f}'.format(
            epoch, epoch_tr_loss/len(tr_loader.dataset), epoch_val_loss/len(val_loader.dataset)))
        tr_losses.append(epoch_tr_loss/len(tr_loader.dataset))
        val_losses.append(epoch_val_loss/len(val_loader.dataset))
        # scheduler.step()

    plt.plot(tr_losses)
    plt.plot(val_losses)
    plt.show()


def test(device,  ts_loader):
    model = Architecture.CNN().to(device)
    model.load_state_dict(torch.load("./resources/model"))
    correct = 0

    for batch_idx, (data, targets) in enumerate(ts_loader):
        data = data.to(device)
        targets = targets.to(device)
        output = model(data)
        correct += torch.argmax(output, dim=1).eq(targets).sum().item()

    print('Accuracy: ', correct/len(ts_loader.dataset))


if __name__ == '__main__':
    device = torch.device("cuda:0")
    model = Architecture.CNN().to(device)
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=1)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.00001)
    # optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    tr_val_set = torchvision.datasets.MNIST("./resources", train=True, download=True, transform=transforms)
    tr_set, val_set = torch.utils.data.random_split(tr_val_set, [50000, 10000])
    tr_loader = DataLoader(tr_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=True)

    ts_set = torchvision.datasets.MNIST("./resources", train=False, download=True, transform=transforms)
    ts_loader = DataLoader(ts_set, batch_size=1000, shuffle=True)

    train(device, model, optimizer, scheduler, criterion, tr_loader, val_loader)
    test(device, ts_loader)

