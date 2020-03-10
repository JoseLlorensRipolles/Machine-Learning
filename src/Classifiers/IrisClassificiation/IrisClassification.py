import torch
import torch.nn.functional as F
from src.Classifiers.IrisClassificiation.IrisDataset import IrisDataset
import src.Classifiers.IrisClassificiation.Architecture as Architecture
from torch.utils.data import DataLoader


def train(model, opt, train_loader, epoch):
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        opt.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, targets)
        loss.backward()
        opt.step()
        if batch_idx % 120 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, i):
    model.eval()
    test_loss = 0
    correct = 0

    for batch_idx, (data, targets) in enumerate(test_loader):
        output = model(data)
        test_loss += F.nll_loss(output, targets, reduction='sum')
        correct += torch.argmax(output, dim=1).eq(targets).sum().item()

    print('Accuracy: ', correct/len(test_loader.dataset))


def main():
    iris_trainset = IrisDataset(train=True)
    iris_testset = IrisDataset(train=False)

    train_loader = DataLoader(iris_trainset, 1000)
    test_loader = DataLoader(iris_testset, 1000)

    model = Architecture.FFNN()
    opt = torch.optim.SGD(model.parameters(), lr=0.001)

    # Only using test as val because its a v basic example
    for i in range(4000):
        train(model, opt, train_loader, i)
        test(model, test_loader, i)


if __name__ == '__main__':
    main()

