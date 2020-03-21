import matplotlib.pyplot as plt
import numpy as np
import torchvision
import pickle

def visualise_patterns(patterns):
    f, axarr = plt.subplots(2, 5, sharex=True)
    for i in range(10):
        axarr[int(i/5), i % 5].axis('off')
        axarr[int(i/5), i % 5].imshow(patterns[i], cmap='binary')
    plt.show()


def visualise_pattern(pattern):
    img = pattern.reshape((28, 28))
    plt.axis('off')
    plt.imshow(img, cmap='binary')
    plt.show()


def train():
    means = pickle.load(open('./means.pt', 'rb'))
    net = np.zeros((784, 784))

    for i in range(784):
        print(i)
        for j in range(i + 1, 784):
            w = 0
            for image in means[0:2]:
                pattern = image.flatten()
                w += pattern[i] * pattern[j]
            net[i, j] += (w /3)
            net[j, i] += (w /3)

    pickle.dump(net, open('./net.pt', 'wb'))

def binarise(pattern):
    pattern[pattern < 90] = -1
    pattern[pattern >= 90] = 1


def evaluate():
    tr_set = torchvision.datasets.MNIST("../resources", train=True, download=True).data.numpy().astype('int')
    net = pickle.load(open('./net.pt', 'rb'))
    pattern = tr_set[1].flatten()
    binarise(pattern)
    visualise_pattern(pattern)

    for i in range(100000):
        idx = np.random.randint(0, 784)
        result_sum = sum([net[idx, j] * pattern[j] for j in range(784)])
        pattern[idx] = 1 if result_sum > 0 else -1
        # img = pattern.reshape((28, 28))
        # plt.axis('off')
        # plt.imshow(img, cmap='binary')
        # plt.savefig('./img/'+str(i)+'.png')

    visualise_pattern(pattern)


def create_tr_set():
    tr_set = torchvision.datasets.MNIST("../resources", train=True, download=True).data.numpy().astype('int')
    tr_labels = torchvision.datasets.MNIST("../resources", train=True, download=True).targets.numpy().astype('int')

    means = np.zeros((10, 28, 28))
    count = np.zeros(10)

    for i in range(len(tr_set)):
        count[tr_labels[i]] += 1
        means[tr_labels[i]] += (tr_set[i] - means[tr_labels[i]]) / count[tr_labels[i]]

    visualise_patterns(means)
    binarise(means)
    visualise_patterns(means)
    pickle.dump(means, open('./means.pt', 'wb'))


if __name__ == "__main__":

    # create_tr_set()
    # train()
    evaluate()

