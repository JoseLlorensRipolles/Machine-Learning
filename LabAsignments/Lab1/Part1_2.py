import numpy as np
import matplotlib.pyplot as plt
import random


def phi(x):
    x = np.multiply(x,-1)
    return 2/(1+np.exp(x))-1


def phiprime(x):
    x = np.array(x)
    z = np.multiply((1+x),(1-x))*0.5
    pass
    return z


def calculate_summed_input_signal(W,patterns):
    return np.matmul(W,patterns)


def calculate_output_signal(Hprime):
    return phi(Hprime)


n_epoch = 40000
n_hidden = 20
dimensionality = 2
eta = 0.02
alpha = 0.9
mu1 = [-0.5, 1]
mu2 = [0.5, -1]
sigma = [[0.2, 0.2], [0.2, 1]]
random_dataset1 = np.random.multivariate_normal(mu1, sigma, 100).tolist()
random_dataset2 = np.random.multivariate_normal(mu2, sigma, 100).tolist()

objects = []


for point in random_dataset1:
    objects.append([1] + point + [1])

for point in random_dataset2:
    objects.append([1] + point + [-1])



random.shuffle(objects)

objects = np.array(objects).transpose()
patterns = objects[0:dimensionality+1, :]
labels = objects[dimensionality+1:, :][0]
input_size = len(labels)

def show_points(patterns, labels):
    for i in range(input_size):
        if labels[i] == 1:
            plt.plot(patterns[1][i], patterns[2][i], 'ro')
        else:
            plt.plot(patterns[1][i], patterns[2][i], 'go')
    plt.show()


show_points(patterns, labels)

W = np.random.normal(0, 0.1, (n_hidden, dimensionality+1))
V = np.random.normal(0, 0.1, (1, n_hidden+1))

v_variation = np.zeros(n_hidden+1)
w_variation = np.zeros((n_hidden, dimensionality+1))


def calculate_error(output_layer_output):
    err = 0
    for i in range(input_size):
        if output_layer_output[0][i] < 0:
            if labels[i] == 1:
                err += 1
        else:
            if labels[i] == -1:
                err += 1
    print(err)



for i in range(n_epoch):

    #Forwarding signal

    hin = phi(np.matmul(W, patterns))
    hout = np.vstack((hin, np.ones(input_size)))

    out = phi(np.matmul(V, hout))

    calculate_error(out)


    #Backpropagation of the error

    delta_o = np.multiply((out - labels), phiprime(out))

    phiderivative = phiprime(hout)
    aux = np.matmul(V.reshape(n_hidden+1,1), delta_o.reshape(1, input_size))
    delta_h = np.multiply(aux, phiderivative)
    delta_h = delta_h[0:n_hidden, :]

    #Weight update
    w_variation = np.multiply(w_variation, alpha) - np.multiply(np.matmul(delta_h, np.transpose(patterns)), (1-alpha))
    w_variation = np.multiply(w_variation, eta)

    v_variation = np.multiply(v_variation, alpha) - np.multiply(np.matmul(delta_o, np.transpose(hout)), (1-alpha))
    v_variation = np.multiply(v_variation, eta)

    W = np.add(W, w_variation)
    V = np.add(V, v_variation)



