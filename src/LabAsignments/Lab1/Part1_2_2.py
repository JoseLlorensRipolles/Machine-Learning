import numpy as np
import random

def phi(x):
    x = np.multiply(x,-1)
    return 2/(1+np.exp(x))-1

hidden = 20
epoch_count = 4000
D = 3
N = 200
eta = 0.02
alpha = 0.9
mu1 = [-0.5, 1]
mu2 = [0.5, -1]
sigma = [[0.2, 0.2], [0.2, 1]]
R1 = np.random.multivariate_normal(mu1, sigma, 100).tolist()
R2 = np.random.multivariate_normal(mu2, sigma, 100).tolist()

R1C = []
for point in R1:
    R1C.append(point + [1, 1])

R2C = []
for point in R2:
    R2C.append(point + [1, -1])
R = R1C + R2C
random.shuffle(R)
R = np.array(R)
T = R[:, 3]
R = R[:, 0:3]
R = np.transpose(R)


W = np.random.normal(0, 0.1, (hidden, D))
V = np.random.normal(0, 0.1, (1, hidden+1))

DW = np.zeros((hidden, D))
DV = np.zeros((1, hidden+1))

for i in range(epoch_count):

    # Forward pass
    HIN = np.matmul(W, R)
    HOUT = np.vstack((phi(HIN), np.ones(N)))

    OIN = np.matmul(V, HOUT)
    OUT = phi(OIN)

    # Backward Pass
    DELTA_O = np.multiply((OUT - T), (np.multiply((1+OUT), (1-OUT))*0.5))
    DELTA_H = np.multiply(np.matmul(V.reshape((21, 1)), DELTA_O), (np.multiply((1+HOUT), (1-HOUT))*0.5))
    DELTA_H = DELTA_H[:hidden, :]

    # WEIGHT UPDATE
    DW = np.multiply(DW, alpha) - np.multiply(np.matmul(DELTA_H, np.transpose(R)), (1-alpha))
    DV = np.multiply(DV, alpha) - np.multiply(np.matmul(DELTA_O, np.transpose(HOUT)), (1-alpha))

    DW = np.multiply(DW, eta)
    DV = np.multiply(DV, eta)

    W = np.add(W, DW)
    V = np.add(V, DV)

    err = 0
    for i in range(N):
        if OUT[0][i] < 0:
            if T[i] == 1:
                err += 1
        else:
            if T[i] == -1:
                err += 1
    print(err)


