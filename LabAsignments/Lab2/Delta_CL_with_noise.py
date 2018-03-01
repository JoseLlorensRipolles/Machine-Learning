import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random

def shuffle_data(x, y):
    training = np.vstack((x, y))
    training = np.transpose(training)
    np.random.shuffle(training)

    x = training[:, 0]
    y = training[:, 1]
    return x, y

n = 10
sigma = 0.5
n_epoch = 100
eta = 0.1
eta_CL = 0.1
iterations_CL = 40000


x = np.arange(0, 2*math.pi+0.05, 0.1)
f1 = [math.sin(x) + np.random.normal(scale=0.1) for x in x]
rbf_mus = np.random.normal(loc=-math.pi, size=n)

plt.plot(x, f1)
plt.plot(rbf_mus, np.zeros(n), 'ro')
# CL for RBF initialisation

for iteration in range(iterations_CL):
    input_point = np.random.randint(0, len(x))
    f = f1[input_point]

    phi = []
    for rbf in range(n):
        phi.append(mlab.normpdf(x[input_point], rbf_mus[rbf], sigma))

    rbf_to_update = phi.index(max(phi))
    distance = x[input_point] - rbf_mus[rbf_to_update]
    rbf_mus[rbf_to_update] += eta_CL * distance
    for rbf in range(n):
        if rbf != rbf_to_update:
            rbf_mus += eta_CL/5 * distance


plt.plot(rbf_mus, np.zeros(n), 'bo')

# Training pass with delta rule.

x, f1 = shuffle_data(x, f1)
w = np.random.normal(size=n)



for epoch in range(n_epoch):
    for i in range(len(x)):
        x_point = x[i]
        phi = []
        for rbf in range(n):
            phi.append(mlab.normpdf(x_point, rbf_mus[rbf], sigma))
        out = np.matmul(phi, w)
        e = f1[i] - out
        variation = np.multiply(eta * e, phi)
        w += variation





test_points = np.arange(0.05, 2*math.pi+0.05, 0.1)

phi = []
for pattern in test_points:
    new_row = []
    for rbf in range(n):
        new_row.append(mlab.normpdf(pattern, rbf_mus[rbf], sigma))
    phi.append(new_row)
phi = np.array(phi)

output = np.matmul(phi, w)

e1 = 0

for point_index in range(len(test_points)):
    expected_output = math.sin(test_points[point_index]) + np.random.normal(scale=0.1)
    e1 += abs(expected_output - output[point_index])

e1 = e1/len(test_points)

plt.plot(test_points, output)
plt.text(3.3, 0.8, 'Error : ' + str(e1))
plt.text(0.3, -0.8, 'n: ' + str(n)+'\nN: ' + str(len(x)) + '\nSigma: '+str(sigma)+'\nEta: '+str(eta)+'\nEpochs: '+str(n_epoch)+'\nCL_iterations: '+str(iterations_CL)+'\nEta CL: '+str(eta_CL))


plt.show()
