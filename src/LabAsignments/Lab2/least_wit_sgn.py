import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Hyperparameters
n = 2
sigma = 2


# Initialisation of the patterns
x = np.arange(0, 2*math.pi+0.05, 0.1)
f2 = [np.sign(math.sin(x)) for x in x]


# Initialisation of the RBF centers
rbf_mus = [x for x in np.linspace(0, 2*math.pi, num=n, endpoint=True)]


# Ploting of the subyacent functions.

plt.plot(x, f2)
plt.plot(rbf_mus, np.zeros(n), 'bo')


# Forward pass

# RBF output calculus
phi = []
for point in x:
    new_row = []
    for rbf in range(n):
        new_row.append(mlab.normpdf(point, rbf_mus[rbf], sigma))
    phi.append(new_row)
phi = np.array(phi)

# Weight update with least squares
transposed_phi = np.transpose(phi)
wf2 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi), phi)), np.transpose(phi)), f2)

# Testing pass

test_points = np.arange(0.05, 2*math.pi+0.05, 0.1)
phi = []
for pattern in test_points:
    new_row = []
    for rbf in range(n):
        new_row.append(mlab.normpdf(pattern, rbf_mus[rbf], sigma))
    phi.append(new_row)
phi = np.array(phi)



outputf2 = np.matmul(phi, wf2)


e2 = 0

for point_index in range(len(test_points)):
    e2 += abs(np.sign(math.sin(test_points[point_index])) - outputf2[point_index])


e2 = e2/len(test_points)



plt.plot(test_points, outputf2)
plt.text(3.3, 0.8, 'Error : '+ str(e2))
plt.text(0.3, -1, 'n: ' + str(n)+'\nN: ' + str(len(x)) + '\nsigma: '+str(sigma))

plt.show()
