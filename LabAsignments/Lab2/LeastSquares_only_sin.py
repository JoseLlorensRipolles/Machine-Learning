import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

# Hyperparameters
n = 5
sigma = 2


# Initialisation of the patterns
x = np.arange(0, 2*math.pi+0.05, 0.1)
f1 = [math.sin(x) for x in x]


# Initialisation of the RBF centers
rbf_mus = [x for x in np.linspace(0, 2*math.pi, num=n, endpoint=True)]


# Ploting of the subyacent functions.
plt.plot(x, f1)
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
wf1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(phi), phi)), np.transpose(phi)), f1)

# Testing pass

test_points = np.arange(0.05, 2*math.pi+0.05, 0.1)
phi = []
for pattern in test_points:
    new_row = []
    for rbf in range(n):
        new_row.append(mlab.normpdf(pattern, rbf_mus[rbf], sigma))
    phi.append(new_row)
phi = np.array(phi)


outputf1 = np.matmul(phi, wf1)

e1 = 0

for point_index in range(len(test_points)):
    e1 += abs(math.sin(test_points[point_index]) - outputf1[point_index])

e1 = e1/len(test_points)


plt.plot(test_points, outputf1)
plt.text(3.3, 0.8, 'Error : ' + str(e1))
plt.text(0.3, -0.8, 'n: ' + str(n)+'\nN: ' + str(len(x)) + '\nsigma: '+str(sigma))



plt.show()
