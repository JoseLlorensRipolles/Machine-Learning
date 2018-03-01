import numpy as np
from scipy.stats import multivariate_normal
from scipy.spatial import distance
import matplotlib.pyplot as plt

def read_file(filepath):
    with open(filepath) as f:
        read_data = f.read()
        m = []
        for line in read_data.splitlines():
            new_row = []
            for number in line.split():
                new_row.append(float(number))
            m.append(new_row)
        return m


ballist = np.array(read_file('data_lab2/ballist.dat'))
np.random.shuffle(ballist)
input_points = ballist[:, 0:2]
output_points = ballist[:, 2:4]

balltest = np.array(read_file('data_lab2/ballist.dat'))
test_input_points = balltest[:,0:2]
test_output_points = balltest[:,2:4]

# Input plotting
x_input = input_points[:, 0]
y_input = input_points[:, 1]
plt.plot(x_input, y_input, 'bo')



n = 10
sigma = 0.05
n_epoch = 100
eta = 0.1
eta_CL = 0.5
iterations_CL = 400

rbfs_means = []
for i in range(n+1):
    for j in range(n+1):
        rbfs_means.append([i/n, j/n])
        plt.plot(i/n, j/n, 'ro')

plt.show()


for iteration in range(iterations_CL):
    input_index = np.random.randint(0, len(input_points))
    input_point = input_points[input_index]

    phi = []
    for rbf in range(len(rbfs_means)):
        phi.append(multivariate_normal(mean=rbfs_means[rbf], cov=sigma).pdf(input_point))

    rbf_to_update = phi.index(max(phi))
    distance = input_point - rbfs_means[rbf_to_update]
    rbfs_means[rbf_to_update] += eta_CL * distance

for rbf in rbfs_means:
    plt.plot(rbf[0], rbf[1], 'ro')
plt.plot(x_input, y_input, 'bo')
plt.show()

w1 = np.random.normal(size=(n+1)*(n+1))
w2 = np.random.normal(size=(n+1)*(n+1))

w = np.vstack((w1, w2))

rbfs = []
for rbf_mean in rbfs_means:
    rbfs.append(multivariate_normal(rbf_mean))


for epoch in range(n_epoch):
    for i in range(len(input_points)):
        point = input_points[i]
        phi = []
        for rbf in rbfs:
            phi.append(rbf.pdf(point))

        phi = np.array(phi)
        out = np.matmul(w, phi)
        phi = np.reshape(phi, (1, 121))
        e = output_points[i] - out
        e = np.reshape(e, (2,1))
        e = e * eta
        variation = np.matmul(e, phi)
        w += variation

w = np.transpose(w)
phi = []
for pattern in test_input_points:
    new_row = []
    for rbf in rbfs:
        new_row.append(rbf.pdf(pattern))
    phi.append(new_row)
output = np.matmul(phi, w)

e = 0
for i in range(len(output)):
    for j in range(2):
        e += abs(output[i][j] - test_output_points[i][j])
print(output)
print(e/len(test_input_points))
