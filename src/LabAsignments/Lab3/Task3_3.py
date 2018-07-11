# Basic, synchronously updated Hopefield network
import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1


def sign_for_parse(x):
    if x == 0:
        return 1
    else:
        return -1


def save(x, iteration=0):
    x = np.reshape(x, (32, 32))
    plt.imshow(x, cmap='binary')
    plt.savefig('Pictures/Iteration'+ str(iteration))


def draw(x):
    x = np.reshape(x, (32, 32))
    plt.imshow(x, cmap='binary')
    plt.show()


def energy(w, x):
    return -sum(w[i][j]*x[i]*x[j] for i in range(len(x)) for j in range(len(x)))

# Patterns to remember

with open('pict.dat') as f:
    data = f.read()

data_splitted = data.split(',')
for i in range(len(data_splitted)):
    data_splitted[i] = int(data_splitted[i])

x = []
for i in range(11):
    x.append(data_splitted[i*1024:(i+1)*1024])

N = len(x[0])

p1 = np.array(x[0])
p2 = np.array(x[1])
p3 = np.array(x[2])
p10 = np.array(x[9])


patterns = [p1, p2, p3]





# Weight calculations:
W = 1/N * np.matmul(np.transpose(patterns), patterns)

# Reacall function

x = p10
save(x, 0)
iteration = 0
energies = [energy(W, x)]
while True:
    iteration += 1
    if iteration % 100 == 0:
        energies.append(energy(W, x))

    unit = np.random.randint(0, 1024)
    output = sign(sum(W[unit][j] * x[j] for j in range(N)))
    x[unit] = output

    if all(x == p1):
        energies.append(energy(W, x))
        break

print(iteration)
print(energies)
print('Energy of p1: ', energy(W, p1))

