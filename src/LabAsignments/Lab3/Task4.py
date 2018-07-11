# Basic, synchronously updated Hopefield network
import random

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

def draw(x):
    x = np.reshape(x, (32, 32))
    plt.imshow(x, cmap='binary')
    plt.show()


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
p4 = np.array(x[3])
p5 = np.array(x[4])
p6 = np.array(x[5])
p7 = np.array(x[6])
p8 = np.array(x[7])
p9 = np.array(x[8])
p10 = np.array(x[9])
p11 = np.array(x[10])


patterns = [p1, p2, p3]





# Weight calculations:
W = np.matmul(np.transpose(patterns), patterns)

# Reacall function


for percent in range(101):
    x = list(p2)
    n_pixels_to_flip = percent/100 * 1024
    index_list = list(range(1024))
    random.shuffle(index_list)
    pixels_to_flip = [index_list[i] for i in range(int(n_pixels_to_flip))]
    iteration = 0

    for pixel_to_flip in pixels_to_flip:
        x[pixel_to_flip] = x[pixel_to_flip] * -1

    while True:
        iteration += 1
        print(iteration)
        unit = np.random.randint(0, 1024)
        output = sign(sum(W[unit][j] * x[j] for j in range(N)))
        x[unit] = output

        if iteration > 10000:
            break
    draw(output)
    if all(output == p2):
        print(percent,': True')
    else:
        print(percent,': False')
