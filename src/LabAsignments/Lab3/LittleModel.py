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
x1=[-1, -1, 1, -1, 1, -1, -1, 1]


patterns = [x1]
print(x1)





# Weight calculations:
W = np.matmul(np.transpose(patterns), patterns)
print(W)
# Reacall function

x = p11
draw(x)
while True:
    output = []
    for i in range(N):
        output.append(sign(sum(W[i][j] * x[j] for j in range(N))))

    if np.array_equal(x, output):
        break
    else:
        x = output


draw(output)