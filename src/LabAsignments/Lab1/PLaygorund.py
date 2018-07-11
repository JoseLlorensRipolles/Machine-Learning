import numpy as np
import random


if __name__ == '__main__':
    x = np.array([1, 2, 3, 4])
    y = np.array([11, 22, 33, 44])

    aux = np.vstack((x,y))
    aux = np.transpose(aux)
    np.random.shuffle(aux)
    print(aux)
    x = aux[:,0]
    y = aux[:,1]
    print(x)
    print(y)
