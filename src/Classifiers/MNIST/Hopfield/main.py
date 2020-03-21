import numpy as np
import matplotlib.pyplot as plt
import math


def visualise_pattern(pattern):
    n = int(math.sqrt(pattern.shape[0]))
    im = pattern.reshape((n, n))
    plt.imshow(im, cmap='binary')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    p1 = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1])
    p2 = np.array([-1, 1, -1, -1, 1, -1, -1, 1, -1])

    patterns = [p1, p2]

    net = np.zeros((9, 9))

    for i in range(9):
        for j in range(i+1, 9):
            w = 0
            for pattern in patterns:
                w += pattern[i] * pattern[j]
            net[i, j] = w / len(patterns)
            net[j, i] = w / len(patterns)

    p_noisy = np.array([-1, 1, -1, 1, -1, -1, -1, 1, -1])
    for _ in range(100):
        idx = np.random.randint(0, 9)
        result_sum = sum([net[idx, j] * p_noisy[j] for j in range(9)])
        p_noisy[idx] = 1 if result_sum > 0 else -1

    visualise_pattern(p_noisy)
