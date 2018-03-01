import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.01, 20.0, 0.01)
e1 = [0.172, 0.00159, 0.00918, 0.01273, 0.1286, 0.2688]
e2 = [0.285, 0.18944, 0.20330, 0.20804, 0.2932, 0.6148]
x = [0.2, 1, 2, 3, 4, 5]

plt.subplot(211)
plt.plot(x, e1)
# plt.semilogy()
plt.title('Error for sin function')
plt.xlabel('Variance of the RBF nodes')
plt.ylabel('Mean of the absolute error')

plt.subplot(212)
plt.plot(x, e2)
# plt.semilogy()
plt.title('Error for sin function')
plt.xlabel('Variance of the RBF nodes')
plt.ylabel('Mean of the absolute error')
plt.show()