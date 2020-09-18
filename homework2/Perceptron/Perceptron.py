import matplotlib.pyplot as plt
import numpy as np


def perceptron(x_vectors, y_vectors):
    theta = np.array([[0, 0]])
    theta_cero = np.array([0])
    k = 0
    salida_rapida = False

    for t in range(10):
        if salida_rapida:
            break

        salida_rapida = True
        for i in range(len(x_vectors)):
            if y_vectors[i] * (np.dot(theta[k], x_vectors[i]) + theta_cero[k]) <= 0:
                theta = np.vstack((theta, theta[k] + y_vectors[i] * x_vectors[i]))
                theta_cero = np.append(theta_cero, theta_cero[k] + y_vectors[i])
                k += 1
                salida_rapida = False

    return theta, theta_cero, k

#
# xs = np.array([[1, 4], [5, -4], [6, 2],
#                [1, -3], [-2, 2], [-3, -5]])
# ys = np.array([1, 1, 1, -1, -1, -1])
# th1, th0, tries = perceptron(xs, ys)
# x1 = xs[:, 0]
# x2 = xs[:, 1]
# t = np.linspace(x1.min(), x1.max())
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(x1, x2, c=ys)
#
# ax.plot(t, -(th1[tries][0] * t + th0[tries]) / th1[tries][1], color='darkblue')
# plt.show()
