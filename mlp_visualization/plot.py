import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


def identity(x):
    return x


def get_plane(x, y, weights, biases, func=identity):
    return (
        (-biases[0] - (weights[0] * func(x)) - (weights[1] * func(y)))
        * 1.0
        / weights[2]
    )


def main():
    weights = np.array([[-1.51774293], [1.76071715], [0.99202684]])
    biases = [-0.18383826]

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    xx, yy = np.meshgrid(np.linspace(-0.99, 0.99, 100), np.linspace(-0.99, 0.99, 100))
    # xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    z = (
        lambda x, y: (
            -biases[0] - (weights[0] * np.arctanh(x)) - (weights[1] * np.arctanh(y))
        )
        * 1.0
        / weights[2]
    )

    ax.plot_surface(xx, yy, z(xx, yy))
    ax.plot_surface(xx, yy, yy)
    plt.show()


def apply_arctanh(x, y):
    return np.arctanh(x), np.arctanh(y)


def apply_biases(x, y, biases):
    return x + biases[0], y + biases[1]


def apply_weights(x, y, weights):
    mult = weights.dot(np.vstack((x, y)))
    return mult[0], mult[1]


def plot_lines():
    x = np.linspace(-10, 10, 1000)
    y = lambda i: -5.75 * i - 0.022

    x1, y1 = apply_arctanh(x, y(x))
    x2, y2 = apply_biases(x1, y1, np.array([0.059, 0.049]))
    x3, y3 = apply_weights(x2, y2, np.array([[0.393, -0.110], [-0.833, 0.974]]))
    x4, y4 = apply_arctanh(x3, y3)
    x5, y5 = apply_biases(x4, y4, np.array([-0.014, -0.025]))
    x6, y6 = apply_weights(x5, y5, np.array([[-2.156, -4.035], [2.672, 4.286]]))
    x7, y7 = apply_arctanh(x6, y6)
    x8, y8 = apply_biases(x7, y7, np.array([-0.193, -1.778]))
    x9, y9 = apply_weights(x8, y8, np.array([[-0.250, 1.500], [-0.248, -0.303]]))

    # plt.plot(x1, y1, "-.g", label="arctanh 1")
    # plt.plot(x2, y2, "-.b", label="bias 1")
    # plt.plot(x3, y3, "-.r", label="transform 1")
    plt.plot(x4, y4, "-.g", label="arctanh 2")
    plt.plot(x5, y5, "-.b", label="bias 2")
    plt.plot(x6, y6, "-.r", label="transform 2")
    # plt.plot(x7, y7, "-.g", label="arctanh 3")
    # plt.plot(x8, y8, "-.b", label="bias 3")
    # plt.plot(x9, y9, "-.r", label="transform 3")

    # plt.plot(np.arctanh(x), np.arctanh(-5.75 * x - 0.022), "-.r")

    # plt.plot(
    #     np.arctanh(x) + 0.059,
    #     np.arctanh(-5.75 * x - 0.022) + 0.049,
    #     "-.b",
    #     label="bias 1",
    # )
    # plt.plot(
    #     0.393 * (np.arctanh(x) + 0.059)
    #     - 0.110 * (np.arctanh(-5.75 * x - 0.022) + 0.059),
    #     -0.833 * (np.arctanh(x) + 0.059)
    #     + 0.974 * (np.arctanh(-5.75 * x - 0.022) + 0.049),
    #     "-.r",
    #     label="transform 1",
    # )

    plt.xlabel("x", color="#1C2833")
    plt.ylabel("y", color="#1C2833")
    plt.legend(loc="upper left")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_lines()
