import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import pyplot as plt, animation as animation
from mpl_toolkits.mplot3d import axes3d as p3

from dataset import get_xor_data, get_circle_data
from model import train
from transform_forward import transform_forward_on_dataset
from math_utils import get_3d_domain


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
    y = lambda i: -5.75 * i - 0.022

    xes = []
    for i in np.linspace(-1, 1, 100):
        if -1 < y(i) < 1:
            xes.append(i)
    min_x, max_x = min(xes), max(xes)

    x = np.linspace(min_x, max_x, 1000)

    x1, y1 = apply_arctanh(x, y(x))
    x2, y2 = apply_biases(x1, y1, np.array([0.059, 0.049]))
    x3, y3 = apply_weights(x2, y2, np.array([[0.393, -0.110], [-0.833, 0.974]]))
    x4, y4 = apply_arctanh(x3, y3)
    x5, y5 = apply_biases(x4, y4, np.array([-0.014, -0.025]))
    x6, y6 = apply_weights(x5, y5, np.array([[-2.156, -4.035], [2.672, 4.286]]))
    x7, y7 = apply_arctanh(x6, y6)
    x8, y8 = apply_biases(x7, y7, np.array([-0.193, -1.778]))
    x9, y9 = apply_weights(x8, y8, np.array([[-0.250, 1.500], [-0.248, -0.303]]))

    plt.plot(x, y(x))
    plt.plot(x1, y1, "-.g", label="arctanh 1")
    plt.plot(x2, y2, "-.b", label="bias 1")
    plt.plot(x3, y3, "-.r", label="transform 1")
    # plt.plot(x4, y4, "-.g", label="arctanh 2")
    # plt.plot(x5, y5, "-.b", label="bias 2")
    # plt.plot(x6, y6, "-.r", label="transform 2")
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


def animate_scatter_points(iteration, points, scatters):
    for i in range(points[0].shape[0]):
        scatters[i]._offsets3d = (
            points[iteration][i, 0:1],
            points[iteration][i, 1:2],
            points[iteration][i, 2:],
        )
    return scatters


def get_color(label):
    if label < 0:
        return "blue"
    return "orange"


def animate_transformations():
    data, labels = get_xor_data(300)
    weights, intercepts = train(data, labels)

    print(weights[-1], intercepts[-1])

    _, point_interpolations = transform_forward_on_dataset(
        data, weights, intercepts, np.tanh
    )

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    ax.set_xlim3d([-6, 6])
    ax.set_xlabel("X")

    ax.set_ylim3d([-6, 6])
    ax.set_ylabel("Y")

    ax.set_zlim3d([-6, 6])
    ax.set_zlabel("Z")

    ax.set_title("3D Transformation")

    scatter_points = [
        ax.scatter(
            point_interpolations[0][i, 0:1],
            point_interpolations[0][i, 1:2],
            point_interpolations[0][i, 2:],
            c=get_color(label),
        )
        for i, label in zip(range(point_interpolations[0].shape[0]), labels)
    ]

    iterations = len(point_interpolations)

    ani = animation.FuncAnimation(
        fig,
        animate_scatter_points,
        iterations,
        fargs=(point_interpolations, scatter_points),
        interval=1,
        blit=False,
    )

    ani

    plt.show()


def plot_decision_boundary():
    data, labels = get_circle_data(300)
    weights, intercepts = train(data, labels)

    # equation of a plane is ax + by + cz + d = 0
    # the x and y values are known because they are given by the meshgrid
    # we can solve for z: z = (-d - ax - by) / c
    xx, yy = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
    z = (
        lambda x, y: (-intercepts[-1][0] - (weights[-1][0] * x) - (weights[-1][1] * y))
        * 1.0
        / weights[-1][2]
    )

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    transformed_points, _ = transform_forward_on_dataset(
        data, weights, intercepts, np.tanh
    )
    scatter_points = [
        ax.scatter(output[0][0:1], output[0][1:2], output[0][2:], c=get_color(label))
        for output, label in zip(transformed_points, labels)
    ]

    ax.plot_surface(xx, yy, z(xx, yy), alpha=0.8)
    plt.show()


def plot_decision_boundary_as_points():
    data, labels = get_circle_data(300)
    weights, intercepts = train(data, labels)

    print(weights[-1], intercepts[-1])
    # equation of a plane is ax + by + cz + d = 0
    # the x and y values are known because they are given by the meshgrid
    # we can solve for z: z = (-d - ax - by) / c

    z = (
        lambda x, y: (-intercepts[-1][0] - (weights[-1][0] * x) - (weights[-1][1] * y))
        * 1.0
        / weights[-1][2]
    )

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    transformed_points, _ = transform_forward_on_dataset(
        data, weights, intercepts, np.tanh
    )
    scatter_points = [
        ax.scatter(output[0][0:1], output[0][1:2], output[0][2:], c=get_color(label))
        for output, label in zip(transformed_points, labels)
    ]
    transformed_points = np.vstack(transformed_points)
    # min_x, min_y, min_z = np.min(transformed_points, axis=0)
    # max_x, max_y, max_z = np.max(transformed_points, axis=0)

    x_range, y_range = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    x_min, x_max, y_min, y_max = get_3d_domain(x_range, y_range, -1, 1, z)
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

    ax.plot_surface(xx, yy, z(xx, yy), alpha=0.8)

    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    z_vals = z(x, y)
    print(z_vals)

    xx, yy = np.meshgrid(x, y)
    z_vals = z(xx, yy)

    for x_val, _ in enumerate(xx):
        for y_val, _ in enumerate(yy):
            z_val = z_vals[x_val][y_val]
            ax.scatter(xx[x_val, y_val], yy[x_val, y_val], z_val, c="black")

    plt.show()


if __name__ == "__main__":
    # todo: get this to work with 2 dim hidden layers
    # command line options for activation func
    # download mp4 with high framerate
    # labels for what transformation is currently being animated
    # show decision boundary transforming back to original input space

    # plot_lines()
    # animate_transformations()
    plot_decision_boundary()
