import numpy as np
from matplotlib import pyplot as plt, animation as animation
from mpl_toolkits.mplot3d import axes3d as p3

from dataset import get_xor_data, get_circle_data
from model import train
from transform_forward import transform_forward_on_dataset
from transform_inverse import transform_inverse_on_dataset


def identity(x):
    return x


def get_plane(x, y, weights, intercepts, func=identity):
    return (
        (-intercepts[0] - (weights[0] * func(x)) - (weights[1] * func(y)))
        * 1.0
        / weights[2]
    )


def plot_plane():
    weights = np.array([[-1.51774293], [1.76071715], [0.99202684]])
    intercepts = [-0.18383826]

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    xx, yy = np.meshgrid(np.linspace(-0.99, 0.99, 100), np.linspace(-0.99, 0.99, 100))
    z = (
        lambda x, y: (
            -intercepts[0] - (weights[0] * np.arctanh(x)) - (weights[1] * np.arctanh(y))
        )
        * 1.0
        / weights[2]
    )

    ax.plot_surface(xx, yy, z(xx, yy))
    ax.plot_surface(xx, yy, yy)
    plt.show()


def apply_arctanh(x, y):
    return np.arctanh(x), np.arctanh(y)


def apply_intercepts(x, y, intercepts):
    return x + intercepts[0], y + intercepts[1]


def apply_weights(x, y, weights):
    mult = weights.dot(np.vstack((x, y)))
    return mult[0], mult[1]


def plot_inverse_transforms_on_line():
    y = lambda i: -5.75 * i - 0.022

    xes = []
    for i in np.linspace(-1, 1, 100):
        if -1 < y(i) < 1:
            xes.append(i)
    min_x, max_x = min(xes), max(xes)

    x = np.linspace(min_x, max_x, 1000)

    x1, y1 = apply_arctanh(x, y(x))
    x2, y2 = apply_intercepts(x1, y1, np.array([0.059, 0.049]))
    x3, y3 = apply_weights(x2, y2, np.array([[0.393, -0.110], [-0.833, 0.974]]))
    x4, y4 = apply_arctanh(x3, y3)
    x5, y5 = apply_intercepts(x4, y4, np.array([-0.014, -0.025]))
    x6, y6 = apply_weights(x5, y5, np.array([[-2.156, -4.035], [2.672, 4.286]]))
    x7, y7 = apply_arctanh(x6, y6)
    x8, y8 = apply_intercepts(x7, y7, np.array([-0.193, -1.778]))
    x9, y9 = apply_weights(x8, y8, np.array([[-0.250, 1.500], [-0.248, -0.303]]))

    plt.plot(x, y(x))
    plt.plot(x1, y1, "-.g", label="arctanh 1")
    plt.plot(x2, y2, "-.b", label="intercepts 1")
    plt.plot(x3, y3, "-.r", label="transform 1")
    # plt.plot(x4, y4, "-.g", label="arctanh 2")
    # plt.plot(x5, y5, "-.b", label="intercepts 2")
    # plt.plot(x6, y6, "-.r", label="transform 2")
    # plt.plot(x7, y7, "-.g", label="arctanh 3")
    # plt.plot(x8, y8, "-.b", label="intercepts 3")
    # plt.plot(x9, y9, "-.r", label="transform 3")

    # plt.plot(np.arctanh(x), np.arctanh(-5.75 * x - 0.022), "-.r")

    # plt.plot(
    #     np.arctanh(x) + 0.059,
    #     np.arctanh(-5.75 * x - 0.022) + 0.049,
    #     "-.b",
    #     label="intercepts 1",
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


def animate_transform_forward(data, labels):
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    ax.set_xlim3d([-6, 6])
    ax.set_xlabel("X")

    ax.set_ylim3d([-6, 6])
    ax.set_ylabel("Y")

    ax.set_zlim3d([-6, 6])
    ax.set_zlabel("Z")

    ax.set_title("3D Transformation")

    weights, intercepts = train(data, labels)

    _, point_interpolations = transform_forward_on_dataset(
        data, weights, intercepts, np.tanh
    )

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


def animate_transform_then_inverse(data, labels):
    fig = plt.figure()
    ax = p3.Axes3D(fig)

    ax.set_xlim3d([-6, 6])
    ax.set_xlabel("X")

    ax.set_ylim3d([-6, 6])
    ax.set_ylabel("Y")

    ax.set_zlim3d([-6, 6])
    ax.set_zlabel("Z")

    ax.set_title("3D Transformation")

    weights, intercepts = train(data, labels)

    transformed_points, forward_point_interpolations = transform_forward_on_dataset(
        data, weights, intercepts, np.tanh
    )

    _, inverse_point_interpolations = transform_inverse_on_dataset(
        transformed_points, weights, intercepts, np.arctanh, num_frames=10
    )

    point_interpolations = np.vstack(
        [forward_point_interpolations, inverse_point_interpolations]
    )

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


def plot_decision_surface(data, labels):
    weights, intercepts = train(data, labels)

    # equation of a plane is ax + by + cz + d = 0
    # the x and y values are known because they are given by the meshgrid
    # we can solve for z: z = (-d - ax - by) / c
    xx, yy = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    z = (
        lambda x, y: (-intercepts[-1][0] - (weights[-1][0] * x) - (weights[-1][1] * y))
        * 1.0
        / weights[-1][2]
    )

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    transformed_points, _ = transform_forward_on_dataset(
        data, weights, intercepts, np.tanh
    )
    scatter_points = [
        ax.scatter(output[0][0:1], output[0][1:2], output[0][2:], c=get_color(label))
        for output, label in zip(transformed_points, labels)
    ]

    ax.plot_surface(xx, yy, z(xx, yy), alpha=0.8)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    plt.show()


def plot_decision_surface_points(data, labels):
    weights, intercepts = train(data, labels)

    z = (
        lambda x, y: (-intercepts[-1][0] - (weights[-1][0] * x) - (weights[-1][1] * y))
        * 1.0
        / weights[-1][2]
    )

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("3D Transformation")

    transformed_points, _ = transform_forward_on_dataset(
        data, weights, intercepts, np.tanh
    )
    scatter_points = [
        ax.scatter(output[0][0:1], output[0][1:2], output[0][2:], c=get_color(label))
        for output, label in zip(transformed_points, labels)
    ]

    x = np.linspace(-0.99, 0.99, 20)
    y = np.linspace(-0.99, 0.99, 20)

    xx, yy = np.meshgrid(x, y)
    z_vals = z(xx, yy)

    decision_surface_points = []
    for x_ind, _ in enumerate(xx):
        for y_ind, _ in enumerate(yy):
            z_val = z_vals[x_ind][y_ind]
            if -1 < z_val < 1:
                scatter_points.append(
                    ax.scatter(xx[x_ind, y_ind], yy[x_ind, y_ind], z_val, c="black")
                )
                decision_surface_points.append(
                    np.array([[xx[x_ind, y_ind], yy[x_ind, y_ind], z_val]])
                )

    plt.show()


def animate_decision_surface_inverse_transforms(data, labels):
    weights, intercepts = train(data, labels)

    z = (
        lambda x, y: (-intercepts[-1][0] - (weights[-1][0] * x) - (weights[-1][1] * y))
        * 1.0
        / weights[-1][2]
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

    transformed_points, _ = transform_forward_on_dataset(
        data, weights, intercepts, np.tanh
    )
    scatter_points = [
        ax.scatter(output[0][0:1], output[0][1:2], output[0][2:], c=get_color(label))
        for output, label in zip(transformed_points, labels)
    ]

    x = np.linspace(-0.99, 0.99, 20)
    y = np.linspace(-0.99, 0.99, 20)

    xx, yy = np.meshgrid(x, y)
    z_vals = z(xx, yy)

    decision_surface_points = []
    for x_ind, _ in enumerate(xx):
        for y_ind, _ in enumerate(yy):
            z_val = z_vals[x_ind][y_ind]
            if -1 < z_val < 1:
                scatter_points.append(
                    ax.scatter(xx[x_ind, y_ind], yy[x_ind, y_ind], z_val, c="black")
                )
                decision_surface_points.append(
                    np.array([[xx[x_ind, y_ind], yy[x_ind, y_ind], z_val]])
                )

    all_points = transformed_points + decision_surface_points
    # all_points = transformed_points
    # all_points = decision_surface_points

    _, point_interpolations = transform_inverse_on_dataset(
        all_points, weights, intercepts, np.arctanh, num_frames=3
    )

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


if __name__ == "__main__":
    # TODO
    # command line options to choose nonlinear func, model hyperparameters
    # allow varying affine transform matrix sizes, i.e. 2x2 then 3x3
    # download animation mp4 at high framerate
    # plot labelling for transformation animation
    # show decision boundary plane transforming back to original input space

    # data, labels = get_xor_data(300)
    data, labels = get_circle_data(300)

    # plot_inverse_transforms_on_line()
    # animate_transform_forward(data, labels)
    plot_decision_surface(data, labels)

    # animate_transform_then_inverse(data, labels)
    # plot_decision_surface_points(data, labels)
    # animate_decision_surface_inverse_transforms(data, labels)
