import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from dataset import get_xor_data, get_circle_data
from model import train


def relu(x):
    return x * (x > 0)


def forward(data, weights, biases, activation_func):
    interpolations = []
    transform = data
    for i, _ in enumerate(weights[:-1]):
        w = weights[i]
        b = biases[i]
        transform = layer_forward(transform, interpolations, w, b, activation_func)

    # output = layer_forward(
    #     transform, interpolations, weights[-1], biases[-1], logistic.cdf
    # )
    # output = output * np.array([1, 0, 0])

    # output = layer_forward2(transform, interpolations, weights[-1], biases[-1])
    output = transform
    return interpolations, output


def layer_forward(data, interpolations, weights, biases, activation_func):
    data = affine_transform(data, weights, biases, interpolations, 3)
    # data = bias_transform(data, biases, interpolations)
    data = nonlinear_transform(data, activation_func, interpolations, 3)
    return data


def layer_forward2(data, interpolations, weights, biases):
    data = affine_transform(data, weights, biases, interpolations, 3)
    # data = bias_transform(data, biases, interpolations)
    # data = nonlinear_transform(data, activation_func, interpolations, 3)
    return data


def interpolate(start_point, target_point, num_frames=10):
    start_point, target_point = equalize_dimensions(start_point, target_point)
    alpha = np.linspace(0, 1, num_frames).reshape((-1, 1))
    return (1 - alpha).dot(start_point) + alpha.dot(target_point)


def convert_1d_to_2d(array):
    if len(array.shape) == 1:
        return np.reshape(array, (-1, array.size))
    return array


def equalize_dimensions(point1, point2):
    point1 = convert_1d_to_2d(point1)
    point2 = convert_1d_to_2d(point2)
    if point1.shape[1] < point2.shape[1]:
        zeros = np.zeros((point1.shape[0], point2.shape[1] - point1.shape[1]))
        point1 = np.column_stack((point1, zeros))
    elif point1.shape[1] > point2.shape[1]:
        zeros = np.zeros((point2.shape[0], point1.shape[1] - point2.shape[1]))
        point2 = np.column_stack((point2, zeros))
    return point1, point2


def affine_transform(data, weights, biases, interpolations, num_frames=10):
    transform = data.dot(weights) + biases
    _, transform = equalize_dimensions(data, transform)
    interp = interpolate(data, transform, num_frames)
    interpolations.append(interp)
    return transform


def bias_transform(data, bias, interpolations, num_frames=10):
    data, bias = equalize_dimensions(data, bias)
    transform = data + bias
    interp = interpolate(data, transform, num_frames)
    interpolations.append(interp)
    return transform


def nonlinear_transform(data, func, interpolations, num_frames=10):
    transform = func(data)
    _, transform = equalize_dimensions(data, transform)
    interp = interpolate(data, transform, num_frames)
    interpolations.append(interp)
    return transform


def flatten_interpolations(interpolations):
    points = []
    for interpolation in interpolations:
        for point in interpolation:
            points.append(point)
    return np.array(points)


def apply_transformations(data, weights, biases, activation_func):
    interpolations = []
    for point in data:
        vector = np.array(point).reshape(1, -1)
        point_interpolations, _ = forward(vector, weights, biases, activation_func)
        flattened = flatten_interpolations(point_interpolations)
        interpolations.append(flattened)
    return np.stack(interpolations, axis=1)


def get_color(label):
    if label < 0:
        return "blue"
    return "orange"


def animate_scatter_points(iteration, points, scatters):
    for i in range(points[0].shape[0]):
        scatters[i]._offsets3d = (
            points[iteration][i, 0:1],
            points[iteration][i, 1:2],
            points[iteration][i, 2:],
        )
    return scatters


def main():
    data, labels = get_xor_data(200)
    weights, biases = train(data, labels)

    print(weights[-1], biases[-1])

    point_transformations = apply_transformations(data, weights, biases, np.tanh)

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
            point_transformations[0][i, 0:1],
            point_transformations[0][i, 1:2],
            point_transformations[0][i, 2:],
            c=get_color(label),
        )
        for i, label in zip(range(point_transformations[0].shape[0]), labels)
    ]

    iterations = len(point_transformations)

    ani = animation.FuncAnimation(
        fig,
        animate_scatter_points,
        iterations,
        fargs=(point_transformations, scatter_points),
        interval=1,
        blit=False,
    )

    ani

    plt.show()


def main2():
    data, labels = get_circle_data(200)
    weights, biases = train(data, labels)


    print(weights[-1], biases[-1])
    # equation of a plane is ax + by + cz + d = 0
    # the x and y values are known because they are given by the meshgrid
    # we can solve for z: z = (-d - ax - by) / c
    xx, yy = np.meshgrid(np.linspace(-3, 3, 10), np.linspace(-3, 3, 10))
    z = (
        lambda x, y: (-biases[-1][0] - (weights[-1][0] * x) - (weights[-1][1] * y))
        * 1.0
        / weights[-1][2]
    )

    fig = plt.figure()
    ax = fig.gca(projection="3d")

    vectorized_data = [np.array(point).reshape(1, -1) for point in data]
    scatter_points = []

    for vector, label in zip(vectorized_data, labels):
        _, output = forward(vector, weights, biases, np.tanh)
        scatter_points.append(
            ax.scatter(
                output[0][0:1], output[0][1:2], output[0][2:], c=get_color(label)
            )
        )

    ax.plot_surface(xx, yy, z(xx, yy), alpha=0.8)
    plt.show()


if __name__ == "__main__":
    # todo: get this to work with 2 dim hidden layers
    # command line options for activation func
    # download mp4 with high framewrate
    # labels for what transformation is currently being animated
    # show decision boundary transforming back to original input space
    # main()
    main2()
