import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from dataset import get_xor_data, get_circle_data
from model import train


def relu(x):
    return x * (x > 0)


def apply_transformations_on_dataset(data, weights, intercepts, nonlinear_func):
    transformations = []
    vectorized_data = [np.array(point).reshape(1, -1) for point in data]
    for vector in vectorized_data:
        _, output = forward(vector, weights, intercepts, nonlinear_func)
        transformations.append(output)
    return transformations


def forward(vector, weights, intercepts, nonlinear_func):
    interpolations = []
    transform = vector
    for i, _ in enumerate(weights[:-1]):
        w = weights[i]
        b = intercepts[i]
        transform = layer_forward(transform, interpolations, w, b, nonlinear_func)

    # output = layer_forward(
    #     transform, interpolations, weights[-1], biases[-1], logistic.cdf
    # )
    # output = output * np.array([1, 0, 0])

    # output = layer_forward2(transform, interpolations, weights[-1], biases[-1])
    output = transform
    return interpolations, output


def layer_forward(vector, interpolations, weights, intercepts, nonlinear_func):
    vector = affine_transform(vector, weights, intercepts, interpolations)
    # data = bias_transform(data, biases, interpolations)
    vector = nonlinear_transform(vector, nonlinear_func, interpolations)
    return vector


def layer_forward2(vector, interpolations, weights, intercepts):
    vector = affine_transform(vector, weights, intercepts, interpolations, 3)
    # data = bias_transform(data, biases, interpolations)
    # data = nonlinear_transform(data, nonlinear_func, interpolations, 3)
    return vector


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


def affine_transform(vector, weights, intercepts, interpolations, num_frames=10):
    transform = vector.dot(weights) + intercepts
    _, transform = equalize_dimensions(vector, transform)
    interp = interpolate(vector, transform, num_frames)
    interpolations.append(interp)
    return transform


def intercepts_translation(vector, intercepts, interpolations, num_frames=10):
    vector, intercepts = equalize_dimensions(vector, intercepts)
    transform = vector + intercepts
    interp = interpolate(vector, transform, num_frames)
    interpolations.append(interp)
    return transform


def nonlinear_transform(vector, func, interpolations, num_frames=10):
    transform = func(vector)
    _, transform = equalize_dimensions(vector, transform)
    interp = interpolate(vector, transform, num_frames)
    interpolations.append(interp)
    return transform


def flatten_interpolations(interpolations):
    points = []
    for interpolation in interpolations:
        for point in interpolation:
            points.append(point)
    return np.array(points)


def apply_transformations(data, weights, intercepts, nonlinear_func):
    interpolations = []
    for point in data:
        vector = np.array(point).reshape(1, -1)
        point_interpolations, _ = forward(vector, weights, intercepts, nonlinear_func)
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


def animate_transformations():
    data, labels = get_xor_data(300)
    weights, intercepts = train(data, labels)

    print(weights[-1], intercepts[-1])

    point_transformations = apply_transformations(data, weights, intercepts, np.tanh)

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


def plot_decision_boundary():
    data, labels = get_circle_data(300)
    weights, intercepts = train(data, labels)

    print(weights[-1], intercepts[-1])
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

    vectorized_data = [np.array(point).reshape(1, -1) for point in data]
    scatter_points = []

    for vector, label in zip(vectorized_data, labels):
        _, output = forward(vector, weights, intercepts, np.tanh)
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
    # refactor the layer_forward methods to only return the transformed vectors
    # calculate interpolations between transformed vectors as a separate method

    animate_transformations()
    # plot_decision_boundary()
