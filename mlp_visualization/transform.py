import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from scipy.stats import logistic
from sklearn.neural_network import MLPClassifier
from dataset import read_2d_data


def train(data, labels):
    classifier = MLPClassifier(
        activation="tanh",
        hidden_layer_sizes=(3, 3, 3),
        verbose=True,
        max_iter=2000,
    )
    classifier.fit(data, labels)
    return classifier.coefs_, classifier.intercepts_


def forward(data, weights, biases):
    interpolations = []
    transform = data
    for i, _ in enumerate(weights[:-1]):
        w = weights[i]
        b = biases[i]
        transform = layer_forward(transform, interpolations, w, b, np.tanh)

    output = layer_forward(
        transform, interpolations, weights[-1], biases[-1], np.tanh
    )
    return interpolations, output


def layer_forward(data, interpolations, weights, biases, activation_func):
    data = affine_transform(data, weights, interpolations)
    data = bias_transform(data, biases, interpolations)
    data = nonlinear_transform(data, activation_func, interpolations)
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


def affine_transform(data, weights, interpolations, num_frames=10):
    transform = data.dot(weights)
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


def apply_transformations(data, weights, biases):
    interpolations = []
    for point in data:
        vector = np.array(point).reshape(1, -1)
        point_interpolations, _ = forward(vector, weights, biases)
        flattened = flatten_interpolations(point_interpolations)
        interpolations.append(flattened)
    return np.stack(interpolations, axis=1)


def main():
    data, labels = read_2d_data("xor_data.csv")
    # weights, biases = train(data, labels)

    weights = [
        np.array(
            [
                [-0.55999292, 0.28192111, -1.34789915],
                [0.13920688, 1.40550969, -0.24589298],
            ]
        ),
        np.array(
            [
                [1.16475447, -0.56179116, 0.64139247],
                [1.03785943, 0.83246431, 1.0993098],
                [0.94497092, -0.47678784, 0.08990984],
            ]
        ),
        np.array(
            [
                [1.18488039, 0.28029689, -1.12000824],
                [1.45068749, 0.41759623, -0.14343635],
                [-1.41138955, -0.39072456, -0.3001835],
            ]
        ),
        np.array(
            [
                [1.25770705, 1.01394874, 0.84465758],
                [0.23366424, 1.11778236, -0.24158952],
                [-0.64499363, -0.85234794, -0.39926406],
            ]
        ),
        np.array(
            [
                [-0.474417, 1.38640964, -0.35385609],
                [-1.20349871, 1.10561642, -1.33743325],
                [-0.38530791, 0.10093747, -1.27291985],
            ]
        ),
        np.array([[-0.98995076], [1.27680552], [-1.78223569]]),
    ]

    biases = [
        np.array([-0.15748354, -0.01573772, 0.49332679]),
        np.array([1.23372424, -1.01715427, 0.14869703]),
        np.array([0.22945446, 0.07431896, -0.46186782]),
        np.array([-0.48909483, -0.58775503, -0.22048673]),
        np.array([-0.09299892, 0.12911864, 0.01070645]),
        np.array([-0.03951273]),
    ]

    point_transformations = apply_transformations(data, weights, biases)
    print(point_transformations[0])
    print(len(point_transformations[0]))

    # fig = plt.figure()
    # ax = p3.Axes3D(fig)
    #
    # # Initialize scatters
    # scatters = [ ax.scatter(data[0][i,0:1], data[0][i,1:2], data[0][i,2:], c="blue") for i in range(data[0].shape[0]) ]
    # print([(data[0][i,0:1], data[0][i,1:2], data[0][i,2:]) for i in range(data[0].shape[0])])
    # print(len(data))
    # print(data[0])
    #
    # # Number of iterations
    # iterations = len(data)
    #
    # # Setting the axes properties
    # ax.set_xlim3d([-50, 50])
    # ax.set_xlabel('X')
    #
    # ax.set_ylim3d([-50, 50])
    # ax.set_ylabel('Y')
    #
    # ax.set_zlim3d([-50, 50])
    # ax.set_zlabel('Z')
    #
    # ax.set_title('3D Animated Scatter Example')
    #
    # # Provide starting angle for the view.
    # ax.view_init(25, 10)
    #
    # # ani = animation.FuncAnimation(fig, animate_scatters, iterations, fargs=(data, scatters),
    # #                                     interval=50, blit=False, repeat=True)
    #
    # plt.show()


if __name__ == "__main__":
    # p1 = np.array([1, 2, 3]).reshape((1, -1))
    # p2 = np.array([7, 5, -2]).reshape((1, -1))
    # p3 = np.array([2, 3, 1]).reshape((1, -1))
    # p4 = np.array([4, 1, -5]).reshape((1, -1))
    #
    # p5 = np.array([1, 2, 3]).reshape((1, -1))
    # p6 = np.array([7, 5, -2]).reshape((1, -1))
    # p7 = np.array([2, 3, 1]).reshape((1, -1))
    # p8 = np.array([4, 1, -5]).reshape((1, -1))
    #
    # x = interpolate(p1, p2, 10)
    # y = interpolate(p3, p4, 10)
    #
    # r = interpolate(p5, p6, 10)
    # t = interpolate(p7, p8, 10)
    #
    # l = []
    # l.append(x)
    # l.append(y)
    # f = flatten_interpolations(l)
    #
    # z = []
    # z.append(r)
    # z.append(t)
    # f2 = flatten_interpolations(z)
    #
    # # print(f)
    # # print(f2)
    #
    # my_list = [f, f2]
    # bla = np.stack(my_list, axis=1)
    # print(bla)
    # print(bla[0])
    main()