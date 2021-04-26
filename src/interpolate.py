import numpy as np


def interpolate(start_point, target_point, num_frames=10):
    start_point, target_point = equalize_dimensions(start_point, target_point)
    alpha = np.linspace(0, 1, num_frames).reshape((-1, 1))
    return (1 - alpha).dot(start_point) + alpha.dot(target_point)


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


def convert_1d_to_2d(array):
    if len(array.shape) == 1:
        return np.reshape(array, (-1, array.size))
    return array


def interpolate_points(points):
    interpolations = []
    for i in range(1, len(points)):
        start = points[i - 1]
        target = points[i]
        interp = interpolate(start, target)
        interpolations.append(interp)
    return flatten_interpolations(interpolations)


def flatten_interpolations(interpolations):
    points = []
    for interpolation in interpolations:
        for point in interpolation:
            points.append(point)
    return np.array(points)


def stack_interpolations(interpolations):
    return np.stack(interpolations, axis=1)
