import numpy as np

from interpolate import equalize_dimensions


def affine_transform_inverse(vector, weights, intercepts):
    inverse_weights = np.linalg.inv(weights)
    transform = (vector - intercepts).dot(inverse_weights)
    return transform


def affine_transform(points, weights, intercepts):
    transform = points.dot(weights) + intercepts
    _, transform = equalize_dimensions(points, transform)
    return transform


def intercepts_translation(point, intercepts):
    point, intercepts = equalize_dimensions(point, intercepts)
    transform = point + intercepts
    return transform


def nonlinear_transform(point, func):
    transform = func(point)
    _, transform = equalize_dimensions(point, transform)
    return transform


def nonlinear_transform_inverse(vector, inverse_nonlinear_func):
    return inverse_nonlinear_func(vector)


def relu(x):
    return x * (x > 0)
