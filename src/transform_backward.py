import numpy as np

from dataset import get_circle_data, get_xor_data, read_2d_data_in_3d
from model import train
from transform import apply_transformations_on_dataset


def apply_inverse_transformations_on_dataset(
    data, weights, intercepts, inverse_nonlinear_func
):
    transformations = []
    # vectorized_data = [np.array(point).reshape(1, -1) for point in data]
    for vector in data:
        inv = inverse(vector, weights, intercepts, inverse_nonlinear_func)
        transformations.append(inv)
    return transformations


def inverse(vector, weights, intercepts, inverse_nonlinear_func):
    transform = vector
    for i, _ in reversed(list(enumerate(weights[:-1]))):
        w = weights[i]
        b = intercepts[i]
        transform = nonlinear_transform_inverse(transform, inverse_nonlinear_func)
        transform = affine_transform_inverse(transform, w, b)
    return transform


def affine_transform_inverse(vector, weights, intercepts):
    inverse_weights = np.linalg.inv(weights)
    transform = (vector - intercepts).dot(inverse_weights)
    return transform


def nonlinear_transform_inverse(vector, inverse_nonlinear_func):
    return inverse_nonlinear_func(vector)


def main():
    data, labels = read_2d_data_in_3d("../data/xor_data.csv")
    weights, intercepts = train(data, labels)

    transformed_output = apply_transformations_on_dataset(
        data, weights, intercepts, np.tanh
    )
    print(transformed_output)
    inv_transformations = apply_inverse_transformations_on_dataset(
        transformed_output, weights, intercepts, np.arctanh
    )
    print(inv_transformations)


if __name__ == "__main__":
    main()
