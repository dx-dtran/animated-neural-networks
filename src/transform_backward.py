import numpy as np

from dataset import read_2d_data_in_3d
from model import train
from transform_forward import apply_transformations_on_dataset
from transformations import affine_transform_inverse, nonlinear_transform_inverse
from interpolate import interpolate_points, stack_interpolations


def transform_inverse_on_dataset(
    data, weights, intercepts, inverse_nonlinear_func
):
    transformations = []
    interpolations = []
    for point in data:
        inv, intermediate_points = transform_inverse(
            point, weights, intercepts, inverse_nonlinear_func
        )
        interps = interpolate_points(intermediate_points)
        interpolations.append(interps)
        transformations.append(inv)
    return transformations, stack_interpolations(interpolations)


def transform_inverse(point, weights, intercepts, inverse_nonlinear_func):
    intermediate_points = [point]
    transformed_point = point
    for i, _ in reversed(list(enumerate(weights[:-1]))):
        w = weights[i]
        b = intercepts[i]

        transformed_point = nonlinear_transform_inverse(
            transformed_point, inverse_nonlinear_func
        )
        intermediate_points.append(transformed_point)

        transformed_point = affine_transform_inverse(transformed_point, w, b)
        intermediate_points.append(transformed_point)
    return transformed_point, intermediate_points


def main():
    data, labels = read_2d_data_in_3d("../data/xor_data.csv")
    weights, intercepts = train(data, labels)

    transformed_output, _ = apply_transformations_on_dataset(
        data, weights, intercepts, np.tanh
    )
    print(transformed_output)
    inv_transformations, _ = transform_inverse_on_dataset(
        transformed_output, weights, intercepts, np.arctanh
    )
    print(inv_transformations)


if __name__ == "__main__":
    main()
