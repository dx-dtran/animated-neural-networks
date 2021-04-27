import numpy as np

from interpolate import interpolate_points, stack_interpolations
from transformations import affine_transform, nonlinear_transform


def transform_forward_on_dataset(data, weights, intercepts, nonlinear_func):
    transformations = []
    interpolations = []
    for val in data:
        point = np.array(val).reshape(1, -1)
        output, intermediate_points = transform_forward(
            point, weights, intercepts, nonlinear_func
        )
        interps = interpolate_points(intermediate_points)
        interpolations.append(interps)
        transformations.append(output)
    return transformations, stack_interpolations(interpolations)


def transform_forward(point, weights, intercepts, nonlinear_func):
    intermediate_points = [point]
    transformed_point = point
    for i, _ in enumerate(weights[:-1]):
        w = weights[i]
        b = intercepts[i]

        transformed_point = affine_transform(transformed_point, w, b)
        intermediate_points.append(transformed_point)

        transformed_point = nonlinear_transform(transformed_point, nonlinear_func)
        intermediate_points.append(transformed_point)
    return transformed_point, intermediate_points
