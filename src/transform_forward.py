import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from dataset import get_xor_data, get_circle_data
from interpolate import interpolate_points, stack_interpolations
from model import train
from transformations import affine_transform, nonlinear_transform


def apply_transformations_on_dataset(data, weights, intercepts, nonlinear_func):
    transformations = []
    interpolations = []
    for val in data:
        point = np.array(val).reshape(1, -1)
        output, intermediate_points = forward(
            point, weights, intercepts, nonlinear_func
        )
        interps = interpolate_points(intermediate_points)
        interpolations.append(interps)
        transformations.append(output)
    return transformations, stack_interpolations(interpolations)


def forward(point, weights, intercepts, nonlinear_func):
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

    _, point_interpolations = apply_transformations_on_dataset(
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
        output, _ = forward(vector, weights, intercepts, np.tanh)
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

    animate_transformations()
    # plot_decision_boundary()
