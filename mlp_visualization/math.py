import numpy as np
from scipy.stats import logistic


def multiply(weights, biases):
    print(
        forward(
            np.array([5.061012059504988, 0.8093913561939445]).reshape(1, -1),
            weights,
            biases,
        )
    )
    print(
        forward(
            np.array([-4.687975658113184, 3.326457840916059]).reshape(1, -1),
            weights,
            biases,
        )
    )
    print(
        forward(
            np.array([3.074848269639795, 1.7081594099083992]).reshape(1, -1),
            weights,
            biases,
        )
    )
    print(
        forward(
            np.array([-2.7194606428352595, -2.8958698109676946]).reshape(1, -1),
            weights,
            biases,
        )
    )
    print(
        forward(
            np.array([3.360942495673471, -5.22121486229057]).reshape(1, -1),
            weights,
            biases,
        )
    )


def forward(x, weights, biases):
    hidden = x
    # for w, b in zip(weights, biases):
    for i in range(len(weights) - 1):
        w = weights[i]
        b = biases[i]
        # print(hidden.dot(w))
        hidden = hidden.dot(w) + b
        hidden = np.tanh(hidden)
    # return logistic.cdf(hidden.dot(weights[-1]) + biases[-1])

    return np.tanh(hidden.dot(weights[-1]) + biases[-1])

if __name__ == "__main__":
    w = [
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

    b = [
        np.array([-0.15748354, -0.01573772, 0.49332679]),
        np.array([1.23372424, -1.01715427, 0.14869703]),
        np.array([0.22945446, 0.07431896, -0.46186782]),
        np.array([-0.48909483, -0.58775503, -0.22048673]),
        np.array([-0.09299892, 0.12911864, 0.01070645]),
        np.array([-0.03951273]),
    ]
    multiply(w, b)
