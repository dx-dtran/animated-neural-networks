from sklearn.neural_network import MLPClassifier
from dataset import read_2d_data
import numpy as np


def train_xor_data():
    X, labels = read_2d_data("../data/xor_data.csv")
    classifier = MLPClassifier(
        activation="tanh",
        hidden_layer_sizes=(3, 3, 3),
        verbose=True,
        max_iter=20000,
    )
    classifier.fit(X, labels)
    print([coef.shape for coef in classifier.coefs_])
    print(classifier.coefs_)
    print(classifier.intercepts_)

    print(
        classifier.predict(
            np.array([5.061012059504988, 0.8093913561939445]).reshape(1, -1)
        )
    )
    print(
        classifier.predict_proba(
            np.array([5.061012059504988, 0.8093913561939445]).reshape(1, -1)
        )
    )
    print(
        classifier.predict_log_proba(
            np.array([5.061012059504988, 0.8093913561939445]).reshape(1, -1)
        )
    )


def train(data, labels):
    classifier = MLPClassifier(
        activation="tanh",
        hidden_layer_sizes=(3, 3, 3),
        verbose=True,
        max_iter=2000,
    )
    classifier.fit(data, labels)
    return classifier.coefs_, classifier.intercepts_


if __name__ == "__main__":
    train_xor_data()
