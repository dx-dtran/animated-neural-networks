from sklearn.neural_network import MLPClassifier


def train(data, labels):
    classifier = MLPClassifier(
        activation="tanh",
        hidden_layer_sizes=(3, 3, 3),
        verbose=True,
        max_iter=2000,
    )
    classifier.fit(data, labels)
    return classifier.coefs_, classifier.intercepts_
