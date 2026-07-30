import random

import matplotlib.pyplot as plt
import numpy as np

from vis import error_visualisation, universal_visualisation

features = np.array([[1, 0], [0, 2], [1, 1], [1, 2], [1, 3], [2, 2], [2, 3], [3, 2]])
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def sigmoid(x):
    return np.exp(x) / (1 + np.exp(x))


def score(weights, bias, features):
    return np.dot(features, weights) + bias


def prediction(weights, bias, features):
    return sigmoid(score(weights, bias, features))


def log_loss(weights, bias, features, label):
    pred = prediction(weights, bias, features)
    return -label * np.log(pred) - (1 - label) * np.log(1 - pred)


def total_log_loss(weights, bias, features, labels):
    total_error = 0
    for i in range(len(features)):
        total_error += log_loss(weights, bias, features[i], labels[i])
    return total_error


def logistic_trick(weights, bias, features, label, learning_rate=0.01):
    pred = prediction(weights, bias, features)
    for i in range(len(weights)):
        weights[i] += (label - pred) * features[i] * learning_rate
        bias += (label - pred) * learning_rate
    return weights, bias


def logistic_regression_algorithm(features, labels, learning_rate=0.01, epochs=1000):
    weights = [1.0 for i in range(len(features[0]))]
    bias = 0.0
    errors = []
    for i in range(epochs):
        errors.append(total_log_loss(weights, bias, features, labels))
        j = random.randint(0, len(features) - 1)
        weights, bias = logistic_trick(weights, bias, features[j], labels[j])
    return weights, bias, errors


if __name__ == "__main__":
    weights, bias, errors = logistic_regression_algorithm(features, labels)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    predict_wrapper = lambda x: prediction(weights, bias, x)
    universal_visualisation(features, labels, predict_wrapper, ax[0])
    error_visualisation(errors, ax[1])
    plt.show()
