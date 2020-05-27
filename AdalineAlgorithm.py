import numpy as np


# function to train and fix weights
def train(features, targets, weights, learning_rate, epochs):
    costs = np.array([])

    for iteration in range(epochs):
        outputs = np.dot(features, weights[1:]) + weights[0]
        errors = (targets - outputs)
        weights[1:] += learning_rate * np.dot(features.T, errors)
        weights[0] += learning_rate * errors.sum()
        cost = (errors ** 2).sum() / 2.0
        costs = np.append(costs, [cost])

    return weights, costs


# function to predict value
def predict(features, weights, threshold, yes_value, no_value):
    return np.where(np.dot(features, weights[1:]) + weights[0] >= threshold, yes_value, no_value)
