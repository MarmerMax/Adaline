import InputReader
# import Adaline
import numpy as np

dataset = InputReader.get_dataset()

train_set_features = np.empty((0, 33), float)
train_set_predictions = np.array([])

test_set_features = np.empty((0, 33), float)
test_set_predictions = np.array([])

size = len(dataset)
train_size = round(size * 0.66)
test_size = size - train_size
counter = 0

for line in dataset:
    temp_x = np.array(line[2:])
    temp_y = line[1:2]
    if counter < train_size:
        train_set_features = np.vstack((train_set_features, temp_x))
        train_set_predictions = np.append(train_set_predictions, [temp_y])
    else:
        test_set_features = np.vstack((test_set_features, temp_x))
        test_set_predictions = np.append(test_set_predictions, [temp_y])
    counter += 1

w = np.zeros(1 + train_set_features.shape[1])
eta = 0.00001
costs = np.array([])


for i in range(30):
    output = np.dot(train_set_features, w[1:]) + w[0]
    errors = (train_set_predictions - output)
    w[1:] += + eta * np.dot(train_set_features.T, errors)
    w[0] += eta * errors.sum()
    print(w[0])
    # cost = (errors ** 2).sum() / 2.0
    # costs = np.append(costs, [cost])
