import InputReader
import numpy as np

# get dataset from file
dataset = InputReader.get_dataset()

# create sets for train
train_set_features = np.empty((0, 33), float)
train_set_diagnoses = np.array([])

# create sets for test
test_set_features = np.empty((0, 33), float)
test_set_diagnoses = np.array([])

# calculate number equals to 66% of dataset length
size = len(dataset)
train_size = round(size * 0.66)
test_size = size - train_size
counter = 0

# split dataset 66% - train set, 33% - test set
for line in dataset:
    temp_x = np.array(line[2:])
    temp_y = line[1:2]
    if counter < train_size:
        train_set_features = np.vstack((train_set_features, temp_x))
        train_set_diagnoses = np.append(train_set_diagnoses, [temp_y])
    else:
        test_set_features = np.vstack((test_set_features, temp_x))
        test_set_diagnoses = np.append(test_set_diagnoses, [temp_y])
    counter += 1


# function to predict value
def predict(test_data):
    return np.where(np.dot(test_data, w[1:]) + w[0] >= 0.0, 1, -1)


# function to standardize values
def standardization(matrix):
    matrix_std = np.copy(matrix)
    for i in range(0, len(matrix_std[0])):
        matrix_std[:, i] = (matrix[:, i] - matrix[:, i].mean()) / matrix[:, i].std()
    return matrix_std


# standardize sets
train_set_features = standardization(train_set_features)
test_set_features = standardization(test_set_features)

# create array of weights
w = np.zeros(1 + train_set_features.shape[1])
# learning rate
eta = 0.0001
# costs = np.array([])

# fit the weights
for iteration in range(1000):
    outputs = np.dot(train_set_features, w[1:]) + w[0]
    errors = (train_set_diagnoses - outputs)
    w[1:] += eta * np.dot(train_set_features.T, errors)
    w[0] += eta * errors.sum()
    # cost = (errors ** 2).sum() / 2.0
    # costs = np.append(costs, [cost])

# print(w)

predicted_recurred = 0
predicted_not_recurred = 0

actual_recurred = 0
actual_not_recurred = 0

actual_and_predicted_recurred = 0
actual_not_recurred_and_predicted_not_recurred = 0

for patient in test_set_diagnoses:
    if patient == 1:
        actual_recurred += 1
    else:
        actual_not_recurred += 1

predictions = predict(test_set_features)

for i in range(0, len(predictions)):
    if test_set_diagnoses[i] == 1 and predictions[i] == 1:
        actual_and_predicted_recurred += 1
    elif test_set_diagnoses[i] == 1 and predictions[i] == -1:
        predicted_not_recurred += 1
    elif test_set_diagnoses[i] == -1 and predictions[i] == 1:
        predicted_recurred += 1
    else:
        actual_not_recurred_and_predicted_not_recurred += 1


print(f"true positive: {actual_and_predicted_recurred}")
print(f"false negative: {predicted_not_recurred}")
print(f"false positive: {predicted_recurred}")
print(f"true negative: {actual_not_recurred_and_predicted_not_recurred}")
