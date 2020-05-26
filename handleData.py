import numpy as np


def getSizeTrainAndTest(dataset, percent):
    size = len(dataset)
    train_size = round(size * 0.66)
    test_size = size - train_size
    return train_size, test_size


def splitData(data: np.ndarray, size: np.int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    print("size",size)
    # create sets for train
    train_set_features = np.empty((0, 33), float)
    train_set_diagnoses = np.array([])

    # create sets for test
    test_set_features = np.empty((0, 33), float)
    test_set_diagnoses = np.array([])

    counter = 0
    for line in data:
        temp_x = np.array(line[2:])
        temp_y = line[1:2]
        if counter < size:
            train_set_features = np.vstack((train_set_features, temp_x))
            train_set_diagnoses = np.append(train_set_diagnoses, [temp_y])
        else:
            test_set_features = np.vstack((test_set_features, temp_x))
            test_set_diagnoses = np.append(test_set_diagnoses, [temp_y])
        counter += 1

    return train_set_features, train_set_diagnoses, test_set_features, test_set_diagnoses


# function to standardize values
def standardization(matrix):
    matrix_std = np.copy(matrix)
    for i in range(0, len(matrix_std[0])):
        matrix_std[:, i] = (matrix[:, i] - matrix[:, i].mean()) / matrix[:, i].std()
    return matrix_std

