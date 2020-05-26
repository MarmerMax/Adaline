import numpy as np


def getSizeTrainAndTest(dataset, percent):
    size = len(dataset)
    train_size = round(size * percent)
    test_size = size - train_size
    return train_size, test_size


def splitData(data: np.ndarray, train_size: np.int, test_size: np.int, from_middle: bool) \
    -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    # print("size",size)
    # create sets for train
    train_set_features = np.empty((0, 33), float)
    train_set_diagnoses = np.array([])

    # create sets for test
    test_set_features = np.empty((0, 33), float)
    test_set_diagnoses = np.array([])

    counter = 0

    if not from_middle:
        for line in data:
            temp_x = np.array(line[2:])
            temp_y = line[1:2]
            if counter < train_size:
                train_set_features = np.vstack((train_set_features, temp_x))
                train_set_diagnoses = np.append(train_set_diagnoses, [temp_y])
            else:
                test_set_features = np.vstack((test_set_features, temp_x))
                test_set_diagnoses = np.append(test_set_diagnoses, [temp_y])
            counter += 1
    else:
        for line in data:
            temp_x = np.array(line[2:])
            temp_y = line[1:2]
            if counter < test_size:
                train_set_features = np.vstack((train_set_features, temp_x))
                train_set_diagnoses = np.append(train_set_diagnoses, [temp_y])
            elif test_size <= counter and counter < test_size * 2:
                test_set_features = np.vstack((test_set_features, temp_x))
                test_set_diagnoses = np.append(test_set_diagnoses, [temp_y])
            else:
                train_set_features = np.vstack((train_set_features, temp_x))
                train_set_diagnoses = np.append(train_set_diagnoses, [temp_y])

            counter += 1

    return train_set_features, train_set_diagnoses, test_set_features, test_set_diagnoses


# function to standardize values
def standardization(matrix):
    matrix_std = np.copy(matrix)
    for i in range(0, len(matrix_std[0])):
        matrix_std[:, i] = (matrix[:, i] - matrix[:, i].mean()) / matrix[:, i].std()
    return matrix_std
