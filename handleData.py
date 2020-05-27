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


# calculate positives and negatives examples
def calculateActual(targets):
    positives = 0
    negatives = 0
    for item in targets:
        if item == 1:
            positives += 1
        else:
            negatives += 1

    return positives, negatives

# calculate prediction correctness
def checkPredictions(target, predictions):
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0
    for i in range(0, len(predictions)):
        if target[i] == 1 and predictions[i] == 1:
            true_positive += 1
        elif target[i] == 1 and predictions[i] == -1:
            false_negative += 1
        elif target[i] == -1 and predictions[i] == 1:
            false_positive += 1
        else:
            true_negative += 1

    print(f"true positive: {true_positive}")
    print(f"false negative: {false_negative}")
    print(f"false positive: {false_positive}")
    print(f"true negative: {true_negative}\n")

    checkScore(true_positive, false_negative, false_positive, true_negative)

    return true_positive, false_negative, false_positive, true_negative



def checkScore(true_positive, false_negative,false_positive,true_negative):
    total_all = true_positive + false_negative + false_positive + true_negative
    total_true =  true_positive + true_negative # total really positive
    acc = total_true /total_all
    total_positive = true_positive +false_positive # total classified as positive
    precision= true_positive / total_positive
    recall = true_positive/(true_positive+ false_negative)
    f_score = (2*recall*precision) / (recall + precision)

    print(f"acc: {acc}")
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"f_score: {f_score}\n")
