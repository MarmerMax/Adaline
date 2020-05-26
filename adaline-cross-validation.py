import InputReader
import numpy as np
import handleData
import AdalineAlgorithm
import matplotlib.pyplot as plt

# get dataset from file
dataset = InputReader.get_dataset()

# calculate number equals to 66% of dataset length
train_size, test_size = handleData.getSizeTrainAndTest(dataset, 0.66)

# create train sets and test sets
train_set_features_1, train_set_diagnoses_1, test_set_features_1, test_set_diagnoses_1 \
    = handleData.splitData(dataset, train_size, test_size, False)

train_set_features_2, train_set_diagnoses_2, test_set_features_2, test_set_diagnoses_2 \
    = handleData.splitData(dataset, train_size, test_size, True)

test_set_features_3, test_set_diagnoses_3, train_set_features_3, train_set_diagnoses_3 \
    = handleData.splitData(dataset, test_size, train_size, False)

# print(f"1 train set: {len(train_set_features_1)} train_set res: {len(train_set_diagnoses_1)} test set: {len(test_set_features_1)} test set res: {len(test_set_diagnoses_1)}")
# print(f"2 train set: {len(train_set_features_2)} train_set res: {len(train_set_diagnoses_2)} test set: {len(test_set_features_2)} test set res: {len(test_set_diagnoses_2)}")
# print(f"3 train set: {len(train_set_features_3)} train_set res: {len(train_set_diagnoses_3)} test set: {len(test_set_features_3)} test set res: {len(test_set_diagnoses_3)}")

# standardize sets
train_set_features_1 = handleData.standardization(train_set_features_1)
test_set_features_1 = handleData.standardization(test_set_features_1)
train_set_features_2 = handleData.standardization(train_set_features_2)
test_set_features_2 = handleData.standardization(test_set_features_2)
train_set_features_3 = handleData.standardization(train_set_features_3)
test_set_features_3 = handleData.standardization(test_set_features_3)

# create array of weights
weights_1 = np.zeros(1 + train_set_features_1.shape[1])

# train first set
weights_1, costs_1 = AdalineAlgorithm.train(train_set_features_1, train_set_diagnoses_1, weights_1, 0.0001)

# predict target of tests features by our weights
predictions_1 = AdalineAlgorithm.predict(test_set_features_1, weights_1, 0.0, 1, -1)

# calculate real positives and negatives examples
actual_recurred_1, actual_not_recurred_1 = handleData.calculateActual(test_set_diagnoses_1)

# check prediction correctness
true_positive_1, false_negative_1, false_positive_1, true_negative_1 = handleData.checkPredictions(test_set_diagnoses_1,
                                                                                           predictions_1)

sum_actual_and_predicted_recurred = true_positive_1 / actual_recurred_1
sum_actual_not_recurred_and_predicted_not_recurred = true_negative_1 / actual_not_recurred_1

# create array of weights
weights_2 = np.zeros(1 + train_set_features_2.shape[1])

# train first set
weights_2, costs_2 = AdalineAlgorithm.train(train_set_features_2, train_set_diagnoses_2, weights_2, 0.0001)

# predict target of tests features by our weights
predictions_2 = AdalineAlgorithm.predict(test_set_features_2, weights_2, 0.0, 1, -1)

# calculate real positives and negatives examples
actual_recurred_2, actual_not_recurred_2 = handleData.calculateActual(test_set_diagnoses_2)

# check prediction correctness
true_positive_2, false_negative_2, false_positive_2, true_negative_2 = handleData.checkPredictions(test_set_diagnoses_2,
                                                                                           predictions_2)

sum_actual_and_predicted_recurred += true_positive_2 / actual_recurred_2
sum_actual_not_recurred_and_predicted_not_recurred += true_negative_2 / actual_not_recurred_2


# create array of weights
weights_3 = np.zeros(1 + train_set_features_3.shape[1])

# train first set
weights_3, costs_3 = AdalineAlgorithm.train(train_set_features_3, train_set_diagnoses_3, weights_3, 0.0001)

# predict target of tests features by our weights
predictions_3 = AdalineAlgorithm.predict(test_set_features_3, weights_3, 0.0, 1, -1)

# calculate real positives and negatives examples
actual_recurred_3, actual_not_recurred_3 = handleData.calculateActual(test_set_diagnoses_3)

# check prediction correctness
true_positive_3, false_negative_3, false_positive_3, true_negative_3 = handleData.checkPredictions(test_set_diagnoses_3,
                                                                                           predictions_3)

sum_actual_and_predicted_recurred += true_positive_3 / actual_recurred_3
sum_actual_not_recurred_and_predicted_not_recurred += true_negative_3 / actual_not_recurred_3

# print(sum_actual_and_predicted_recurred)

print(f"First method result: cross-validation")
print(f"true positive: {round(sum_actual_and_predicted_recurred / 3, 2)}%")
# print(f"false negative: {round(predicted_not_recurred / 3)}")
# print(f"false positive: {round(predicted_recurred / 3)}")
print(f"true negative: {round(sum_actual_not_recurred_and_predicted_not_recurred / 3, 2)}%")
