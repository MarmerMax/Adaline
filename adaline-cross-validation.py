import InputReader
import numpy as np
import handleData
import matplotlib.pyplot as plt

# get dataset from file
dataset = InputReader.get_dataset()

# calculate number equals to 66% of dataset length
train_size, test_size = handleData.getSizeTrainAndTest(dataset, 0.66)

train_set_features_1, train_set_diagnoses_1, test_set_features_1, test_set_diagnoses_1 \
    = handleData.splitData(dataset, train_size, test_size, False)

train_set_features_2, train_set_diagnoses_2, test_set_features_2, test_set_diagnoses_2 \
    = handleData.splitData(dataset, train_size, test_size, True)

test_set_features_3, test_set_diagnoses_3, train_set_features_3, train_set_diagnoses_3 \
    = handleData.splitData(dataset, test_size, train_size, False)

print(f"1 train set: {len(train_set_features_1)} train_set res: {len(train_set_diagnoses_1)} test set: {len(test_set_features_1)} test set res: {len(test_set_diagnoses_1)}")
print(f"2 train set: {len(train_set_features_2)} train_set res: {len(train_set_diagnoses_2)} test set: {len(test_set_features_2)} test set res: {len(test_set_diagnoses_2)}")
print(f"3 train set: {len(train_set_features_3)} train_set res: {len(train_set_diagnoses_3)} test set: {len(test_set_features_3)} test set res: {len(test_set_diagnoses_3)}")

# standardize sets
# train_set_features = handleData.standardization(train_set_features)
# test_set_features = handleData.standardization(test_set_features)


# # function to predict value
# def predict(test_data):
#     return np.where(np.dot(test_data, w[1:]) + w[0] >= 0.0, 1, -1)
#
#
# # create array of weights
# w = np.zeros(1 + train_set_features.shape[1])
#
# # learning rate
# eta = 0.0001
# costs = np.array([])
#
# # fit the weights
# for iteration in range(1000):
#     outputs = np.dot(train_set_features, w[1:]) + w[0]
#     errors = (train_set_diagnoses - outputs)
#     w[1:] += eta * np.dot(train_set_features.T, errors)
#     w[0] += eta * errors.sum()
#     cost = (errors ** 2).sum() / 2.0
#     costs = np.append(costs, [cost])
#
# # Plot the training error
# plt.plot(range(1, len(costs) + 1), costs, marker='o', color='red')
# plt.xlabel('Epochs')
# plt.ylabel('Sum-squared-error')
# plt.show()
#
# predicted_recurred = 0
# predicted_not_recurred = 0
#
# actual_recurred = 0
# actual_not_recurred = 0
#
# actual_and_predicted_recurred = 0
# actual_not_recurred_and_predicted_not_recurred = 0
#
# for patient in test_set_diagnoses:
#     if patient == 1:
#         actual_recurred += 1
#     else:
#         actual_not_recurred += 1
#
# predictions = predict(test_set_features)
#
# for i in range(0, len(predictions)):
#     if test_set_diagnoses[i] == 1 and predictions[i] == 1:
#         actual_and_predicted_recurred += 1
#     elif test_set_diagnoses[i] == 1 and predictions[i] == -1:
#         predicted_not_recurred += 1
#     elif test_set_diagnoses[i] == -1 and predictions[i] == 1:
#         predicted_recurred += 1
#     else:
#         actual_not_recurred_and_predicted_not_recurred += 1
#
# print(f"First method result: 66%-33%")
# print(f"true positive: {actual_and_predicted_recurred}")
# print(f"false negative: {predicted_not_recurred}")
# print(f"false positive: {predicted_recurred}")
# print(f"true negative: {actual_not_recurred_and_predicted_not_recurred}")
