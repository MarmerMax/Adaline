import InputReader
import numpy as np
import handleData
import AdalineAlgorithm
import matplotlib.pyplot as plt
import time

# start time report
import backpropagation

start_time = time.time()

# get dataset from file
dataset = InputReader.get_dataset()

# calculate number equals to 66% of dataset length
train_size, test_size = handleData.getSizeTrainAndTest(dataset, 0.66)

# create tree train sets and tree test sets
train_set_features_1, train_set_diagnoses_1, test_set_features_1, test_set_diagnoses_1 \
    = handleData.splitData(dataset, train_size, test_size, False)

train_set_features_2, train_set_diagnoses_2, test_set_features_2, test_set_diagnoses_2 \
    = handleData.splitData(dataset, train_size, test_size, True)

test_set_features_3, test_set_diagnoses_3, train_set_features_3, train_set_diagnoses_3 \
    = handleData.splitData(dataset, test_size, train_size, False)

# standardize sets
train_set_features_1 = handleData.standardization(train_set_features_1)
test_set_features_1 = handleData.standardization(test_set_features_1)
train_set_features_2 = handleData.standardization(train_set_features_2)
test_set_features_2 = handleData.standardization(test_set_features_2)
train_set_features_3 = handleData.standardization(train_set_features_3)
test_set_features_3 = handleData.standardization(test_set_features_3)

print("first features set:")
# working on first features set
# 1. create array of weights
# 2. train first set
# 3. predict target of tests features by our weights
# 4. calculate real positives and negatives examples
# 5. check prediction correctness
weights_1 = np.zeros(1 + train_set_features_1.shape[1])
weights_1, costs_1 = AdalineAlgorithm.train(train_set_features_1, train_set_diagnoses_1, weights_1, 0.0001)
predictions_1 = AdalineAlgorithm.predict(test_set_features_1, weights_1, 0.0, 1, -1)
actual_recurred_1, actual_not_recurred_1 = handleData.calculateActual(test_set_diagnoses_1)
true_positive_1, false_negative_1, false_positive_1, true_negative_1 = handleData.checkPredictions(test_set_diagnoses_1,
                                                                                                   predictions_1)
print("second features set:")
# working on second features set
weights_2 = np.zeros(1 + train_set_features_2.shape[1])
weights_2, costs_2 = AdalineAlgorithm.train(train_set_features_2, train_set_diagnoses_2, weights_2, 0.0001)
predictions_2 = AdalineAlgorithm.predict(test_set_features_2, weights_2, 0.0, 1, -1)
actual_recurred_2, actual_not_recurred_2 = handleData.calculateActual(test_set_diagnoses_2)
true_positive_2, false_negative_2, false_positive_2, true_negative_2 = handleData.checkPredictions(test_set_diagnoses_2,

                                                                                                   predictions_2)
print("third features set:")
# working on third features set
weights_3 = np.zeros(1 + train_set_features_3.shape[1])
weights_3, costs_3 = AdalineAlgorithm.train(train_set_features_3, train_set_diagnoses_3, weights_3, 0.0001)
predictions_3 = AdalineAlgorithm.predict(test_set_features_3, weights_3, 0.0, 1, -1)
actual_recurred_3, actual_not_recurred_3 = handleData.calculateActual(test_set_diagnoses_3)
true_positive_3, false_negative_3, false_positive_3, true_negative_3 = handleData.checkPredictions(test_set_diagnoses_3,
                                                                                                   predictions_3)
# Plot the training error
plt.plot(range(1, len(costs_1) + 1), costs_1, color='red', label='set1', linewidth=5.0)
plt.plot(range(1, len(costs_2) + 1), costs_2, color='green', label='set2', linewidth=5.0)
plt.plot(range(1, len(costs_3) + 1), costs_3, color='blue', label='set3', linewidth=5.0)
plt.plot(range(1, len(costs_3) + 1), 1 / 3 * (costs_1 + costs_2 + costs_3), color='black', label='avg')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.legend(loc='best')
plt.show()

# sum of true predictions
sum_true_positive = \
    (true_positive_1 / actual_recurred_1) + \
    (true_positive_2 / actual_recurred_2) + \
    (true_positive_3 / actual_recurred_3)

sum_true_negative = \
    (true_negative_1 / actual_not_recurred_1) + \
    (true_negative_2 / actual_not_recurred_2) + \
    (true_negative_3 / actual_not_recurred_3)

print(f"Second method: cross-validation")
print(f"true positive: {round(sum_true_positive / 3, 2)}%")
# print(f"false negative: {round(predicted_not_recurred / 3)}")
# print(f"false positive: {round(predicted_recurred / 3)}")
print(f"true negative: {round(sum_true_negative / 3, 2)}%")
print(f"Code execution time: {round(time.time() - start_time, 2)} seconds")

backprop = False

#####
# back-prop with cross-validation
#####
if backprop:
    print("###########\n Back-Propagation\n ###########")
    train_set_diagnoses_1 = train_set_diagnoses_1.reshape(train_set_diagnoses_1.shape[0], -1)  # (128,) -> (128,1)
    train_set_diagnoses_2 = train_set_diagnoses_2.reshape(train_set_diagnoses_2.shape[0], -1)  # (128,) -> (128,1)
    train_set_diagnoses_3 = train_set_diagnoses_3.reshape(train_set_diagnoses_3.shape[0], -1)  # (128,) -> (128,1)

    # train set 1
    x_1, y_1, sess_1, costs_back_1 = backpropagation.train(train_set_features_1,train_set_diagnoses_1)
    # train set 2
    x_2, y_2, sess_2, costs_back_2 = backpropagation.train(train_set_features_2,train_set_diagnoses_2)
    # train set 3
    x_3, y_3, sess_3, costs_back_3 = backpropagation.train(train_set_features_3,train_set_diagnoses_3)


    # Plot the training error
    plt.plot(range(1 * 1000, (len(costs_back_1) + 1) * 1000, 1000), costs_back_1, color='red', label='set1', linewidth=5.0)
    plt.plot(range(1 * 1000, (len(costs_back_2) + 1) * 1000, 1000), costs_back_2, color='green', label='set2', linewidth=5.0)
    plt.plot(range(1 * 1000, (len(costs_back_3) + 1) * 1000, 1000), costs_back_3, color='blue', label='set3', linewidth=5.0)
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.legend(loc='best')
    plt.show()

    # test 1
    print("##set1:")
    handleData.checkPredictions(test_set_diagnoses_1,backpropagation.predict(x_1,y_1,sess_1,test_set_features_1))
    # test 2
    print("##set2:")
    handleData.checkPredictions(test_set_diagnoses_1,backpropagation.predict(x_2,y_2,sess_2,test_set_features_2))
    # test 3
    print("##set3:")
    handleData.checkPredictions(test_set_diagnoses_1,backpropagation.predict(x_3,y_3,sess_3,test_set_features_3))