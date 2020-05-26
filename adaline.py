import InputReader
import numpy as np
import handleData
import AdalineAlgorithm
import matplotlib.pyplot as plt
import time

# start time report
start_time = time.time()

# get dataset from file
dataset = InputReader.get_dataset()

# calculate number equals to 66% of dataset length
train_size, test_size = handleData.getSizeTrainAndTest(dataset, 0.66)

# create sets for train data, create sets for test data
# standardize sets
train_set_features, train_set_diagnoses, test_set_features, test_set_diagnoses \
    = handleData.splitData(dataset, train_size, test_size, False)
train_set_features = handleData.standardization(train_set_features)
test_set_features = handleData.standardization(test_set_features)

# create array of weights
# train the algorithm on given data and weights
weights = np.zeros(1 + train_set_features.shape[1])
weights, costs = AdalineAlgorithm.train(train_set_features, train_set_diagnoses, weights, 0.0001)

# Plot the training error
plt.plot(range(1, len(costs) + 1), costs, color='red')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

# predict target of tests features by our weights
# calculate real positives and negatives examples
# check prediction correctness
predictions = AdalineAlgorithm.predict(test_set_features, weights, 0.0, 1, -1)
actual_recurred, actual_not_recurred = handleData.calculateActual(test_set_diagnoses)
true_positive, false_negative, false_positive, true_negative = handleData.checkPredictions(test_set_diagnoses,
                                                                                           predictions)

print(f"First method: 66%-33%")
print(f"true positive: {round(true_positive / actual_recurred, 2)}%")
print(f"false negative: {round(false_negative / actual_recurred, 2)}%")
print(f"false positive: {round(false_positive / actual_not_recurred, 2)}%")
print(f"true negative: {round(true_negative / actual_not_recurred, 2)}%")
print(f"Code execution time: {round(time.time() - start_time, 2)} seconds")
