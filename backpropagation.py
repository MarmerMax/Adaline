import InputReader
import handleData
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time



def train(features, targets):

    features_size = len(features[0])
    output = 1

    x = tf.placeholder(tf.float32, [None, features_size])
    y_ = tf.placeholder(tf.float32, [None, output])

    (hidden1_size, hidden2_size) = (10, 5)
    w_1 = tf.Variable(tf.truncated_normal([features_size, hidden1_size], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
    z_1 = tf.nn.sigmoid(tf.matmul(x, w_1) + b_1)

    w_2 = tf.Variable(tf.truncated_normal([hidden1_size, output], stddev=0.1))
    b_2 = tf.Variable(0.1, [output], dtype=tf.float32)
    # b_2 = tf.Variable(0.)

    y = tf.nn.sigmoid(tf.matmul(z_1, w_2) + b_2)
    loss = tf.reduce_sum(tf.square(y_ - y))

    update = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    costs = np.array([])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(0, 10000):
        sess.run(update, feed_dict={x: features, y_: targets})
        if i % 1000 == 0:
            # print('Iteration:', i, ' W2:', sess.run(w_2), ' b2:', sess.run(b_2), ' loss:',
            #       loss.eval(session=sess, feed_dict={x: features, y_: targets}))
            costs = np.append(costs, [loss.eval(session=sess, feed_dict={x: features, y_: targets})])


    return x, y, sess, costs


# function to predict value
def predict(x, y,sess,features_test):
    val = y.eval(session=sess, feed_dict={x: features_test})
    return np.where(val >= 0.5, 1, -1)


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


#####
# back-prop with cross-validation
#####
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
