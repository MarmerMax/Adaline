import InputReader
import handleData
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



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


