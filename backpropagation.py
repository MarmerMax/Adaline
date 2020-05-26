import InputReader
import handleData
import numpy as np
import tensorflow as tf

dataset = InputReader.get_dataset()

# calculate number equals to 66% of dataset length
train_size, test_size = handleData.getSizeTrainAndTest(dataset, 0.66)

print("train_size", train_size)
print("test_size", test_size)

train_set_features, train_set_diagnoses, test_set_features, test_set_diagnoses = handleData.splitData(dataset,
                                                                                                      train_size)

# standardize sets
train_set_features = handleData.standardization(train_set_features)
test_set_features = handleData.standardization(test_set_features)

train_set_diagnoses = train_set_diagnoses.reshape(train_set_diagnoses.shape[0], -1)  # (128,) -> (128,1)
# test_set_diagnoses = test_set_diagnoses.reshape(test_set_diagnoses.shape[0], -1)  # (66,) -> (66,1)

print("train_set_features", train_set_features.shape)
print("train_set_diagnoses", train_set_diagnoses.shape)
print("test_set_features", test_set_features.shape)
print("test_set_diagnoses", test_set_diagnoses.shape)

features = 33
output = 1

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, output])

(hidden1_size, hidden2_size) = (10, 5)
w_1 = tf.Variable(tf.truncated_normal([features, hidden1_size], stddev=0.1))
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
    sess.run(update, feed_dict={x: train_set_features, y_: train_set_diagnoses})
    if i % 1000 == 0:
        print('Iteration:', i, ' W2:', sess.run(w_2), ' b2:', sess.run(b_2), ' loss:', loss.eval(session=sess, feed_dict={x: train_set_features, y_: train_set_diagnoses}))


# function to predict value
def predict(test_data):
    val = y.eval(session=sess, feed_dict={x: test_data})
    return np.where(val >= 0.5, 1, -1)


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