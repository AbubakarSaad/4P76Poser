import tensorflow as tf
import numpy as np
import sys
np.set_printoptions(threshold=np.nan)


squattingData = np.genfromtxt(sys.path[0] + '\\tf-pose-estimation\\src\\trainingSquattingClean.csv', delimiter=',')
standingData = np.genfromtxt(sys.path[0] + '\\tf-pose-estimation\\src\\trainingStandingClean.csv', delimiter=',')

squattingDataExpected = np.ones((squattingData.shape[0], 1))
standingDataExpected = np.zeros((standingData.shape[0], 1))

# Sample Dataset
x_data = np.concatenate((squattingData , standingData), axis=0)
y_data = np.concatenate((squattingDataExpected, standingDataExpected), axis=0)

print(x_data.shape)
print(y_data.shape)

# output_layerperparamters
n_input = 36
n_hidden = 10
n_output = 1
learning_rate = 0.001
epochs = 10000
display_step = 10

# Placeholders
inputs = tf.placeholder(tf.float32)
expected_output = tf.placeholder(tf.float32)

# Weights
weights_1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
weights_2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

# Bias
bias1 = tf.Variable(tf.random_uniform([n_hidden], -1.0, 1.0))
bias2 = tf.Variable(tf.random_uniform([n_output], -1.0, 1.0))

sigmoided_hidden_layer = tf.sigmoid(tf.matmul(inputs, weights_1) + bias1)
sigmoided_output_layer = tf.sigmoid(tf.matmul(sigmoided_hidden_layer, weights_2) + bias2)
sigmoided_output_layer = tf.Print(sigmoided_output_layer, [sigmoided_output_layer], "Output Layer:")

cost = tf.reduce_sum(tf.square(expected_output - sigmoided_output_layer))
#cost = tf.reduce_mean(- expected_output * tf.log(sigmoided_output_layer) - (1-expected_output) * tf.log(1 - sigmoided_output_layer))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(epochs):
        sess.run(optimizer, feed_dict = {inputs: x_data, expected_output: y_data})
        print(sess.run(cost, feed_dict = {inputs: x_data, expected_output: y_data}))

    #print(sess.run([sigmoided_output_layer], feed_dict = {inputs: x_data, expected_output: y_data}))
    