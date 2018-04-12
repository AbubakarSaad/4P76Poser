import tensorflow as tf
import numpy as np
import sys
np.set_printoptions(threshold=np.nan)


squattingData = np.genfromtxt(sys.path[0] + '\\tf-pose-estimation\\src\\trainingSquattingClean.csv', delimiter=',')
standingData = np.genfromtxt(sys.path[0] + '\\tf-pose-estimation\\src\\trainingStandingClean.csv', delimiter=',')


# Sample Dataset
x_data = np.concatenate((squattingData , standingData), axis=0)
y_data = np.concatenate( (np.ones(len(squattingData)), np.zeros(len(standingData))), axis=0)


print(len(x_data))
print(len(y_data))

# output_layerperparamters
n_input = 36
n_hidden = 50
n_output = 1
learning_rate = 0.1
epochs = 1000
display_step = 10

# Placeholders
inputs = tf.placeholder(tf.float32)
expected_output = tf.placeholder(tf.float32)

# Weights
weights_1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
weights_2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))

# Bias
bias1 = tf.Variable(tf.zeros([n_hidden]))
bias2 = tf.Variable(tf.zeros([n_output]))

sigmoided_hidden_layer = tf.sigmoid(tf.matmul(inputs, weights_1) + bias1)
sigmoided_output_layer = tf.sigmoid(tf.matmul(sigmoided_hidden_layer, weights_2) + bias2)


cost = tf.reduce_sum(tf.square(expected_output - sigmoided_output_layer))
#cost = tf.reduce_mean(- expected_output * tf.log(sigmoided_output_layer) - (1-expected_output) * tf.log(1 - sigmoided_output_layer))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(epochs):
        sess.run(optimizer, feed_dict = {inputs: x_data, expected_output: y_data})
        print(sess.run(cost, feed_dict = {inputs: x_data, expected_output: y_data}))
        if step % display_step == 0:
            #print(sess.run(cost, feed_dict = {inputs: x_data, expected_output: y_data}))
            continue

    print(sess.run([sigmoided_output_layer], feed_dict = {inputs: x_data, expected_output: y_data}))
    