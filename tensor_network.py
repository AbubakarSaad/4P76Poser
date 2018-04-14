import tensorflow as tf
import numpy as np
import sys


squattingData = np.genfromtxt(sys.path[0] + r'/tf-pose-estimation/src/trainingSquattingClean.csv', delimiter=',')
standingData = np.genfromtxt(sys.path[0] + r'/tf-pose-estimation/src/trainingStandingClean.csv', delimiter=',')

squattingDataExpected = np.ones((squattingData.shape[0], 1))
standingDataExpected = np.zeros((standingData.shape[0], 1))

# Sample Dataset
data_x = np.concatenate((squattingData , standingData), axis=0)
data_y = np.concatenate((squattingDataExpected, standingDataExpected), axis=0)

# replace -1 with 0
data_x[data_x < 0] = 0

# The combined data data for shuffling purposes.
xy_data = np.concatenate((data_x, data_y), axis=1)

# The combined data data for shuffling purposes.
xy_data = np.concatenate((data_x, data_y), axis=1)

# Network Parameters
n_input_nodes = 36
n_nodes_hl1 = 50
n_nodes_hl2 = 50

n_output_node = 1
hm_epochs = 100
learning_rate = 0.01
minWeight = -1.0
maxWeight = 1.0


x = tf.placeholder('float', [None, n_input_nodes])
y = tf.placeholder('float', [None, n_output_node])

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_uniform([n_input_nodes, n_nodes_hl1], minWeight, maxWeight)),
                      'biases':tf.Variable(tf.random_uniform([n_nodes_hl1], minWeight, maxWeight))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_uniform([n_nodes_hl1, n_nodes_hl2], minWeight, maxWeight)),
                      'biases':tf.Variable(tf.random_uniform([n_nodes_hl2], minWeight, maxWeight))}

    output_layer = {'weights':tf.Variable(tf.random_uniform([n_nodes_hl2, n_output_node], minWeight, maxWeight)),
                    'biases':tf.Variable(tf.random_uniform([n_output_node], minWeight, maxWeight))}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.sigmoid(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.sigmoid(l2)

    output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']
    output = tf.nn.sigmoid(output)
    # output = tf.Print(output, [output], "Output Layer: ")

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)

    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    cost = tf.reduce_sum(tf.square(y - prediction))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            accuracy_standing = 0
            accuracy_squating = 0
            
            np.random.shuffle(xy_data)
            data_x = xy_data[0:328, 0:36]
            data_y = xy_data[0:328, 36]
            data_y = np.resize(data_y, (328, 1))

            for piece in range(len(data_x)):
                input_x =  [data_x[piece]]
                expected_y = [data_y[piece]]
                _, c, predict = sess.run([optimizer, cost, prediction], feed_dict={x: input_x, y: expected_y})

                epoch_loss += c
                # print(expected_y[0], predict[0])
                # print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
                if predict[0] > 0.500000:
                    accuracy_squating += 1
                elif predict[0] < 0.50000:
                    accuracy_standing += 1
 
            print("Epoch: ", epoch, ", accuracy_squating: ", (accuracy_squating)/len(data_x))
            print("Epoch: ", epoch, ", accuracy_standing: ", (accuracy_standing)/len(data_x))
            my_acc = tf.reduce_sum(tf.cast(tf.equal(x, y), tf.float32))
            # print(sess.run(my_acc, feed_dict={x: input_x, y: expected_y}))  # 1.0


train_neural_network(x)



'''
CODE IM BASING MY STUFF ON

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_nodes_hl1 = 200
n_nodes_hl2 = 200
n_nodes_hl3 = 200

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)

'''