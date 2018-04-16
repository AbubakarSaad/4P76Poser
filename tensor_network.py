import tensorflow as tf
import numpy as np
import sys
import math
import importlib
dataGenerator = importlib.import_module('tf-pose-estimation.src.dataScriptGenerator', None)

dataGenerator.dataScriptGenerator()


np.set_printoptions(threshold=np.nan)

# Network Parameters
n_input_nodes = 36
n_nodes_hl1 = 50
n_nodes_hl2 = 50
n_output_node = 2
hm_epochs = 100
learning_rate = 0.01
minWeight = -1.0
maxWeight = 1.0
trainingPercent = 0.7
testingPercent = 0.3

squattingData = np.genfromtxt(sys.path[0] + r'/tf-pose-estimation/src/trainingSquattingClean.csv', delimiter=',')
standingData = np.genfromtxt(sys.path[0] + r'/tf-pose-estimation/src/trainingStandingClean.csv', delimiter=',')

squattingDataExpected = np.tile([1,0], (squattingData.shape[0],1))
standingDataExpected = np.tile([0,1], (standingData.shape[0],1))


# Sample Dataset
data_x = np.concatenate((squattingData , standingData), axis=0)
data_y = np.concatenate((squattingDataExpected, standingDataExpected), axis=0)

# replace -1 with 0 in the whole dataset.
data_x[data_x < 0] = 0

# The specifc size for each dataset
trainingDataSize = math.floor(data_x.shape[0] * trainingPercent)
testingDataSize = math.ceil(data_x.shape[0] * testingPercent) + trainingDataSize

# The training data 
trainingData = data_x[0 : trainingDataSize]
trainingDataClass = data_y[0 : trainingDataSize]

# The testing data
testingData = data_x[trainingDataSize : testingDataSize]
testingDataClass = data_y[trainingDataSize : testingDataSize]

# The combined data for shuffling purposes.
xy_dataTraining = np.concatenate((trainingData, trainingDataClass), axis=1)
xy_dataTesting = np.concatenate((testingData, testingDataClass), axis=1)

#print(xy_dataTraining)

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
            accuracy = 0
            
            np.random.shuffle(xy_dataTraining)
            data_x = xy_dataTraining[0:trainingDataSize, 0:36]
            data_y = xy_dataTraining[0:trainingDataSize, 36:38]

            # print(data_y)

            data_y = np.resize(data_y, (328, 2))

            # print(data_y[0])


            for piece in range(len(data_x)):
                input_x =  [data_x[piece]]
                expected_y = [data_y[piece]]
                _, c, predict = sess.run([optimizer, cost, prediction], feed_dict={x: input_x, y: expected_y})

                epoch_loss += c
                # print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
                # print(predict[0], " " , expected_y[0])
                if np.argmax(predict[0]) == np.argmax(expected_y[0]):
                    accuracy += 1


            print("Epoch: ", epoch, ", accuracy_standing: ", (accuracy/len(data_x)))



        # TESTING OUR NETWORK, SAME SESSION

        accuracy = 0

        data_x = xy_dataTesting[0:testingDataSize, 0:36]
        data_y = xy_dataTesting[0:testingDataSize, 36:38]
        
        for piece in range(len(data_x)):
            input_x =  [data_x[piece]]
            expected_y = [data_y[piece]]
            predict = sess.run([prediction], feed_dict={x: input_x, y: expected_y})

            print(predict[0], " " , expected_y[0],2)
            if np.argmax(predict[0]) == np.argmax(expected_y[0]):
                accuracy += 1

        print("Testing Accuracy on Entire Set:  ", (accuracy/len(data_x)))


        # Input single picture into network:

        # - Call data script generator on new image
        # dataGeneration()
        # - Pass new results into network and get prediction

        # accuracy = 0

        # data_x = xy_dataTesting[0:testingDataSize, 0:36]
        # data_y = xy_dataTesting[0:testingDataSize, 36:38]
        
        # for piece in range(len(data_x)):
        #     input_x =  [data_x[piece]]
        #     expected_y = [data_y[piece]]
        #     predict = sess.run([prediction], feed_dict={x: input_x, y: expected_y})

        #     print(predict[0], " " , expected_y[0],2)
        #     if np.argmax(predict[0]) == np.argmax(expected_y[0]):
        #         accuracy += 1

        # print("Testing Accuracy on Entire Set:  ", (accuracy/len(data_x)))


train_neural_network(x)