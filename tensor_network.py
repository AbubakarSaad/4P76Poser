import tensorflow as tf
import numpy as np
import sys
import math
import importlib
import real_time, liveCam
import winsound
import csv

sys.path.insert(0, sys.path[0] + "\\tf-pose-estimation/src")

dataGenerator = importlib.import_module('tf-pose-estimation.src.dataScriptGenerator', None)

np.set_printoptions(threshold=np.nan)

csvOutputName = "tests/Tanh.csv"

# The .CSV file to write to for each run
file = open(csvOutputName, 'w', newline='')
wr = csv.writer(file, quoting=csv.QUOTE_ALL)


# Network Parameters
n_input_nodes = 36
n_nodes_hl1 = 30
n_nodes_hl2 = 25
n_output_node = 2
hm_epochs = 100
learning_rate = 0.02
minWeight = -1.0
maxWeight = 1.0
trainingPercent = 0.7
testingPercent = 0.3
dropOutRateL1 = 0.5
dropOutRateL2 = 0.5
numOfRuns = 30

# Print the network params to the csv file.
wr.writerow([ ])
wr.writerow(["Epochs: ", hm_epochs])
wr.writerow(["Number of hidden nodes layer 1: ", n_nodes_hl1])
wr.writerow(["Number of hidden nodes layer 2: ", n_nodes_hl2])
wr.writerow(["Learning rate: ", learning_rate])
wr.writerow(["Drop out Rate Layer 1: ", dropOutRateL1])
wr.writerow(["Drop out Rate Layer 2: ", dropOutRateL2])
wr.writerow(["Training Percent: ", trainingPercent])
wr.writerow(["Testing Percent: ", testingPercent])
wr.writerow([ ])


squattingData = np.genfromtxt(sys.path[0] + '\\trainingSquattingClean.csv', delimiter=',')
standingData = np.genfromtxt(sys.path[0] + '\\trainingStandingClean.csv', delimiter=',')

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

x = tf.placeholder('float', [None, n_input_nodes])
y = tf.placeholder('float', [None, n_output_node])

# Probability for drop out in each layer.
keep_probL1 = tf.placeholder(tf.float32)
keep_probL2 = tf.placeholder(tf.float32)

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_uniform([n_input_nodes, n_nodes_hl1], minWeight, maxWeight)),
                      'biases':tf.Variable(tf.random_uniform([n_nodes_hl1], minWeight, maxWeight))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_uniform([n_nodes_hl1, n_nodes_hl2], minWeight, maxWeight)),
                      'biases':tf.Variable(tf.random_uniform([n_nodes_hl2], minWeight, maxWeight))}

    output_layer = {'weights':tf.Variable(tf.random_uniform([n_nodes_hl2, n_output_node], minWeight, maxWeight)),
                    'biases':tf.Variable(tf.random_uniform([n_output_node], minWeight, maxWeight))}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.tanh(l1)

    # Apply dropout to the first layer.
    dropOutL1 = tf.nn.dropout(l1, keep_probL1)

    l2 = tf.add(tf.matmul(dropOutL1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.tanh(l2)
    
    dropOutL2 = tf.nn.dropout(l2, keep_probL2)

    output = tf.matmul(dropOutL2, output_layer['weights']) + output_layer['biases']
    output = tf.nn.sigmoid(output)

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)

    cost = tf.reduce_sum(tf.square(y - prediction))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    wr.writerow([ "Run Number", "Training Accuracy", "Testing Accuracy" ])

    with tf.Session() as sess:

        ##########################
        # TRAINING FUNCTIONALITY #
        ##########################

        sess.run(tf.global_variables_initializer())
        # This loop will run of the number of total run required.
        for i in range(numOfRuns):  
            accuracy = 0

            for epoch in range(hm_epochs):
                epoch_loss = 0
                accuracy = 0
                
                np.random.shuffle(xy_dataTraining)
                data_x = xy_dataTraining[0:trainingDataSize, 0:36]
                data_y = xy_dataTraining[0:trainingDataSize, 36:38]

                data_y = np.resize(data_y, (328, 2))

                for piece in range(len(data_x)):
                    input_x =  [data_x[piece]]
                    expected_y = [data_y[piece]]
                    _, c, predict = sess.run([optimizer, cost, prediction], feed_dict={
                            x: input_x, 
                            y: expected_y, 
                            keep_probL1: dropOutRateL1,
                            keep_probL2: dropOutRateL2
                        })

                    epoch_loss += c

                    if np.argmax(predict[0]) == np.argmax(expected_y[0]):
                        accuracy += 1
                accuracyPercentage = (accuracy/len(data_x))
                print("Epoch: ", epoch, ", accuracy_standing: ", accuracyPercentage)
            runLabel = "Accuracy for run " + str(i) + ":"

            ######################
            # IMAGE TESTING BULK #
            ######################

            accuracy = 0
            accuracyPercentage2 = 0

            data_x = xy_dataTesting[0:testingDataSize, 0:36]
            data_y = xy_dataTesting[0:testingDataSize, 36:38]
            
            for piece in range(len(data_x)):
                input_x =  [data_x[piece]]
                expected_y = [data_y[piece]]
                predict = sess.run([prediction], feed_dict={
                        x: input_x, 
                        y: expected_y, 
                        keep_probL1: 1.0,
                        keep_probL2: 1.0
                    })

                
                if np.argmax(predict[0]) == np.argmax(expected_y[0]):
                    accuracy += 1
            accuracyPercentage2 = (accuracy/len(data_x))
            trialLabel = "Testing Accuracy on Entire Set:  "
            print(trialLabel, accuracyPercentage2)
            wr.writerow([i, accuracyPercentage, accuracyPercentage2])
        
        # Play a sound to signal the numOfRuns complete
        winsound.PlaySound("SystemExit", winsound.SND_ALIAS)

        ########################
        # AD HOC IMAGE TESTING #
        ########################

        # Preload cmu
        dG = dataGenerator.dataScriptGenerator()

        continueProcessing = "N"

        while(continueProcessing == "Y" or continueProcessing == "y"):
            
            # Call data script generator on new image and generate csv
            dG.adHocData()
            
            # Pass new csv data into network and print out prediction
            data_x = np.genfromtxt(sys.path[0] + '\\aClean.csv', delimiter=',')
            
            if data_x.ndim == 1:
                data_x = np.resize(data_x, (1,36))

            data_x[data_x < 0] = 0
            
            if (len(data_x)) > 0:
                for piece in range(len(data_x)):
                    
                    input_x =  [data_x[piece]]
                    predict = sess.run([prediction], feed_dict={
                            x: input_x, 
                            keep_probL1: 1.0, 
                            keep_probL2: 1.0
                        })

                    print("Network Output: " , predict[0])

            continueProcessing = input("Continue processing? ('Y' or 'N'), use LIVECAM? ('live'): ")


        #######################
        # LIVE CAMERA TESTING #
        #######################

        while (continueProcessing == 'live' or continueProcessing == "y" or continueProcessing == "Y"):
            
            # Call image capture method

            # real_time.realTimeCapture()
            liveCam.getImage()

            # Run Datascriptgenerator
            dG.liveData()

            try:
                # Run network test on new aClean.csv file
                data_x = np.genfromtxt(sys.path[0] + '\\a.csv', delimiter=',')
                
                if data_x.ndim == 1:
                    data_x = np.resize(data_x, (1,36))

                data_x[data_x < 0] = 0

                if (len(data_x)) > 0:
                    for piece in range(len(data_x)):
                        
                        input_x =  [data_x[piece]]
                        predict = sess.run([prediction], feed_dict={
                                x: input_x, 
                                keep_probL1: 1.0, 
                                keep_probL2: 1.0
                            })

                        # print("Network Output: " , predict[0])

                        # Print "SQUATTING" or "STANDING" output
                        if (np.argmax(predict[0]) == 0):
                            print("SQUATTING")
                        elif (np.argmax(predict[0]) == 1):
                            print("STANDING")
                
            except:
                print("no human found")

            # continueProcessing = input("Continue Processing? ('Y' or 'N')")


train_neural_network(x)
