import tensorflow as tf
import numpy as np
import sys
np.set_printoptions(threshold=np.nan)


# Datasets
squattingData = np.genfromtxt(sys.path[0] + '/tf-pose-estimation/src/trainingSquattingClean.csv', delimiter=',')
standingData = np.genfromtxt(sys.path[0] + '/tf-pose-estimation/src/trainingStandingClean.csv', delimiter=',')

x_data = np.concatenate((squattingData , standingData), axis=0)

squattingData_y = np.ones((len(squattingData), 1))
standingData_y = np.zeros((len(standingData), 1))

y_data = np.concatenate( (squattingData_y, standingData_y), axis=0)


# Parameters
learningRate = 0.1
numSteps = 5000
batchSize = 128
displayStep = 100

# Network Parameters
nHidden1 = 50
nHidden2 = 50
inputNodes = 36     # input nodes
outputNode = 1      # output nodes

# tf Graph input
X = tf.placeholder("float", [None,  inputNodes])
Y = tf.placeholder("float", [1, outputNode])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_uniform([inputNodes, nHidden1], -1.0, 1.0)),
    'h2': tf.Variable(tf.random_uniform([nHidden1, nHidden2], -1.0, 1.0)),
    'out': tf.Variable(tf.random_uniform([nHidden2, outputNode], -1.0, 1.0)),
}

bias = {
    'b1': tf.Variable(tf.random_uniform([nHidden1], -1.0, 1.0)),
    'b2': tf.Variable(tf.random_uniform([nHidden2], -1.0, 1.0)),
    'out': tf.Variable(tf.random_uniform([outputNode], -1.0, 1.0))
}


# Create a model
def neuralNet(data):
    # Hidden fully connected with 50 neurons
    layer1 = tf.add(tf.matmul(data, weights['h1']), bias['b1'])

    # Hidden fully connect layer with 50 neurons
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), bias['b2'])

    # Output fully connected layer with a neuron for each classes
    outputLayer = tf.matmul(layer2, weights['out']) + bias['out']
    
    return outputLayer

# Construct model
logits = neuralNet(X)
prediction = tf.nn.sigmoid(logits)

# Define loss and optimizer
# lossOp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
lossOp = tf.reduce_sum(tf.square(Y - prediction))
optimizer = tf.train.AdamOptimizer(learningRate).minimize(lossOp)
# trainOp = optimizer.minimize(lossOp)

# Evaluate model
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accurary = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# Initialize the variables (i.e assign their default value)
init = tf.global_variables_initializer()


# Start session
with tf.Session() as sess:

    # Run the initizalizer
    sess.run(init)
   
    for step in range(1, numSteps+1):
        
        sess.run(optimizer, feed_dict = {X: batch[0], Y: batch[1]})

        loss, acc = sess.run([lossOp, accurary], feed_dict={X: batch[0], Y: batch[1]})

        # print("Step: " + str(step) + ", Data Loss " + "{:.4f}".format(loss) + ", Training accurary= " + "{:.3f}".format(acc) )

    print("Optimization finished!")

    # calculate accurary for the Dataset
    # print("Testing Accurary: ", sess.run(accurary, feed_dict={X: testSet, Y:testSetY}))