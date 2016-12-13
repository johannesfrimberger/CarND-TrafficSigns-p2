# Load the modules
import pickle
import math
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from tqdm import tqdm
import matplotlib.pyplot as plt

import random

def visualize_data(X_train, y_train, n_classes):
    ### Data exploration visualization goes here.
    ### Feel free to use as many code cells as needed.

    #fig = plt.figure()
    #n, bins, patches = plt.hist(y_train, n_classes)
    #plt.xlabel('Traffic Sign Classes')
    #plt.ylabel('occurrences')
    #plt.show()

    fig = plt.figure()
    fig.suptitle('Overview Traffic Signs', fontsize=16)

    pltRows = 5
    pltCols = (n_classes / pltRows) + 1

    for el in range(n_classes):
        for i in range(0, len(y_train)):
            if (y_train[i] == el):
                plt.subplot(pltRows, pltCols, el + 1)
                fig = plt.imshow(X_train[i, :, :, :], interpolation='nearest')
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                break
    plt.show()

def centerImage(image, coords):
    """

    :param image: image to center
    :param coords: coords of bounding box
    :return: centered image
    """
    pt1 = (coords[0], coords[1])
    pt2 = (coords[2], coords[1])
    pt3 = (coords[2], coords[3])
    pt4 = (coords[0], coords[3])
    pts1 = np.float32([pt1, pt2, pt3, pt4])
    pts2 = np.float32([[0, 0], [32, 0], [32, 32], [0, 32]])

    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(image, M, (32, 32))

def normalizeData(input, coords, rng = random):
    """
    Convert rgb to yuv color space and normalize y channel
    :param input: np Array of images in rgb color space
    :param coords: np Array of bounding boxes for the traffic sign
    :param rng: random number generator used. By default a new rng is initialized
    :return: np Array of images in yuv color space
    """
    n_newFeatures = 1
    newFeatures = np.zeros((input.shape[0] * n_newFeatures, input.shape[1], input.shape[2], input.shape[3]))

    for i in tqdm(range(input.shape[0]), unit="Frames"):

        indStart = i
        item = input[i, :]

        # Convert to yuv color space
        newFeatures[indStart, :] = cv2.cvtColor(item, cv2.COLOR_BGR2YUV)
        # Normalize y channel
        newFeatures[indStart, :, :, 0] = cv2.equalizeHist(newFeatures[indStart, :, :, 0].astype(input.dtype))

        # Add centered image
        #newFeatures[indStart, :] = centerImage(newFeatures[indStart, :], coords[i, :])

        # Convert image from uint8 to float representation
        newFeatures[indStart:indStart+n_newFeatures, :] = newFeatures[indStart:indStart+n_newFeatures, :].astype(float) / 255.0

    return newFeatures

def generate_validation_set(x, y, pVal = 0.7, rng=random):
    """

    :param x:
    :param y:
    :param pVal:
    :param rng:
    :return:
    """
    nTrainingSamples = x.shape[0]
    nTrainingSamplesNew = int(nTrainingSamples * pVal)

    selection = list(range(nTrainingSamples))
    rng.shuffle(selection)

    selectionTraining = selection[0:nTrainingSamplesNew]
    selectionValidation = selection[nTrainingSamplesNew:]

    x_Train = x[selectionTraining, :]
    y_Train = y[selectionTraining, :]

    x_Val = x[selectionValidation, :]
    y_Val = y[selectionValidation, :]

    return [x_Train, y_Train, x_Val, y_Val]

def splitIntoBatches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    assert len(features) == len(labels)
    outout_batches = []

    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)

    return outout_batches

def generateModel(image_shape, n_classes, X_train, y_train, X_valid, y_valid, X_test, y_test):

    # Pad 0s to 32x32. Centers the digit further.
    # Add 2 rows/columns on each side for height and width dimensions.
    x = tf.placeholder("float", [None, image_shape[0], image_shape[1], 3])
    y = tf.placeholder("float", [None, n_classes])

    #x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6)))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    conv1 = tf.nn.relu(conv1)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # 10x10x16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16)))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    conv2 = tf.nn.relu(conv2)

    # 5x5x16
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten
    fc1 = flatten(conv2)
    # (5 * 5 * 16, 120)
    fc1_shape = (fc1.get_shape().as_list()[-1], 120)

    fc1_W = tf.Variable(tf.truncated_normal(shape=(fc1_shape)))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, n_classes)))
    fc2_b = tf.Variable(tf.zeros(n_classes))

    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(fc2, y))
    #opt = tf.train.AdamOptimizer()
    #train_op = opt.minimize(loss_op)
    #correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
    #accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate) \
        .minimize(loss_op)

    EPOCHS = 10
    BATCH_SIZE = 50

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        batch = splitIntoBatches(BATCH_SIZE, X_train, y_train)
        batchVal = splitIntoBatches(100, X_valid, y_valid)
        batchTest = splitIntoBatches(100, X_test, y_test)

        # Training cycle
        for epoch in tqdm(range(EPOCHS)):
            # Loop over all batches
            for data, label in tqdm(batch):
                loss = sess.run(optimizer, feed_dict={x: data, y: label})

            # Display logs per epoch step
            c = sess.run(loss_op, feed_dict={x: X_train, y: y_train})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(fc2, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(
            "Accuracy:",
            accuracy.eval({x: X_test, y: y_test}))

def one_hot(input, n_classes):
    """

    :param input:
    :param n_classes:
    :return:
    """

    output = np.zeros((input.shape[0], n_classes), dtype=int)
    for i, el in enumerate(input):
        output[i, el] = 1

    return output

def main():
    # Fill this in based on where you saved the training and testing data
    training_file = 'traffic-signs-data/train.p'
    testing_file = 'traffic-signs-data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train, c_train = train['features'], train['labels'], train['coords']
    X_test, y_test, c_test = test['features'], test['labels'], train['coords']

    ### To start off let's do a basic data summary.

    # Number of training examples
    n_train = X_train.shape[0]

    # Number of testing examples
    n_test = X_test.shape[0]

    # What's the shape of an image?
    image_shape = X_train.shape[1:3]

    # How many classes are in the dataset
    n_classes = len(set(y_train))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    # visualize_data(X_train, y_train, n_classes)

    # Normalize and pre process data
    rng = random
    rng.seed(100)
    X_train = normalizeData(X_train, c_train, rng)
    X_test = normalizeData(X_test, c_test, rng)

    y_train = one_hot(y_train, n_classes)
    y_test = one_hot(y_test, n_classes)

    X_train, y_train, X_valid, y_valid = generate_validation_set(X_train, y_train, rng=rng)

    fc2 = generateModel(image_shape, n_classes, X_train, y_train, X_valid, y_valid, X_test, y_test)

if __name__ == "__main__":
    main()