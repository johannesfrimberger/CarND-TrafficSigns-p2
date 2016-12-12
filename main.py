# Load pickled data
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
from tqdm import tqdm

import hashlib
import os
import pickle
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
from tqdm import tqdm
from zipfile import ZipFile

# Load the modules
import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

def load_data():
    training_file = 'traffic-signs-data/train.p'
    testing_file = 'traffic-signs-data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    X_train, y_train, c_train = train['features'], train['labels'], train['coords']
    X_test, y_test, c_test = test['features'], test['labels'], test['coords']

    return [X_train, y_train, c_train, X_test, y_test, c_test]

def data_summary(X_train, y_train, X_test, y_test):
    ### To start off let's do a basic data summary.

    shape_training_data = X_train.shape
    shape_test_data = X_test.shape

    # Number of training examples
    n_train = shape_training_data[0]

    # Number of testing examples
    n_test = shape_test_data[0]

    # What's the shape of an image?
    image_shape = shape_training_data[1:3]

    # How many classes are in the dataset
    n_classes = len(set(y_train))

    print("Number of training examples =", n_train)
    print("Number of testing examples =", n_test)
    print("Image data shape =", image_shape)
    print("Number of classes =", n_classes)

    return [n_train, n_test, image_shape, n_classes]

def load_sign_names():

    with open('signnames.csv', mode='r') as infile:
        reader = csv.reader(infile)
        signnames = [rows[1] for rows in reader]
        return signnames[1:]

def visualize_data(X_train, y_train, n_classes):

    fig = plt.figure()
    a = fig.add_subplot(1, 2, 1)
    plt.imshow(X_train[0, :, :, :])
    a = fig.add_subplot(1, 2, 2)
    class_dist = np.histogram(y_train, bins=range(n_classes))
    plt.hist(y_train, bins=range(n_classes))

    fig.savefig('overview.png')

def main():

    ######################
    ####### Step 1 #######
    ######################

    # Load Data
    [X_train, y_train, c_train, X_test, y_test, c_test] = load_data()

    #print(type(X_train))

    # Summarize Data
    [n_train, n_test, image_shape, n_classes] = data_summary(X_train, y_train, X_test, y_test)

    # Read sign names
    sign_names = load_sign_names()

    # Visualize data
    #visualize_data(X_train, y_train, n_classes)

    ######################
    ####### Step 2 #######
    ######################

    features_count = image_shape[0] * image_shape[1]
    labels_count = n_classes

    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)
    weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
    biases = tf.Variable(tf.zeros(labels_count))

    train_features = np.zeros([n_train, features_count])
    encoder = LabelBinarizer()
    encoder.fit(y_train)
    train_labels = encoder.transform(y_train)

    for i in range(0, features_count):
        currentImage = X_train[i, :, :]
        grayScaleImage = np.dot(currentImage[..., :3], [1., 1., 1.]) / 3
        train_features[i, :] = grayScaleImage.reshape(-1)

    train_feed_dict = {features: train_features, labels: train_labels}
    #test_feed_dict = {features: test_features, labels: test_labels}

    # Linear Function WX + b
    logits = tf.matmul(features, weights) + biases

    prediction = tf.nn.softmax(logits)

    # Cross entropy
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

    # Training loss
    loss = tf.reduce_mean(cross_entropy)

    # Create an operation that initializes all variables
    #init = tf.global_variables_initializer
    init = tf.initialize_all_variables()

    # Test Cases
    with tf.Session() as session:
        session.run(init)
        session.run(loss, feed_dict=train_feed_dict)
        biases_data = session.run(biases)

    assert not np.count_nonzero(biases_data), 'biases must be zeros'

    # Determine if the predictions are correct
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    # Calculate the accuracy of the predictions
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

    # TODO: Find the best parameters for each configuration
    epochs = 5
    batch_size = 50
    learning_rate = 0.1

    ### DON'T MODIFY ANYTHING BELOW ###
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # The accuracy measured against the validation set
    validation_accuracy = 0.0

    # Measurements use for graphing loss and accuracy
    log_batch_step = 50
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []

    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(train_features) / batch_size))

        for epoch_i in range(epochs):

            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i * batch_size
                batch_features = train_features[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer and get loss
                _, l = session.run(
                    [optimizer, loss],
                    feed_dict={features: batch_features, labels: batch_labels})

                # Log every 50 batches
                if not batch_i % log_batch_step:
                    # Calculate Training and Validation accuracy
                    training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                    #validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                    # Log batches
                    previous_batch = batches[-1] if batches else 0
                    batches.append(log_batch_step + previous_batch)
                    loss_batch.append(l)
                    train_acc_batch.append(training_accuracy)
                    #valid_acc_batch.append(validation_accuracy)

            # Check accuracy against Validation data
            #validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

if __name__ == "__main__":
    main()