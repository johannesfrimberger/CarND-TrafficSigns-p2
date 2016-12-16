# Load modules
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import math

import os

class TrafficSignClassifier:
    """

    """

    def __init__(self, folder):
        """
        Initialize TrafficSignClassifier class
        :param folder: Storage folder of training and test data
        """

        # Load training and test data from pickle file
        train, test = self.load_data(folder)

        self.training_features = train[0]
        self.training_labels = train[1]
        self.test_features = test[0]
        self.test_labels = test[1]
        self.valid_features = np.zeros_like(self.training_features)
        self.valid_labels = np.zeros_like(self.training_labels)

        self.n_train = 0
        self.n_test = 0
        self.image_shape = (0, 0)
        self.n_classes = 0

        self.logits = 0
        self.prediction = 0
        self.loss = 0
        self.accuracy = 0

        self.dense_layer_1 = 512
        self.dense_layer_2 = 256

    def train(self):
        """
        Run all required methods to train traffic sign classifier
        """
        self.basic_summary()
        #self.visualize_training_set()
        self.generate_additional_training_features()
        self.pre_process_features()
        self.generate_ohe_encoding()
        self.split_training_set()

        # Store placeholders for features and labels
        features = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels = tf.placeholder(tf.float32, [None, self.n_classes])

        # Reshape features for 1d input
        features = tf.reshape(features, [-1, self.image_shape[0] * self.image_shape[1]])
        self.logits = self.create_deep_layer(features)

        self.define_metrics(labels)

        # Create an operation that initializes all variables
        init = tf.global_variables_initializer()
        # Test Cases
        with tf.Session() as session:
            session.run(init)
            session.run(self.loss, feed_dict={features: self.training_features[0:5], labels: self.training_labels[0:5]})

        # Parameters
        training_epochs = 100
        batch_size = 20
        learning_rate = 0.001

        opt = tf.train.AdamOptimizer()
        optimizer = opt.minimize(self.loss)

        start_time = time.time()

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        sess = tf.Session()
        sess.run(init)

        batch_count = int(math.ceil(len(self.training_features) / batch_size))

        # Training cycle
        for epoch_i in range(training_epochs):

            totalLoss = 0
            totalAccuracy = 0

            # The training cycle
            for batch_i in range(batch_count):

                # Get a batch of training features and labels
                batch_start = batch_i * batch_size
                batch_features = self.training_features[batch_start:batch_start + batch_size]
                batch_labels = self.training_labels[batch_start:batch_start + batch_size]

                # Run optimizer and get loss
                _, l, a = sess.run([optimizer, self.loss, self.accuracy], feed_dict={features: batch_features, labels: batch_labels})

                totalLoss += l
                totalAccuracy += a

            totalLoss /= batch_count
            totalAccuracy /= batch_count
            print("Epoch {}/{} with Loss of {:.6f} and Accuracy of {:.6f}".format(epoch_i + 1, training_epochs, totalLoss, totalAccuracy))

            l, a = sess.run([self.loss, self.accuracy],
                               feed_dict={features: self.valid_features, labels: self.valid_labels})
            print("Validation Loss of {:.6f} and Accuracy of {:.6f}".format(l, a))
        sess.close()

        end_time = time.time()
        time_dif = end_time - start_time

        l, a = sess.run([self.loss, self.accuracy],
                        feed_dict={features: self.test_features, labels: self.test_labels})
        print("Test Loss of {:.6f} and Accuracy of {:.6f}".format(l, a))

        print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def visualize_training_set(self):
        """
        Visualize the training set
        """
        self.visualize_dataset(self.training_features, self.training_labels)

    def load_data(self, folder):
        """
        Load training and test data from given folder and unpack them
        :param folder: Storage folder of training and test data
        :return: training and test data
        """
        training_file = os.path.join(folder, 'train.p')
        testing_file = os.path.join(folder, 'test.p')

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        # Unpack training and test data
        X_train, y_train = train['features'], train['labels']
        X_test, y_test = test['features'], test['labels']

        return (X_train, y_train), (X_test, y_test)

    def basic_summary(self):
        """
        Give a basic summary on training and test data
        """
        # Number of training examples
        self.n_train = self.training_features.shape[0]

        # Number of testing examples
        self.n_test = self.test_features.shape[0]

        # What's the shape of an image?
        self.image_shape = self.training_features.shape[1:3]

        # How many classes are in the dataset
        self.n_classes = len(set(self.training_labels))

        print("Number of training examples =", self.n_train)
        print("Number of testing examples =", self.n_test)
        print("Image data shape =", self.image_shape)
        print("Number of classes =", self.n_classes)

    def visualize_dataset(self, features, labels):
        """
        Visualize the given features and labels.
        It shows a histogram of the distribution of classes and examples of the classes.
        :param features: List of evaluated features
        :param labels: List of evaluated labels
        """

        # Create a histogram of training lables
        fig = plt.figure()
        n, bins, patches = plt.hist(labels, self.n_classes)
        plt.xlabel('Traffic Sign Classes')
        plt.ylabel('occurrences')
        plt.show()

        fig = plt.figure()
        fig.suptitle('Overview Traffic Signs', fontsize=16)

        # Create an overview of trafic sign classes
        pltRows = 5
        pltCols = (self.n_classes / pltRows) + 1
        for el in range(self.n_classes):
            for i in range(0, len(labels)):
                if (labels[i] == el):
                    plt.subplot(pltRows, pltCols, el + 1)
                    fig = plt.imshow(features[i, :, :, :], interpolation='nearest')
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
                    break
        plt.show()

    def pre_process_image(self, image):
        """
        Convert image to YUV space and normalize results
        :return: Normalized image in yuv space
        """
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv = cv2.equalizeHist(yuv[:, :, 0])
        return (yuv.flatten() / 255. * 2.) - 1.

    def pre_process_features(self):
        """
        Preprocess features to improve performance of classifier
        """
        self.training_features = np.array([self.pre_process_image(self.training_features[i]) for i in range(len(self.training_features))],
                                  dtype=np.float32)
        self.test_features = np.array([self.pre_process_image(self.test_features[i]) for i in range(len(self.test_features))],
                                 dtype=np.float32)

    def generate_additional_training_features(self):
        """
        Generate addtional training features to have a more uniform distribution of labels
        """
        test = 0

    def generate_ohe_encoding(self):
        """
        Transform labels to one-hot encoding
        """
        encoder = LabelBinarizer()
        encoder.fit(self.training_labels)
        self.training_labels = encoder.transform(self.training_labels)
        self.test_labels = encoder.transform(self.test_labels)

    def split_training_set(self):
        """
        Split training set into traning and validation set
        """
        self.training_features, self.valid_features, self.training_labels, self.valid_labels = train_test_split(
            self.training_features,
            self.training_labels,
            test_size=0.15,
            random_state=832289)

    def create_deep_layer(self, dense_input):

        input_size = dense_input.get_shape().as_list()
        # Store layers weight & bias
        weights = {
            'hidden_layer': tf.Variable(tf.random_normal([input_size[1], self.dense_layer_1])),
            'hidden_layer2': tf.Variable(tf.random_normal([self.dense_layer_1, self.dense_layer_2])),
            'out': tf.Variable(tf.random_normal([self.dense_layer_2, self.n_classes]))
        }
        biases = {
            'hidden_layer': tf.Variable(tf.random_normal([self.dense_layer_1])),
            'hidden_layer2': tf.Variable(tf.random_normal([self.dense_layer_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(dense_input, weights['hidden_layer']), biases['hidden_layer'])
        layer_1 = tf.nn.relu(layer_1)
        layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['hidden_layer2'])
        layer_2 = tf.nn.relu(layer_2)

        # Output layer with linear activation
        logits = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

        return logits

    def define_metrics(self, labels):
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, labels))
        # Determine if the predictions are correct
        is_correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(labels, 1))
        # Calculate the accuracy of the predictions
        self.accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))