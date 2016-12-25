# Load modules
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
import time
from datetime import timedelta
import math
import csv
import scipy.interpolate as interpolate

import os


class TrafficSignClassifier:
    """
    Common class to access methods of the traffic sign classifier
    """

    def __init__(self, folder):
        """
        Initialize TrafficSignClassifier class
        :param folder: Storage folder of training and test data
        """

        # Parameters for the convolutional neural net with 2 dense layers
        self.cnn_depth = 64
        self.dense_layer_1 = 512
        self.dense_layer_2 = 512
        self.beta = 0.001

        # Filename to store current model
        self.save_file = "model"

        # Load training and test data from pickle file
        train, test = self.load_data(folder)

        # Init internal storage of training, test and validation set
        self.training_features = train[0]
        self.training_labels = train[1]
        self.test_features = test[0]
        self.test_labels = test[1]
        self.valid_features = np.zeros_like(self.training_features)
        self.valid_labels = np.zeros_like(self.training_labels)

        # Read class labels and store them
        self.sign_name = self.load_sign_names("signnames.csv")

        # What's the shape of an image?
        self.image_shape = self.training_features.shape[1:3]
        # How many classes are in the dataset
        self.n_classes = len(set(self.training_labels))

        # Init random generator
        np.random.seed(2000)

        # Store tensorflow placeholders for features and labels
        self.features = tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.labels = tf.placeholder(tf.float32, [None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32)

        # Initialize tensorflow places
        self.logits = 0
        self.prediction = 0
        self.loss = 0
        self.accuracy = 0

        self.weights = {
            'hidden_layer': tf.Variable(tf.random_normal([1, self.dense_layer_1])),
            'hidden_layer2': tf.Variable(tf.random_normal([self.dense_layer_1, self.dense_layer_2])),
            'out': tf.Variable(tf.random_normal([self.dense_layer_2, self.n_classes]))
        }
        self.biases = {
            'hidden_layer': tf.Variable(tf.random_normal([self.dense_layer_1])),
            'hidden_layer2': tf.Variable(tf.random_normal([self.dense_layer_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Internal storage for training progress visualization
        self.hist_train_loss = []
        self.hist_train_acc = []
        self.hist_valid_loss = []
        self.hist_valid_acc = []


    def train(self, dropout=False, l2_reg=False, training_epochs=100, batch_size=50, run_optimization=True):
        """
        Run all required methods to train traffic sign classifier optimization
        :param dropout:
        :param l2_reg:
        :param training_epochs:
        :param batch_size:
        :param run_optimization:
        :return:
        """

        # Reshape features for 1d input
        cnn = self.create_cnn()
        self.logits = self.create_deep_layer(flatten(cnn), dropout=dropout)
        self.define_metrics(l2_reg)

        # Create an operation that initializes all variables
        init = tf.global_variables_initializer()

        # Test model with a small set of training data
        with tf.Session() as session:
            session.run(init)
            session.run(self.loss, feed_dict={self.features: self.training_features[0:2],
                                              self.labels: self.training_labels[0:2],
                                              self.keep_prob: 1.0})
            session.run(self.loss, feed_dict={self.features: self.valid_features[0:2],
                                              self.labels: self.valid_labels[0:2],
                                              self.keep_prob: 1.0})
            session.run(self.loss, feed_dict={self.features: self.test_features[0:2],
                                              self.labels: self.test_labels[0:2],
                                              self.keep_prob: 1.0})

        # Run optimization algorithm only if requested
        if run_optimization:

            # Reset internal storage for visualization
            self.hist_train_loss = []
            self.hist_train_acc = []
            self.hist_valid_loss = []
            self.hist_valid_acc = []

            # Use Adam optimizer and minimize the loss
            opt = tf.train.AdamOptimizer()
            optimizer = opt.minimize(self.loss)

            # Store
            start_time = time.time()

            # Initializing the variables
            init = tf.global_variables_initializer()

            # Class used to save and/or restore Tensor Variables
            saver = tf.train.Saver()

            # Launch the graph
            sess = tf.Session()
            sess.run(init)

            # Determine
            batch_count = int(math.ceil(self.training_features.shape[0] / batch_size))

            # Training cycle
            for epoch_i in range(training_epochs):

                total_loss = []
                total_accuracy = []

                # Run all batches
                for batch_i in range(batch_count):

                    # Get a batch of training features and labels
                    batch_start = batch_i * batch_size

                    batch_features = self.training_features[batch_start:(batch_start + batch_size)]
                    batch_labels = self.training_labels[batch_start:(batch_start + batch_size)]

                    # Run optimizer and determine loss + accuracy for this batch
                    sess.run(optimizer, feed_dict={self.features: batch_features, self.labels: batch_labels,
                                                   self.keep_prob: 0.5})

                    l, a = sess.run([self.loss, self.accuracy],
                                    feed_dict={self.features: batch_features, self.labels: batch_labels,
                                               self.keep_prob: 1.0})

                    # Add loss and accuracy of this batch to list
                    total_loss.append(l)
                    total_accuracy.append(a)

                # Calculate mean of loss and accuracy list + Print this information
                total_loss = np.mean(total_loss)
                total_accuracy = np.mean(total_accuracy)
                print("Epoch {}/{} with Loss of {:.6f} and Accuracy of {:.6f}"
                      .format(epoch_i + 1, training_epochs, total_loss, total_accuracy))

                # Store history
                self.hist_train_loss.append(total_loss)
                self.hist_train_acc.append(total_accuracy)

                total_loss = []
                total_accuracy = []

                valid_batch_count = int(math.ceil(self.valid_features.shape[0] / batch_size))

                for batch_i in range(valid_batch_count):
                    # Get a batch of training features and labels
                    batch_start = batch_i * batch_size

                    batch_features = self.valid_features[batch_start:(batch_start + batch_size)]
                    batch_labels = self.valid_labels[batch_start:(batch_start + batch_size)]
                    l, a = sess.run([self.loss, self.accuracy],
                                    feed_dict={self.features: batch_features, self.labels: batch_labels,
                                               self.keep_prob: 1.0})

                    total_loss.append(l)
                    total_accuracy.append(a)

                # Calculate mean of loss and accuracy list + Print this information
                total_loss = np.mean(total_loss)
                total_accuracy = np.mean(total_accuracy)
                print("Validation Loss of {:.6f} and Accuracy of {:.6f}".format(total_loss, total_accuracy))

                # Store history
                self.hist_valid_loss.append(total_loss)
                self.hist_valid_acc.append(total_accuracy)

            # Save model
            saver.save(sess, self.save_file)

            test_batch_count = int(math.ceil(self.test_features.shape[0] / batch_size))

            for batch_i in range(test_batch_count):
                # Get a batch of training features and labels
                batch_start = batch_i * batch_size

                batch_features = self.test_features[batch_start:(batch_start + batch_size)]
                batch_labels = self.test_labels[batch_start:(batch_start + batch_size)]
                l, a = sess.run([self.loss, self.accuracy],
                                feed_dict={self.features: batch_features, self.labels: batch_labels,
                                           self.keep_prob: 1.0})

                total_loss += l
                total_accuracy += a

            sess.close()

            total_loss /= test_batch_count
            total_accuracy /= test_batch_count

            print("Test Loss of {:.6f} and Accuracy of {:.6f}".format(total_loss, total_accuracy))

            sess.close()

            # Determine time used for training and print information
            end_time = time.time()
            time_dif = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

    def visualize_training_progress(self):
        """

        """

        loss_plot = plt.subplot(211)
        loss_plot.set_title('Loss')
        loss_plot.plot(range(0, len(self.hist_train_loss)), self.hist_train_loss, 'g')
        acc_plot = plt.subplot(212)
        acc_plot.set_title('Accuracy')
        acc_plot.plot(range(0, len(self.hist_train_loss)), self.hist_train_acc, 'r', label='Training Accuracy')
        acc_plot.plot(range(0, len(self.hist_train_loss)), self.hist_valid_acc, 'b', label='Validation Accuracy')
        acc_plot.set_ylim([0, 1.0])
        acc_plot.legend(loc=4)
        plt.tight_layout()
        plt.show()

    def visualize_training_set(self):
        """
        Visualize the training set
        """
        self.visualize_dataset(self.training_features, self.training_labels, self.n_classes)

    def evaluate_test_set(self, batch_size=50):
        """
        Run current classifier on test set
        """
        total_loss = 0
        total_accuracy = 0
        test_batch_count = int(math.ceil(self.test_features.shape[0] / batch_size))

        # Launch the graph
        sess = tf.Session()

        saver = tf.train.import_meta_graph(self.save_file + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        for batch_i in range(test_batch_count):
            # Get a batch of training features and labels
            batch_start = batch_i * batch_size

            batch_features = self.test_features[batch_start:(batch_start + batch_size)]
            batch_labels = self.test_labels[batch_start:(batch_start + batch_size)]

            l, a = sess.run([self.loss, self.accuracy],
                            feed_dict={self.features: batch_features, self.labels: batch_labels, self.keep_prob: 1.0})

            total_loss += l
            total_accuracy += a

        sess.close()

        total_loss /= test_batch_count
        total_accuracy /= test_batch_count

        print("Test Loss of {:.6f} and Accuracy of {:.6f}".format(total_loss, total_accuracy))

    def evaluate_image(self, image_list):
        """
        Determine traffic sign class for all images in given list
        :param image_list: np array containing all images that should be evaluated
        """


    @staticmethod
    def load_data(folder):
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

    @staticmethod
    def load_sign_names(filename):
        """
        Load sign names from csv file
        :param filename: csv file with class id in the first and sign name in the second column
        :return: list with sign names
        """

        with open(filename, 'r') as f:
            reader = csv.reader(f)
            sign_names = list(reader)

        return sign_names[1:]

    @staticmethod
    def visualize_dataset(features, labels, n_classes):
        """
        Visualize the given features and labels.
        It shows a histogram of the distribution of classes and examples of the classes.
        :param features: List of evaluated features
        :param labels: List of evaluated labels
        """

        # Create a histogram of training lables
        fig = plt.figure()
        n, bins, patches = plt.hist(labels, n_classes)

        plt.xlabel('Traffic Sign Classes')
        plt.ylabel('occurrences')
        plt.show()

        fig = plt.figure()
        fig.suptitle('Overview Traffic Signs', fontsize=16)

        # Create an overview of trafic sign classes
        pltRows = 5
        pltCols = (n_classes / pltRows) + 1
        for el in range(n_classes):
            for i in range(0, len(labels)):
                if (labels[i] == el):
                    plt.subplot(pltRows, pltCols, el + 1)
                    fig = plt.imshow(features[i, :, :, :], interpolation='nearest')
                    fig.axes.get_xaxis().set_visible(False)
                    fig.axes.get_yaxis().set_visible(False)
                    break
        plt.show()

    @staticmethod
    def pre_process_image(image):
        """
        Convert image to YUV space and normalize results
        :return: Normalized image in yuv space
        """
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        # yuv = yuv[:, :, np.newaxis]
        return (yuv / 255. * 2.) - 1.

    @staticmethod
    def shift_and_rotate_image(img, shift=(0, 0), rotation=0, scale=1.0):
        """
        Shift and rotate image by given coefficients
        :param img: Input image
        :param shift: Tuple for shift in x and y direction
        :param rotation: Rotation in deg
        :param scale: Scaling that should be applied to the image
        :return: Shifted and rotated input image
        """
        rows, cols, c = img.shape

        m = np.float32([[1, 0, shift[0]], [0, 1, shift[0]]])
        shifted = cv2.warpAffine(img, m, (cols, rows))
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, scale)
        transformed = cv2.warpAffine(shifted, m, (cols, rows))

        return transformed

    @staticmethod
    def inverse_transform_sampling(input_data, n_bins, n_samples):
        """
        Analyse distribution of input data and draw samples to increase uniformity of the distribution
        Code snippet taken from
        http://www.nehalemlabs.net/prototype/blog/2013/12/16/how-to-do-inverse-transformation-sampling-in-scipy-and-numpy/
        :param input_data:
        :param n_bins:
        :param n_samples:
        :return:
        """
        # Calc histogram of input
        hist, bin_edges = np.histogram(input_data, bins=n_bins, density=True)

        # Inverse histogram and normalize again
        new_hist = 1 / hist
        new_hist = new_hist / sum(new_hist)

        # Calc cumulative sum
        cum_values = np.zeros(bin_edges.shape)
        cum_values[1:] = np.cumsum(new_hist * np.diff(bin_edges))

        inv_cdf = interpolate.interp1d(cum_values, bin_edges)

        # Draw n_samples random numbers and ensure range is within cumulative sum elements
        r = np.random.rand(n_samples) * cum_values[-1]
        return inv_cdf(r)

    def basic_summary(self):
        """
        Give a basic summary on training and test data
        """
        print("Number of training examples =", self.training_features.shape[0])
        print("Number of testing examples =", self.test_features.shape[0])
        print("Image data shape =", self.image_shape)
        print("Number of classes =", self.n_classes)

    def pre_process_features(self):
        """
        Preprocess features to improve performance of classifier
        """
        self.training_features = np.array(
            [self.pre_process_image(self.training_features[i]) for i in range(len(self.training_features))],
            dtype=np.float32)
        self.valid_features = np.array(
            [self.pre_process_image(self.valid_features[i]) for i in range(len(self.valid_features))],
            dtype=np.float32)
        self.test_features = np.array(
            [self.pre_process_image(self.test_features[i]) for i in range(len(self.test_features))],
            dtype=np.float32)

    def generate_additional_training_features(self, n_additional_features=10000):
        """
        Generate additional training features to have a more uniform distribution of labels
        """

        new_feature_dist = self.inverse_transform_sampling(self.training_labels, self.n_classes,
                                                           n_additional_features)
        new_feature_dist = np.round(new_feature_dist)
        new_feature_dist = new_feature_dist.astype(int)

        unique, counts = np.unique(new_feature_dist, return_counts=True)

        new_labels = []
        new_features = np.zeros([n_additional_features, self.training_features.shape[1],
                                 self.training_features.shape[2], self.training_features.shape[3]], dtype=np.uint8)

        write_pos = 0
        for ind, number in zip(unique, counts):

            # Get index of existing images of this class
            item_index = np.where(self.training_labels == ind)[0]
            n_items = len(item_index)
            iterations = int(np.ceil(number / n_items))

            image_basis = np.copy(item_index)
            for it in range(iterations - 1):
                np.random.shuffle(item_index)
                image_basis = np.append(image_basis, item_index)

            image_basis = image_basis[0:number]

            for img_number in image_basis:
                img = self.training_features[img_number]

                #shift = np.random.randint(-2, 2, (2, 1))
                rot = np.random.randint(-2, 2) * 5
                #scale = float(np.random.randint(90, 110)) / 100.
                img = self.shift_and_rotate_image(img, rotation=rot)

                new_labels.append(ind)
                new_features[write_pos, :, :, :] = img
                write_pos += 1

        self.training_features = np.append(self.training_features, new_features, axis=0)
        self.training_labels = np.append(self.training_labels, new_labels)

    def generate_ohe_encoding(self):
        """
        Transform labels to one-hot encoding
        """
        encoder = LabelBinarizer()
        encoder.fit(self.training_labels)
        self.training_labels = encoder.transform(self.training_labels)
        self.valid_labels = encoder.transform(self.valid_labels)
        self.test_labels = encoder.transform(self.test_labels)

    def split_training_set(self, valid_size=0.15):
        """
        Split training set into traning and validation set
        :param test_size:
        """
        self.training_features, self.valid_features, self.training_labels, self.valid_labels = train_test_split(
            self.training_features,
            self.training_labels,
            test_size=valid_size,
            random_state=10)

    def create_deep_layer(self, dense_input, dropout=False):
        """

        :param dense_input:
        :param dropout:
        :return:
        """

        # Store layers weight & bias
        self.weights = {
            'hidden_layer': tf.Variable(tf.random_normal([dense_input.get_shape().as_list()[-1], self.dense_layer_1])),
            'hidden_layer2': tf.Variable(tf.random_normal([self.dense_layer_1, self.dense_layer_2])),
            'out': tf.Variable(tf.random_normal([self.dense_layer_2, self.n_classes]))
        }
        self.biases = {
            'hidden_layer': tf.Variable(tf.random_normal([self.dense_layer_1])),
            'hidden_layer2': tf.Variable(tf.random_normal([self.dense_layer_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Hidden layer with RELU activation
        layer_1 = tf.nn.xw_plus_b(dense_input, self.weights['hidden_layer'], self.biases['hidden_layer'])
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.nn.xw_plus_b(layer_1, self.weights['hidden_layer2'], self.biases['hidden_layer2'])
        layer_2 = tf.nn.relu(layer_2)
        if dropout:
            layer_2 = tf.nn.dropout(layer_2, self.keep_prob)

        # Output layer with linear activation
        logits = tf.add(tf.matmul(layer_2, self.weights['out']), self.biases['out'])

        return logits

    def create_cnn(self):
        """

        :return:
        """

        x = tf.reshape(self.features, (-1, 32, 32, 3))
        # Pad 0s to 36x36. Centers the digit further.
        # Add 2 rows/columns on each side for height and width dimensions.
        x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode="CONSTANT")

        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, self.cnn_depth)))
        conv1_b = tf.Variable(tf.zeros(self.cnn_depth))
        conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
        conv1 = tf.nn.relu(conv1)

        return conv1

    def define_metrics(self, l2_reg=False):
        """

        :param l2_reg:
        :return:
        """
        self.prediction = tf.nn.softmax(self.logits)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.labels))

        if l2_reg:
            self.loss = (self.loss +
                         self.beta * tf.nn.l2_loss(self.weights['hidden_layer']) +
                         self.beta * tf.nn.l2_loss(self.weights['hidden_layer2']) +
                         self.beta * tf.nn.l2_loss(self.weights['out']) +
                         self.beta * tf.nn.l2_loss(self.biases['hidden_layer']) +
                         self.beta * tf.nn.l2_loss(self.biases['hidden_layer2']) +
                         self.beta * tf.nn.l2_loss(self.weights['out']))

        # Determine if the predictions are correct
        is_correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        # Calculate the accuracy of the predictions
        self.accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
