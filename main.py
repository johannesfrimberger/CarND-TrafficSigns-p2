# Load modules
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tqdm import tqdm

def one_hot(input, n_classes):
    """
    Convert input arrey of labels into 2d array of one-hot encoded labels
    :param input: np array of labels
    :param n_classes: number of classes
    :return:
    """

    output = np.zeros((input.shape[0], n_classes), dtype=int)
    for i, el in enumerate(input):
        output[i, el] = 1

    return output

def convert_to_yuv(input):
    """

    :param input:
    :return:
    """
    yuv = cv2.cvtColor(input, cv2.COLOR_RGB2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    yuvNormalized = cv2.normalize(yuv.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) - 0.5

    return yuvNormalized

def preprocess_inputs(input):
    """

    :param input:
    :return:
    """
    output = np.zeros_like(input)

    for ind in range(input.shape[0]):
        output[ind, :] = convert_to_yuv(input[ind, :])

    return output

def generate_network(image_shape, n_classes):

    n_input = image_shape[0] * image_shape[1]
    n_hidden_layer = 512
    n_hidden_layer2 = 256

    weights = {
        'hidden_layer1': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_layer, n_hidden_layer2])),
        'out': tf.Variable(tf.random_normal([n_hidden_layer2, n_classes]))
    }
    biases = {
        'hidden_layer1': tf.Variable(tf.random_normal([n_hidden_layer])),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_layer2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Dataset consists of 32x32x3 yuv images
    x = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], 1))
    # Classify over 43 classes
    y = tf.placeholder(tf.float32, (None, n_classes))

    # ToDO: Reshape to vector as long as CNN is not used
    x_flat = tf.reshape(x, [-1, image_shape[0] * image_shape[1]])

    fc_layer1 = tf.add(tf.matmul(x_flat, weights['hidden_layer1']), biases['hidden_layer1'])
    fc_layer1 = tf.nn.relu(fc_layer1)

    fc_layer2 = tf.add(tf.matmul(fc_layer1, weights['hidden_layer2']), biases['hidden_layer2'])
    fc_layer2 = tf.nn.relu(fc_layer2)

    # Output layer with linear activation
    logits = tf.add(tf.matmul(fc_layer2, weights['out']), biases['out'])

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y))

    prediction = tf.nn.softmax(logits)
    # Determine if the predictions are correct
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    # Calculate the accuracy of the predictions
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

    return cost, accuracy, x, y

def get_batch(features, labels, batch_size, pos):
    startPos = pos * batch_size
    endPos = (pos+1) * batch_size
    return features[startPos:endPos, :], labels[startPos:endPos, :]

if __name__ == "__main__":

    # Fill this in based on where you saved the training and testing data
    training_file = 'traffic-signs-data/train.p'
    testing_file = 'traffic-signs-data/test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    # Unpack training and test data
    X_train, y_train = train['features'], train['labels']
    X_test, y_test = test['features'], test['labels']

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

    # Generate one-hot encoding for lables
    Y_train = one_hot(y_train, n_classes)
    Y_test = one_hot(y_test, n_classes)

    # Normalize data
    X_train = preprocess_inputs(X_train)
    X_test = preprocess_inputs(X_test)

    # Drop color (u,v) channel ToDo: Reactivate color information and use CNN
    X_train = X_train[:, :, :, 0]
    X_test = X_test[:, :, :, 0]
    X_train = X_train[:, :, :, np.newaxis]
    X_test = X_test[:, :, :, np.newaxis]

    # Split training set into training and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    # Update n_train after split
    n_train = X_train.shape[0]

    # Optimizer
    learning_rate = 0.0001
    training_epochs = 20
    batch_size = 100
    display_step = 1

    cost, accuracy, x, y = generate_network(image_shape, n_classes)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Add single element ToDo: Better use rounding
    total_batch = int(n_train / batch_size) + 1

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_x, batch_y = get_batch(X_train, y_train, batch_size, i)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
                c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                #print("Cost=", "{:.9f}".format(c))

            c = sess.run(cost, feed_dict={x: X_train, y: y_train})
            print("Test Cost=", "{:.9f}".format(c))
            c = sess.run(cost, feed_dict={x: X_val, y: y_val})
            print("Validation Cost=", "{:.9f}".format(c))
            validation_accuracy = sess.run(accuracy, feed_dict={x: X_val, y: y_val})
            print('Validation accuracy at {}'.format(validation_accuracy))

        print("Optimization Finished!")