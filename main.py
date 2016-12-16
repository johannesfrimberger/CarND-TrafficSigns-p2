# Load modules
import pickle
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tqdm import tqdm
import time
from datetime import timedelta
import math
import matplotlib.pyplot as plt

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
    #yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    yuvNormalized = cv2.normalize(yuv.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

    yuv[:, :, 0] = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

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

def get_batch(features, labels, batch_size, pos):
    startPos = pos * batch_size
    endPos = (pos+1) * batch_size
    return features[startPos:endPos, :], labels[startPos:endPos, :]

def generate_network():
    return 0

def pre_process_image(image):

    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
    return (yuv[:, :, 0].flatten()/255. * 2.) - 1.

def main():
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

    train_features = np.array([pre_process_image(X_train[i]) for i in range(len(X_train))],
                              dtype=np.float32)
    test_features = np.array([pre_process_image(X_test[i]) for i in range(len(X_test))],
                             dtype=np.float32)

    ### OHE encoder

    # Turn labels into numbers and apply One-Hot Encoding
    encoder = LabelBinarizer()
    encoder.fit(y_train)
    train_labels = encoder.transform(y_train)
    test_labels = encoder.transform(y_test)

    # Change to float32, so that it can be multiplied against the features in TensorFlow which are float32
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    is_labels_encod = True

    print('Labels One-Hot Encoded')

    ### Randomize data

    # Get randomized datasets for training and validation

    train_features, valid_features, train_labels, valid_labels = train_test_split(
        train_features,
        train_labels,
        test_size=0.15,
        random_state=832289)

    print('Training features and labels randomized and split')
    print(train_features.shape)
    print(train_labels.shape)

    ### Setting up model

    ### Preprocess the data here.
    ### Feel free to use as many code cells as needed.
    import tensorflow as tf

    features_count = train_features.shape[1]
    # features_count = 3072

    # labels_count = 43
    labels_count = train_labels.shape[1]

    features = tf.placeholder(tf.float32, [None, train_features.shape[1]])
    labels = tf.placeholder(tf.float32, [None, train_labels.shape[1]])

    n_hidden_layer = 512  # layer number of features

    # Store layers weight & bias
    weights = {
        'hidden_layer': tf.Variable(tf.random_normal([train_features.shape[1], n_hidden_layer])),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_layer, n_hidden_layer])),
        'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
    }
    biases = {
        'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
        'hidden_layer2': tf.Variable(tf.random_normal([n_hidden_layer])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(features, weights['hidden_layer']), biases['hidden_layer'])
    layer_1 = tf.nn.relu(layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer2']), biases['hidden_layer2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    logits = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

    weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
    biases = tf.Variable(tf.zeros(labels_count))

    train_dict = {features: train_features, labels: train_labels}
    valid_dict = {features: valid_features, labels: valid_labels}
    test_dict = {features: test_features, labels: test_labels}

    # Linear Function WX + b
    #logits = tf.matmul(features, weights) + biases

    prediction = tf.nn.softmax(logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    print("loss ", loss)

    # Create an operation that initializes all variables
    init = tf.initialize_all_variables()

    # Test Cases
    with tf.Session() as session:
        session.run(init)
        session.run(loss, feed_dict=train_dict)
        session.run(loss, feed_dict=test_dict)
        biases_data = session.run(biases)

    # Determine if the predictions are correct
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    print('is_correct_prediction', is_correct_prediction)
    # Calculate the accuracy of the predictions
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
    print('accuracy', accuracy)
    print('Accuracy function created')

    import tensorflow as tf
    from tqdm import tqdm
    import math
    import matplotlib.pyplot as plt

    # Parameters
    training_epochs = 100
    batch_size = 20
    learning_rate = 0.001

    ### DON'T MODIFY ANYTHING BELOW ###
    # Gradient Descent
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    opt = tf.train.AdamOptimizer()
    optimizer = opt.minimize(loss)

    # The accuracy measured against the validation set
    validation_accuracy = 0.0

    # Measurements use for graphing loss and accuracy
    log_batch_step = 2000
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []

    start_time = time.time()

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph

    sess = tf.Session()
    sess.run(init)
    batch_count = int(math.ceil(len(train_features) / batch_size))
    # Training cycle
    for epoch_i in tqdm(range(training_epochs)):

        # Progress bar
        # batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, training_epochs),unit='batches')

        # The training cycle
        for batch_i in tqdm(range(1000)):
            # Get a batch of training features and labels
            batch_start = batch_i * batch_size
            batch_features = train_features[batch_start:batch_start + batch_size]
            batch_labels = train_labels[batch_start:batch_start + batch_size]

            # Run optimizer and get loss
            _, l = sess.run([optimizer, loss], feed_dict={features: batch_features, labels: batch_labels})

            # Log every 50 batches
            if not batch_i % log_batch_step:
                # Calculate Training and Validation accuracy
                training_accuracy = sess.run(accuracy, feed_dict=train_dict)
                validation_accuracy = sess.run(accuracy, feed_dict=valid_dict)

                # Log batches
                previous_batch = batches[-1] if batches else 0
                batches.append(log_batch_step + previous_batch)
                loss_batch.append(l)
                train_acc_batch.append(training_accuracy)
                print('training accuracy at {}'.format(training_accuracy))
                valid_acc_batch.append(validation_accuracy)
                print('Validation accuracy at {}'.format(validation_accuracy))

        # Check accuracy against testing data #change later to check against validation data
        validation_accuracy = sess.run(accuracy, feed_dict=valid_dict)

    sess.close()

    loss_plot = plt.subplot(211)
    loss_plot.set_title('Loss')
    loss_plot.plot(batches, loss_batch, 'g')
    loss_plot.set_xlim([batches[0], batches[-1]])
    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')

    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
    acc_plot.set_ylim([0, 1.0])
    acc_plot.set_xlim([batches[0], batches[-1]])
    acc_plot.legend(loc=4)
    plt.tight_layout()
    plt.show()

    print('Validation accuracy at {}'.format(validation_accuracy))

    end_time = time.time()
    time_dif = end_time - start_time

    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

if __name__ == "__main__":
    main()