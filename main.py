# Load the modules
import pickle
import math
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

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

def normalizeData(input, coords):
    """
    Convert rgb to yuv color space and normalize y channel
    :param input: np Array of images in rgb color space
    :param coords: np Array of bounding boxes for the traffic sign
    :return: np Array of images in yuv color space
    """
    output = np.zeros_like(input)
    newFeatures = np.zeros_like(input)
    for i, item in enumerate(input):

        xTop = coords[i, 0]
        yTop = coords[i, 1]
        xBot = coords[i, 2]
        yBot = coords[i, 3]
        imSize = output[i, :].shape

        pts1 = np.float32([[xTop, yTop], [xBot, yTop], [xBot, yBot], [xTop, yTop]])
        pts2 = np.float32([[0, 0], [imSize[0], 0], [0, imSize[1]], [imSize[0], imSize[1]]])

        M = cv2.getPerspectiveTransform(pts1, pts2)

        newFeatures[i, :] = cv2.warpPerspective(item, M, (imSize[0], imSize[1]))

        # Convert to yuv color space
        output[i, :] = cv2.cvtColor(item, cv2.COLOR_BGR2YUV)
        # Normalize y channel
        output[i, :, :, 0] = cv2.equalizeHist(output[i, :, :, 0])

    # Convert image from uint8 to float
    output = output.astype(float) / 255.0

    return output

def generate_validation_set(training_set, test_set):
    return 0

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

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    for item in X_train:
        image = cv2.cvtColor(item, cv2.COLOR_BGR2RGB)
        pt1 = (c_train[0, 0], c_train[0, 1])
        pt3 = (c_train[0, 2], c_train[0, 3])
        pt4 = (c_train[0, 0], c_train[0, 3])
        pt2 = (c_train[0, 2], c_train[0, 1])

        pts1 = np.float32([pt1, pt2, pt3, pt4])
        pts2 = np.float32([[0, 0], [28, 0], [28, 28], [0, 28]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (28, 28))

        cv2.imshow("image", cv2.resize(image, (500, 500)))
        break

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Normalize and pre process data
    #X_train = normalizeData(X_train, c_train)
    #X_test = normalizeData(X_test, c_test)

    #visualize_data(X_train, y_train, n_classes)

if __name__ == "__main__":
    main()