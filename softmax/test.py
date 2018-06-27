import cv2
import tensorflow as tf
import os
import sys
import numpy as np

learning_rate = 0.001

num_input = 784  # MNIST data input (img shape: 28*28)
num_classes = 10  # MNIST total classes (0-9 digits)


def config_session():
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, [None, num_input])
    y = tf.placeholder(tf.float32, [None, num_classes])

    W = tf.Variable(tf.zeros([num_input, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    activation = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    session = tf.Session()

    return session, x, y, optimizer, cost, activation


def matrix_argmax(data):
    ret_ind = []
    for item in data:
        ret_ind.append(np.argmax(item))
    return ret_ind


def img2array(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, gray) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    resize = cv2.resize(255 - gray, (28, 28))
    """
        all images in the training set have an range from 0-1 and not from 0-255 so we divide our flatten images
        (a one dimensional vector with our 784 pixels) to use the same 0-1 based range
    """
    flatten = resize.flatten() / 255.0
    return [flatten]


def predict_label(image_path):
    # from tensorflow.examples.tutorials.mnist import input_data
    # mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

    """ load the trained model """
    sess, x, y, optimizer, cost, activation = config_session()

    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    model_path = "model/model_bin.ckpt"
    saver.restore(sess, model_path)

    """ predict the label """
    img = cv2.imread(image_path)
    _x = img2array(image_path)
    _y = sess.run(activation, feed_dict={x: _x})
    print(image_path, np.argmax(_y))
    cv2.imshow("data", img)
    cv2.waitKey(0)


if __name__ == '__main__':
    dir = "../data/"
    fns = [fn for fn in os.listdir(dir) if os.path.splitext(fn)[1].lower() in [".png", ".jpg"]]
    for fn in fns:
        predict_label(os.path.join(dir, fn))
