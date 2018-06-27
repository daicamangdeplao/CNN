import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random
from sklearn.datasets import fetch_mldata

n_classes = 10
batch_size = 100

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    #                        size of window      movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def one_hot_encoding(my_list):
    encoding_list = np.array([])
    for e in my_list:
        tmp = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        tmp[int(e)] = 1.0
        print('======== DEBUG: {}'.format(tmp))
        encoding_list = np.append(encoding_list, tmp)

    return encoding_list


def convolutional_neural_network(x_plah, weights, biases):
    x_plah = tf.reshape(x_plah, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x_plah, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(mnist_data, x_plah, y_plah, predictions, optimizer, loss):
    hm_epochs = 10
    n_samples = mnist_data.data.shape[0]
    n_batches = int(n_samples / batch_size)
    features_train = mnist_data.data.astype(np.float32)
    labels_train = one_hot_encoding(mnist_data.target)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            total_loss = 0
            count = 0
            for i in range(n_batches):
                start = i * batch_size
                end = i * batch_size + batch_size - 1

                features_train = features_train[start: end]
                labels_train = labels_train[start: end]
                # print('======== DEBUG: {} '
                #       'dtype of features_train {}, '
                #       'type of features_train {}'.format(count, features_train.dtype, type(features_train)))
                # print('======== DEBUG: {} '
                #       'dtype of labels_train {}, '
                #       'type of labels_train {}'.format(count, labels_train.dtype, type(labels_train)))
                _, loss_value = sess.run([optimizer, loss], feed_dict={x_plah: features_train, y_plah: labels_train})
                total_loss += loss_value
                count += 1

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', total_loss)


def main():
    # TODO Load data
    print('Load data')
    mnist_data = fetch_mldata('MNIST original')

    # TODO Define placeholders
    print('Define placeholders')
    x_plah = tf.placeholder(tf.float32, shape=[None, 784], name='x_plah')
    y_plah = tf.placeholder(tf.float32, name='y_plah')

    # TODO Define variables
    print('Define variables')
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([n_classes]))}

    # TODO Define hypothesis function
    print('Define hypothesis function')
    predictions = convolutional_neural_network(x_plah, weights, biases)

    # TODO Define loss function
    print('Define loss function')
    # loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=y_plah, logits=predictions))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=y_plah))

    # TODO Define optimizer
    print('Define optimizer')
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # TODO Execute graph
    print('Execute graph')
    train_neural_network(mnist_data, x_plah, y_plah, predictions, optimizer, loss)


if __name__ == '__main__':
    main()
