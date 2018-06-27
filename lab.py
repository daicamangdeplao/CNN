import tensorflow as tf
from sklearn.datasets import fetch_mldata
from tensorflow.examples.tutorials.mnist import input_data
import random
import numpy as np


def one_hot_encoding(my_list):
    encoding_list = []
    for e in my_list:
        tmp = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        tmp[int(e)] = 1.0
        encoding_list.append(tmp)

    assert len(encoding_list) == len(my_list)
    return np.array(encoding_list)


def main():
    # Load mnist data by tensor
    mnist_data_t = input_data.read_data_sets("/tmp/data/", one_hot=True)
    features_t, labels_t = mnist_data_t.train.next_batch(100)
    print('======== DEBUG: type of labels_t {}'
          '\nshape of labels_t {}'
          '\ndtype of labels_t {}'
          '\nsample of labels_t'.format(type(labels_t), labels_t.shape, labels_t.dtype))
    print('======== DEBUG: type of features_t {}'
          '\nshape of features_t {}'
          '\ndtype of features_t {}'.format(type(features_t), features_t.shape, features_t.dtype))

    # Load input data by sklearn
    mnist_data_s = fetch_mldata('MNIST original')
    features_s = mnist_data_s.data
    labels_s = one_hot_encoding(mnist_data_s.target)
    print('======== DEBUG: type of labels_s {}'
          '\nshape of labels_s {}'
          '\ndtype of labels_s {}'.format(type(labels_s), labels_s.shape, labels_s.dtype))
    print('======== DEBUG: type of features_s {}'
          '\nshape of features_s {}'
          '\ndtype of features_s {}'.format(type(features_s), features_s.shape, features_s.dtype))

    print(labels_t[0])
    print('\n')
    print(mnist_data_s.target[0])
    print(labels_s[0])


if __name__ == '__main__':
    main()
