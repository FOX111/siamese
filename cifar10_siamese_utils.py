
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cPickle as pickle

from six.moves import xrange

import random
from tensorflow.contrib.learn.python.learn.datasets import base


CIFAR10_FOLDER = 'cifar10/cifar-10-batches-py'

def load_cifar10_batch(batch_filename):

  with open(batch_filename, 'rb') as f:
    batch = pickle.load(f)
    X = batch['data']
    Y = batch['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.float32)
    Y = np.array(Y)
    return X, Y

def load_cifar10(cifar10_folder):

  Xs = []
  Ys = []
  for b in range(1, 6):
    batch_filename = os.path.join(cifar10_folder, 'data_batch_' + str(b))
    X, Y = load_cifar10_batch(batch_filename)
    Xs.append(X)
    Ys.append(Y)
  X_train = np.concatenate(Xs)
  Y_train = np.concatenate(Ys)
  X_test, Y_test = load_cifar10_batch(os.path.join(cifar10_folder, 'test_batch'))
  return X_train, Y_train, X_test, Y_test

def get_cifar10_raw_data(data_dir):


  X_train, Y_train, X_test, Y_test = load_cifar10(data_dir)

  return X_train, Y_train, X_test, Y_test

def preprocess_cifar10_data(X_train_raw, Y_train_raw, X_test_raw, Y_test_raw):

  X_train = X_train_raw.copy()
  Y_train = Y_train_raw.copy()
  X_test = X_test_raw.copy()
  Y_test = Y_test_raw.copy()

  # Substract the mean
  mean_image = np.mean(X_train, axis=0)
  X_train -= mean_image
  X_test -= mean_image

  return X_train, Y_train, X_test, Y_test

def dense_to_one_hot(labels_dense, num_classes):

  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

def create_dataset(data, source='Train', num_tuples = 500, batch_size = 128, fraction_same = 0.2):

  dset = []
  for i in range(num_tuples):
    minibatch = []
    left, right, labels = data.next_batch(batch_size)
    minibatch.append(left)
    minibatch.append(right)
    minibatch.append(labels)
    dset.append(minibatch)

  return dset

class DataSet(object):

  def __init__(self, images, labels):

    assert images.shape[0] == labels.shape[0], (
          "images.shape: {0}, labels.shape: {1}".format(str(images.shape), str(labels.shape)))

    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._id_list = []

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fraction_same = 0.2):
    same_count = int(np.floor(batch_size*fraction_same))
    opp_count = batch_size - same_count
    left = []    #batch for the left branch of siamese 
    right = []   #batch for the right branch of siamese
    labels = []  #1 - similar, 0 - dissimilar
    all_indices = np.arange(self.num_examples)
    index_same = np.random.choice(all_indices) #one of the images in this batch will always be from index_same class
    #same_indices = np.random.choice(all_indices, size = same_count, replace = True)
    same_indices = np.ones(same_count)*index_same
    same_indices = [int(index) for index in same_indices]
    opp_indices = np.random.choice(all_indices, size = opp_count, replace = True)
    # generating similar labels
    for i in same_indices:
      left.append(self.images[i])
      found_pair = False
      while found_pair != True:
        j = np.random.choice(all_indices)
        if (self.labels[j] == self.labels[i]) & (i!=j):
          right.append(self.images[j])
          found_pair = True 
          labels.append(0)
          #print('Picked indices (similar): ', i, j)

    # generating dissimilar images
    for i in opp_indices:
      left.append(self.images[i])
      found_pair = False
      while found_pair != True:
        j = np.random.choice(all_indices)
        if self.labels[j] != self.labels[i]:
          right.append(self.images[j])
          found_pair = True
          labels.append(1)


    return left, right, labels

def read_data_sets(data_dir, one_hot = True, validation_size = 0):

  train_images_raw, train_labels_raw, test_images_raw, test_labels_raw = \
      get_cifar10_raw_data(data_dir)
  train_images, train_labels, test_images, test_labels = \
      preprocess_cifar10_data(train_images_raw, train_labels_raw, test_images_raw, test_labels_raw)

  # Apply one-hot encoding if specified
  if one_hot:
    num_classes = len(np.unique(train_labels))
    train_labels = dense_to_one_hot(train_labels, num_classes)
    test_labels = dense_to_one_hot(test_labels, num_classes)

  # Subsample the validation set from the train set
  if not 0 <= validation_size <= len(train_images):
    raise ValueError("Validation size should be between 0 and {0}. Received: {1}.".format(
        len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  # Create datasets
  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)


def get_cifar10(data_dir = CIFAR10_FOLDER, one_hot = True, validation_size = 0):

  return read_data_sets(data_dir, one_hot, validation_size)
