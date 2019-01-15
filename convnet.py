from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class ConvNet(object):


    def __init__(self, n_classes = 10):

        self.n_classes = n_classes

    def inference(self, x):



        with tf.variable_scope('conv1', initializer = tf.random_normal_initializer(stddev = 1e-3)): # First convolution

            kernel = tf.get_variable('kernel', (5, 5, 3, 64))
            biases = tf.get_variable('biases', (64))

            logits = tf.nn.conv2d(x, kernel, strides = [1, 1, 1, 1], padding = 'SAME') + biases # Convolution applied
            logits = tf.nn.relu(logits) # applying non-linearity
            logits = tf.nn.max_pool(logits, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME') # max pooling

        with tf.variable_scope('conv2', initializer = tf.random_normal_initializer(stddev = 1e-3)): # Second convolution
            kernel = tf.get_variable('kernel', (5, 5, 64, 64))
            biases = tf.get_variable('biases', (64))

            logits = tf.nn.conv2d(logits, kernel, strides = [1, 1, 1, 1], padding = 'SAME') + biases
            logits = tf.nn.relu(logits)
            logits = tf.nn.max_pool(logits, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        with tf.variable_scope('flatten', initializer = tf.random_normal_initializer(stddev = 1e-3)):
            logits = tf.reshape(logits, [-1, 64*8*8]) # flattening each convolution of an image to a vector

        with tf.variable_scope('fc1', initializer = tf.random_normal_initializer(stddev = 1e-3)): # First densely connected layer
            weights = tf.get_variable('weights', (64*8*8, 384))
            biases = tf.get_variable('biases', (384))
            logits = tf.matmul(logits, weights) + biases # linear transfromation
            logits = tf.nn.relu(logits) # non-linearity

            logits = tf.nn.dropout(logits, 0.5)
        with tf.variable_scope('fc2', initializer = tf.random_normal_initializer(stddev = 1e-3)): # Second densely connected layer
            weights = tf.get_variable('weights', (384, 192))
            biases = tf.get_variable('biases', (192))
            logits = tf.matmul(logits, weights) + biases # linear transfromation
            logits = tf.nn.relu(logits) # non-linearity 

            logits = tf.nn.dropout(logits, 0.5)
        with tf.variable_scope('fc3', initializer = tf.random_normal_initializer(stddev = 1e-3)): # Second densely connected layer
            weights = tf.get_variable('weights', (192, 10))
            biases = tf.get_variable('biases', (10))
            logits = tf.matmul(logits, weights) + biases # linear transfromation
            # softmax is embedded into loss and accuracy function
        #logits = tf.nn.dropout(logits, 0.5)
        return logits

    def accuracy(self, logits, labels):

        with tf.name_scope('Accuracy'):
          correct = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
          accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        tf.scalar_summary('accuracy', accuracy)
        return accuracy

    def loss(self, logits, labels):

        with tf.name_scope('Loss'):
          cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    tf.cast(logits, tf.float32), tf.cast(labels, tf.float32), name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')


        penalty_coef = 0.2
        with tf.variable_scope('fc1', reuse = True):
            weights = tf.get_variable('weights', (64*8*8, 384))
            biases = tf.get_variable('biases', (384))
        pen_term = tf.nn.l2_loss(weights)
        pen_term += tf.nn.l2_loss(biases)
        with tf.variable_scope('fc2', reuse = True):
            weights = tf.get_variable('weights', (384, 192))
            biases = tf.get_variable('biases', (192))
        pen_term += tf.nn.l2_loss(weights)
        pen_term += tf.nn.l2_loss(biases)
        with tf.variable_scope('fc3', reuse = True):
            weights = tf.get_variable('weights', (192, 10))
            biases = tf.get_variable('biases', (10))
        pen_term += tf.nn.l2_loss(weights)
        pen_term += tf.nn.l2_loss(biases)

        loss -= penalty_coef * pen_term

        tf.scalar_summary('loss', loss)


        return loss
