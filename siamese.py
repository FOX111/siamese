from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np



class Siamese(object):


    def inference(self, x, reuse):

        with tf.variable_scope('conv1', initializer = tf.random_normal_initializer(stddev = 1e-3), reuse = reuse): # First convolution

            kernel = tf.get_variable('kernel', (5, 5, 3, 64))
            biases = tf.get_variable('biases', (64))

            logits = tf.nn.conv2d(x, kernel, strides = [1, 1, 1, 1], padding = 'SAME') + biases # Convolution applied
            logits = tf.nn.relu(logits) # applying non-linearity
            logits = tf.nn.max_pool(logits, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME') # max pooling

        with tf.variable_scope('conv2', initializer = tf.random_normal_initializer(stddev = 1e-3), reuse = reuse): # Second convolution
            kernel = tf.get_variable('kernel', (5, 5, 64, 64))
            biases = tf.get_variable('biases', (64))

            logits = tf.nn.conv2d(logits, kernel, strides = [1, 1, 1, 1], padding = 'SAME') + biases
            logits = tf.nn.relu(logits)
            logits = tf.nn.max_pool(logits, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        with tf.variable_scope('flatten'):
            logits = tf.reshape(logits, [-1, 64*8*8]) # flattening each convolution of an image to a vector

        with tf.variable_scope('fc1', initializer = tf.random_normal_initializer(stddev = 1e-3), reuse = reuse): # First densely connected layer
            weights = tf.get_variable('weights', (64*8*8, 384))
            biases = tf.get_variable('biases', (384))
            logits = tf.matmul(logits, weights) + biases # linear transfromation
            logits = tf.nn.relu(logits) # non-linearity

        with tf.variable_scope('fc2', initializer = tf.random_normal_initializer(stddev = 1e-3), reuse = reuse): # Second densely connected layer
            weights = tf.get_variable('weights', (384, 192))
            biases = tf.get_variable('biases', (192))
            logits = tf.matmul(logits, weights) + biases # linear transfromation
            logits = tf.nn.relu(logits) # non-linearity 
            
        with tf.variable_scope('L2-norm'): # l2-normalization
            l2_out = tf.nn.l2_normalize(logits, dim = 1)
            print (l2_out)

        return l2_out


    def loss(self, channel_1, channel_2, label, margin):

        with tf.name_scope('Loss'):
            label = tf.cast(label, tf.float32)
            with tf.name_scope("L2_distance"):
                d = tf.sqrt(tf.reduce_sum(tf.square(channel_1 - channel_2), 1))
            with tf.name_scope("Similar_loss"):
                similar_loss = label * tf.square(d)

            with tf.name_scope("Dissimilar_loss"):
                dissimilar_loss = (1 - label) * tf.square(tf.maximum(margin - d, 0.))

            with tf.name_scope("Contrastive_loss"):
                loss = tf.reduce_mean(similar_loss + dissimilar_loss)
            
            tf.scalar_summary("Contrastive_loss", loss)
            tf.scalar_summary("Dissimilar_loss", tf.reduce_mean(similar_loss))
            tf.scalar_summary("Similar_loss", tf.reduce_mean(dissimilar_loss))


        return loss

