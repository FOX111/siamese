from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import tensorflow as tf
import numpy as np

from convnet import *
from siamese import *
import cifar10_utils
import cifar10_siamese_utils

import h5py
import ast

LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 128
MAX_STEPS_DEFAULT = 15000
EVAL_FREQ_DEFAULT = 1000
CHECKPOINT_FREQ_DEFAULT = 5000
PRINT_FREQ_DEFAULT = 10
OPTIMIZER_DEFAULT = 'ADAM'

DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'
LOG_DIR_DEFAULT = './logs/cifar10'
CHECKPOINT_DIR_DEFAULT = './checkpoints'



def train():

    tf.set_random_seed(42)
    np.random.seed(42)

    max_steps = int(FLAGS.max_steps)
    learning_rate = float(FLAGS.learning_rate)
    batch_size = int(FLAGS.batch_size)
    eval_freq = int(FLAGS.eval_freq)
    print_freq = int(FLAGS.print_freq)



    conv_net = ConvNet()
    labels_placeholder = tf.placeholder(tf.int32, shape=(None, 10))
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

    logits = conv_net.inference(images_placeholder)
    loss = conv_net.loss(logits, labels_placeholder)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    accuracy = conv_net.accuracy(logits, labels_placeholder)

    cifar10 = cifar10_utils.get_cifar10('cifar10/cifar10-10-batches-py', one_hot = True)
    x_test, y_test = cifar10.test.images, cifar10.test.labels
    x_test = x_test.reshape((-1, 32, 32, 3))  #Creating 4D ndarray from a set

    sess = tf.InteractiveSession()
    writer_test = tf.train.SummaryWriter(FLAGS.log_dir + '/linear/test', sess.graph)
    writer_train = tf.train.SummaryWriter(FLAGS.log_dir + '/linear/train', sess.graph)
    sess.run(tf.initialize_all_variables())

    for i in range(max_steps):
        merged = tf.merge_all_summaries()

        x_batch, y_batch = cifar10.train.next_batch(batch_size)
        x_batch = x_batch.reshape((-1, 32, 32, 3)) #Creating 4D ndarray from a batch
        feed_dict_train = {images_placeholder: x_batch, 
                           labels_placeholder: y_batch}
        if i % print_freq == 0:
            _, merged_summary, value_accuracy = sess.run([train_step, merged, accuracy],
                                        feed_dict=feed_dict_train)
            print('Iteration: ', i, 'Train accuracy: ', value_accuracy)
            writer_train.add_summary(merged_summary, i)
        else:
            sess.run(train_step, feed_dict = feed_dict_train)

        if i % eval_freq == 0:
            feed_dict_test = {images_placeholder: x_test,
                             labels_placeholder: y_test}
            value_accuracy, merged_summary = sess.run([accuracy, merged],
                                     feed_dict = feed_dict_test)
            writer_test.add_summary(merged_summary, i)
            print('Iteration: ', i, 'Test accuracy: ', value_accuracy)





    saver = tf.train.Saver()
    saver.save(sess, "./checkpoints/linear/model.ckpt")



def train_siamese():

    tf.set_random_seed(42)
    np.random.seed(42)

    max_steps = int(FLAGS.max_steps)
    learning_rate = float(FLAGS.learning_rate)
    batch_size = int(FLAGS.batch_size)
    eval_freq = int(FLAGS.eval_freq)
    print_freq = int(FLAGS.print_freq)

    n_batches_test = 500

    siam_net = Siamese()
    labels_placeholder = tf.placeholder(tf.int32, shape=(None))
    l_images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    r_images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

    left = siam_net.inference(l_images_placeholder, reuse = None)
    right = siam_net.inference(r_images_placeholder, reuse = True)

    loss = siam_net.loss(left, right, labels_placeholder, margin = 0.3)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    cifar10 = cifar10_siamese_utils.get_cifar10('cifar10/cifar10-10-batches-py', one_hot = False)
    test_dset = cifar10_siamese_utils.create_dataset(cifar10.test, num_tuples = n_batches_test, batch_size = batch_size)

    sess = tf.InteractiveSession()
    writer_test = tf.train.SummaryWriter(FLAGS.log_dir + '/siamese/try/test', sess.graph)
    writer_train = tf.train.SummaryWriter(FLAGS.log_dir + '/siamese/try/train', sess.graph)
    sess.run(tf.initialize_all_variables())
    merged = tf.merge_all_summaries()
    saver = tf.train.Saver()

    for i in range(max_steps):

        left_train, right_train, labels_train = cifar10.train.next_batch(batch_size)
        feed_dict_train = {l_images_placeholder: left_train,
                           r_images_placeholder: right_train,
                           labels_placeholder: labels_train}

        if i % print_freq == 0:
            _, merged_summary, loss_val = sess.run([train_step, merged, loss], feed_dict=feed_dict_train)
            writer_train.add_summary(merged_summary, i)
            print('Iteration ', i, ' Loss', loss_val)
        else:
            sess.run(train_step, feed_dict = feed_dict_train) 

        if i % eval_freq == 0:
            loss_sum = 0.0
            for batch in test_dset:
                left_test, right_test, labels_test = batch
                feed_dict_test = {l_images_placeholder: left_test,
                                  r_images_placeholder: right_test,
                                  labels_placeholder: labels_test}
                loss_val = sess.run(loss, feed_dict = feed_dict_test)
                loss_sum += loss_val
            loss_av = loss_sum / len(test_dset)
            print('Average loss:' , loss_av)
            writer_test.add_summary(sess.run(tf.scalar_summary('Av_loss', loss_av)), i)

        if (i == 5000):
            saver.save(sess, "./checkpoints/siamese/model5k.ckpt")
        if (i == 10000):
            saver.save(sess, "./checkpoints/siamese/model10k.ckpt")
        if (i == 15000):
            saver.save(sess, "./checkpoints/siamese/model15k.ckpt")
        if (i == 20000):
            saver.save(sess, "./checkpoints/siamese/model20k.ckpt")



def feature_extraction():
    sess = tf.InteractiveSession()
    if (FLAGS.train_model == 'linear'):
        images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(None, 10))
        conv_net = ConvNet()
        logits = conv_net.inference(images_placeholder)
        saver = tf.train.Saver()
        saver.restore(sess, "./checkpoints/linear/model.ckpt")
        
        fc2_relu = tf.get_default_graph().get_tensor_by_name("fc2/Relu:0")
        fc1_relu = tf.get_default_graph().get_tensor_by_name("fc1/Relu:0")
        flatten = tf.get_default_graph().get_tensor_by_name("flatten/Reshape:0")

        cifar10 = cifar10_utils.get_cifar10('cifar10/cifar10-10-batches-py', one_hot = True)
        x_test, y_test = cifar10.test.images, cifar10.test.labels
        x_train, y_train = cifar10.train.images, cifar10.train.labels
        x_test = x_test.reshape((-1, 32, 32, 3))  #Creating 4D ndarray from a test set
        x_train = x_train.reshape((-1, 32, 32, 3)) #Creating 4D ndarray from a train set
        feed_dict_test = {images_placeholder: x_test,
                        labels_placeholder: y_test}
        feed_dict_train = {images_placeholder: x_train,
                        labels_placeholder: y_train}
        fc2_relu_features, fc1_relu_features, flatten_features = sess.run([fc2_relu, fc1_relu, flatten], feed_dict = feed_dict_test)

        with h5py.File('./features/linear/features_test.h5', 'w') as hf:
            hf.create_dataset('test:fc2_relu', data=fc2_relu_features)
            hf.create_dataset('test:fc1_relu', data=fc1_relu_features)
            hf.create_dataset('test:flatten', data=flatten_features)

        fc2_relu_features, fc1_relu_features, flatten_features = sess.run([fc2_relu, fc1_relu, flatten], feed_dict = feed_dict_train)

        with h5py.File('./features/linear/features_train.h5', 'w') as hf:
            hf.create_dataset('train:fc2_relu', data=fc2_relu_features)
            hf.create_dataset('train:fc1_relu', data=fc1_relu_features)
            hf.create_dataset('train:flatten', data=flatten_features)



    if (FLAGS.train_model == 'siamese'):
        images_placeholder = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
        #labels_placeholder = tf.placeholder(tf.int32, shape=(None))
        siamese_net = Siamese()
        logits = siamese_net.inference(images_placeholder, reuse = None)
        print(logits.name)
        saver = tf.train.Saver()
        saver.restore(sess, "./checkpoints/siamese/model10k.ckpt")
        
        #l2 = tf.get_default_graph().get_tensor_by_name("L2-norm/l2_normalize:0")
        l2 = logits

        cifar10 = cifar10_utils.get_cifar10('cifar10/cifar10-10-batches-py', one_hot = False)
        x_test, y_test = cifar10.test.images, cifar10.test.labels
        x_train, y_train = cifar10.train.images, cifar10.train.labels
        x_test = x_test.reshape((-1, 32, 32, 3))  #Creating 4D ndarray from a test set
        x_train = x_train.reshape((-1, 32, 32, 3)) #Creating 4D ndarray from a train set
        feed_dict_test = {images_placeholder: x_test}
        feed_dict_train = {images_placeholder: x_train}

        l2_features = sess.run(l2, feed_dict = feed_dict_test)
        with h5py.File('./features/siamese/features_test.h5', 'w') as hf:
            hf.create_dataset('test:l2', data=l2_features)
            hf.create_dataset('test:labels', data=y_test)
        l2_features = sess.run(l2, feed_dict = feed_dict_train)
        with h5py.File('./features/siamese/features_train.h5', 'w') as hf:
            hf.create_dataset('train:l2', data=l2_features)
            hf.create_dataset('train:labels', data=y_train)



def initialize_folders():
    """
    Initializes all folders in FLAGS variable.
    """

    if not tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.MakeDirs(FLAGS.log_dir)

    if not tf.gfile.Exists(FLAGS.data_dir):
        tf.gfile.MakeDirs(FLAGS.data_dir)

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main(_):
    print_flags()

    initialize_folders()
    FLAGS.is_train = ast.literal_eval(FLAGS.is_train)
    if FLAGS.is_train:
        if FLAGS.train_model == 'linear':
            train()
        elif FLAGS.train_model == 'siamese':
            train_siamese()
        else:
            raise ValueError("--train_model argument can be linear or siamese")
    else:
        feature_extraction()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')

    parser.add_argument('--max_steps', type = int, default = MAX_STEPS_DEFAULT,
                      help='Number of steps to run trainer.')

    parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')

    parser.add_argument('--print_freq', type = int, default = PRINT_FREQ_DEFAULT,
                      help='Frequency of evaluation on the train set')

    parser.add_argument('--eval_freq', type = int, default = EVAL_FREQ_DEFAULT,
                      help='Frequency of evaluation on the test set')

    parser.add_argument('--checkpoint_freq', type = int, default = CHECKPOINT_FREQ_DEFAULT,
                      help='Frequency with which the model state is saved.')

    parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')

    parser.add_argument('--log_dir', type = str, default = LOG_DIR_DEFAULT,
                      help='Summaries log directory')

    parser.add_argument('--checkpoint_dir', type = str, default = CHECKPOINT_DIR_DEFAULT,
                      help='Checkpoint directory')

    parser.add_argument('--is_train', type = str, default = 'True',
                      help='Training or feature extraction')

    parser.add_argument('--train_model', type = str, default = 'linear',
                      help='Type of model. Possible options: linear and siamese')

    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
