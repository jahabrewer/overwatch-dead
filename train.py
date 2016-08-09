#!/usr/bin/env python

# adapted from TensorFlow tutorial code

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import collections

# Import data
# from tensorflow.examples.tutorials.mnist import input_data

import gzip

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

import tensorflow as tf

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("train_image_file", help="IDX file containing training images")
parser.add_argument("train_label_file", help="IDX file containing training labels")
parser.add_argument("test_image_file", help="IDX file containing test images")
parser.add_argument("test_label_file", help="IDX file containing test labels")
parser.add_argument("--validation-size", "-vs", type=int, help="number of training examples to segregate into the validation set")
parser.add_argument("--num-steps", "-ns", type=int, help="number of training steps")
parser.add_argument("--checkpoint-interval", "-ci", type=int, help="number of training steps between each checkpoint")
parser.add_argument("--batch-size", "-b", type=int, help="number of examples to use in each training step")
parser.add_argument("--learning-rate", "-l", type=float, help="learning rate")
parser.add_argument("--dropout", "-d", type=float, help="probability that a neuron will be kept in dropout layers", \
  default=0.5)
parser.add_argument("--summaries-dir", "-s", help="the directory in which to write summary logs", \
  type=str, default="/tmp/overwatch-dead-logs")
parser.add_argument("-v", "--verbose", action="store_true")
parser.add_argument("-t", "--run-test", action="store_true", help="run on test data at the end")
args = parser.parse_args()

# TODO get rid of these constants
IMAGE_HEIGHT=24
IMAGE_WIDTH=24
IMAGE_CHANNELS=3
NUM_CLASSES=24

# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

### START MNIST.PY

def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2052:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    channels = _read32(bytestream)
    buf = bytestream.read(rows * cols * channels * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, channels)
    return data


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False, num_classes=NUM_CLASSES):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
      self._num_examples = images.shape[0]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 3)
      if reshape:
        assert images.shape[3] == IMAGE_CHANNELS
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2] * images.shape[3])
      if dtype == dtypes.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

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

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)
      ]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_image_file,
                   train_label_file,
                   test_image_file,
                   test_label_file,
                   validation_size,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True):
  if fake_data:

    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

    train = fake()
    validation = fake()
    test = fake()
    return base.Datasets(train=train, validation=validation, test=test)

  train_images = extract_images(train_image_file)
  train_labels = extract_labels(train_label_file, one_hot=one_hot)
  test_images = extract_images(test_image_file)
  test_labels = extract_labels(test_label_file, one_hot=one_hot)

  # TODO use shuffle_batch?
  # Shuffle the data.
  train_size = train_images.shape[0]
  train_shuffle_indices = numpy.array(range(train_size))
  numpy.random.shuffle(train_shuffle_indices)
  train_images = train_images[train_shuffle_indices, ...]
  train_labels = train_labels[train_shuffle_indices, ...]
  test_size = test_images.shape[0]
  test_shuffle_indices = numpy.array(range(test_size))
  numpy.random.shuffle(test_shuffle_indices)
  test_images = test_images[test_shuffle_indices, ...]
  test_labels = test_labels[test_shuffle_indices, ...]

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_images,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape)
  if args.run_test:
    test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)
  else:
    test = None

  # Print label composition of sets
  if args.verbose:
    print('Label composition')
    for set_name, labels in [('train', train_labels), ('validation', validation_labels), ('test', test_labels)]:
      print(set_name)
      counts = collections.Counter([l.argmax() for l in labels])

      for label_index in range(NUM_CLASSES):
        print('{0}: {1}'.format(label_index, counts[label_index]))

      print()


  return base.Datasets(train=train, validation=validation, test=test)

### END MNIST.PY

def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
    tf.scalar_summary('sttdev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

mnist = read_data_sets(args.train_image_file,
                       args.train_label_file,
                       args.test_image_file,
                       args.test_label_file,
                       args.validation_size,
                       one_hot=True)

with tf.Session() as sess:

  with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNELS])
    x_image = tf.reshape(x, [-1,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])
    # tf.image_summary('xinput', x_image, max_images=3)
    y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, IMAGE_CHANNELS, 32])
    variable_summaries(W_conv1, 'conv1/weights')
    b_conv1 = bias_variable([32])
    variable_summaries(b_conv1, 'conv1/biases')

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    variable_summaries(W_conv2, 'conv2/weights')
    b_conv2 = bias_variable([64])
    variable_summaries(b_conv2, 'conv2/biases')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

  with tf.name_scope('fc1'):
    pooled_dimension = IMAGE_HEIGHT//4 * IMAGE_WIDTH//4 * 64
    W_fc1 = weight_variable([pooled_dimension, 1024])
    variable_summaries(W_fc1, 'fc1/weights')
    b_fc1 = bias_variable([1024])
    variable_summaries(b_fc1, 'fc1/biases')

    h_pool2_flat = tf.reshape(h_pool2, [-1, pooled_dimension])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, NUM_CLASSES])
    variable_summaries(W_fc2, 'fc2/weights')
    b_fc2 = bias_variable([NUM_CLASSES])
    variable_summaries(b_fc2, 'fc2/biases')

  y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  with tf.name_scope('cross_entropy'):
    diff = y_ * tf.log(y)
    with tf.name_scope('total'):
      cross_entropy = -tf.reduce_mean(diff)
    tf.scalar_summary('cross entropy', cross_entropy)
  train_step = tf.train.AdamOptimizer(args.learning_rate).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  tf.scalar_summary('accuracy', accuracy)

  # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
  merged = tf.merge_all_summaries()
  train_writer = tf.train.SummaryWriter(args.summaries_dir + '/train',
                                        sess.graph)
  validation_writer = tf.train.SummaryWriter(args.summaries_dir + '/validation')
  test_writer = tf.train.SummaryWriter(args.summaries_dir + '/test')

  # log the incorrectly predicted images
  incorrect_prediction = tf.not_equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  incorrect_prediction_indices = tf.where(incorrect_prediction)
  # the flattened images that were incorrectly predicted
  incorrect_x_flat = tf.gather(x, incorrect_prediction_indices)
  # the correctly shaped images
  incorrect_x = tf.reshape(incorrect_x_flat, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
  # the (incorrectly) predicted classes for these images
  incorrect_y = tf.gather(tf.argmax(y, 1), incorrect_prediction_indices)
  incorrect_y = tf.Print(incorrect_y, [incorrect_y], "inc y")
  incorrect_summary_op = tf.image_summary("incorrect", incorrect_x, max_images=100)

  def feed_dict(dataset):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if dataset == 'train':
      xs, ys = mnist.train.next_batch(args.batch_size)
      k = args.dropout
    elif dataset == 'validation':
      xs, ys = mnist.validation.images, mnist.validation.labels
      k = 1.0
    elif dataset == 'test':
      xs, ys = mnist.test.images, mnist.test.labels
      k = 1.0
    else:
      raise ValueError('unknown dataset: {0}'.format(dataset))
    return {x: xs, y_: ys, keep_prob: k}

  sess.run(tf.initialize_all_variables())

  for i in range(args.num_steps):
    if i % args.checkpoint_interval == 0:  # Record summaries and test-set accuracy
      summary, acc, incorrect_summary, _ = sess.run([merged, accuracy, incorrect_summary_op, incorrect_y],
        feed_dict=feed_dict('validation'))
      validation_writer.add_summary(summary, i)
      # only log incorrect images on the last checkpoint
      if i >= (args.num_steps - args.checkpoint_interval):
        validation_writer.add_summary(incorrect_summary, i)
      print('Validation accuracy at step %s: %s' % (i, acc))
    else:  # Record train set summaries, and train
      print('.', end='')
      sys.stdout.flush()
      if i % args.checkpoint_interval == args.checkpoint_interval-1:  # Record execution stats
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        summary, _ = sess.run([merged, train_step],
                              feed_dict=feed_dict('train'),
                              options=run_options,
                              run_metadata=run_metadata)
        train_writer.add_run_metadata(run_metadata, 'step%d' % i)
        train_writer.add_summary(summary, i)
        # print('Adding run metadata for', i)
      else:  # Record a summary
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict('train'))
        train_writer.add_summary(summary, i)

  if args.run_test:
    acc, incorrect_summary, _ = sess.run([accuracy, incorrect_summary_op, incorrect_y], feed_dict=feed_dict('test'))
    test_writer.add_summary(incorrect_summary, args.num_steps)
    print("Final test accuracy {0}".format(acc))
