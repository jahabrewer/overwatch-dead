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
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

# TODO get rid of these constants
IMAGE_HEIGHT=24
IMAGE_WIDTH=64
IMAGE_CHANNELS=3
NUM_CLASSES=2

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

  # TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  # TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  # TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  # TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  # VALIDATION_SIZE = 5000

  # local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
  #                                  SOURCE_URL + TRAIN_IMAGES)
  train_images = extract_images(train_image_file)

  # local_file = base.maybe_download(TRAIN_LABELS, train_dir,
  #                                  SOURCE_URL + TRAIN_LABELS)
  train_labels = extract_labels(train_label_file, one_hot=one_hot)

  # local_file = base.maybe_download(TEST_IMAGES, train_dir,
  #                                  SOURCE_URL + TEST_IMAGES)
  test_images = extract_images(test_image_file)

  # local_file = base.maybe_download(TEST_LABELS, train_dir,
  #                                  SOURCE_URL + TEST_LABELS)
  test_labels = extract_labels(test_label_file, one_hot=one_hot)

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
  validation = DataSet(validation_images,
                       validation_labels,
                       dtype=dtype,
                       reshape=reshape)
  test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

  return base.Datasets(train=train, validation=validation, test=test)

### END MNIST.PY

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
  # sess = tf.InteractiveSession()

  x = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNELS])
  y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

  #W = tf.Variable(tf.zeros([784,10]))
  #b = tf.Variable(tf.zeros([10]))
  W_conv1 = weight_variable([5, 5, IMAGE_CHANNELS, 32])
  b_conv1 = bias_variable([32])

  x_image = tf.reshape(x, [-1,IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])

  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  # W_fc1 = weight_variable([IMAGE_HEIGHT/4 * IMAGE_WIDTH/4 * 64, 1024])
  pooled_dimension = int(IMAGE_HEIGHT/4 * IMAGE_WIDTH/4 * 64)
  W_fc1 = weight_variable([pooled_dimension, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_pool2, [-1, pooled_dimension])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, NUM_CLASSES])
  b_fc2 = bias_variable([NUM_CLASSES])

  y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  sess.run(tf.initialize_all_variables())
  for i in range(1000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
    print('.', end='')
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    sys.stdout.flush()

  print("test accuracy %g"%accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
