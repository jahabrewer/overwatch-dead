#!/usr/bin/env python

import sys
import tensorflow as tf
import numpy as np
from PIL import Image

# Remember to generate a file name queue of you 'train.TFRecord' file path
def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'image/height': tf.FixedLenFeature([], tf.int64),
      'image/width': tf.FixedLenFeature([], tf.int64),
      'image/colorspace': tf.FixedLenFeature([], tf.string),
      'image/channels': tf.FixedLenFeature([], tf.int64),
      'image/class/label': tf.FixedLenFeature([], tf.int64),
      'image/class/text': tf.FixedLenFeature([], tf.string),
      'image/format': tf.FixedLenFeature([], tf.string),
      'image/filename': tf.FixedLenFeature([], tf.string),
      'image/encoded': tf.FixedLenFeature([], tf.string)
    })

  # Convert from a scalar string tensor (whose single string has
  # image = tf.decode_raw(features['image/encoded'], tf.uint8)
  image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
  height = tf.cast(features['image/height'], tf.int32)
  width = tf.cast(features['image/width'], tf.int32)
  channels = tf.cast(features['image/channels'], tf.int32)

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['image/class/label'], tf.int32)

  image = tf.reshape(image, tf.pack([height, width, channels]))
  image.set_shape([34,11,3])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  # image = tf.cast(image, tf.float32)
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Flatten
  image = tf.reshape(image, [-1])

  return image, label

# from fully_connected_reader.py
def inputs(filename, batch_size, num_epochs):
  if not num_epochs: num_epochs = None
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])

    image, label = read_and_decode(filename_queue)

    images, sparse_labels = tf.train.shuffle_batch(
      [image, label], batch_size=batch_size, capacity=1000 + 3*batch_size,
      min_after_dequeue=1000)

    return images, sparse_labels

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

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

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses relu to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      weights = weight_variable([input_dim, output_dim])
      variable_summaries(weights, layer_name + '/weights')
    with tf.name_scope('biases'):
      biases = bias_variable([output_dim])
      variable_summaries(biases, layer_name + '/biases')
    with tf.name_scope('Wx_plus_b'):
      preactivate = tf.matmul(input_tensor, weights) + biases
      tf.histogram_summary(layer_name + '/pre_activations', preactivate)
    activations = act(preactivate, 'activation')
    tf.histogram_summary(layer_name + '/activations', activations)
    return activations

# def feed_dict(train, train_x, train_y_, test_x, test_y_, dropout):
#   """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
#   if train:
#     xs, ys = train_x, train_y_
#     k = dropout
#   else:
#     xs, ys = test_x, test_y_
#     k = 1.0
#   return dict(x= xs, y_= ys, keep_prob= k)
  # return {x: xs, y_: ys, keep_prob: k}

def main(training_file, validation_file):
  with tf.Session() as sess:
    # TODO 1122 is the hardcoded size of the image
    IMAGE_LENGTH=1122
    OUTPUT_CLASSES=2
    L1_L2_CONNECTIVITY = 800
    DROPOUT=0.9 # keep probability
    LEARNING_RATE=0.001
    NUM_STEPS=101

    # inputs
    with tf.name_scope('input'):
      x = tf.placeholder(tf.float32, [None, IMAGE_LENGTH], name='x-input')
      y_ = tf.placeholder(tf.float32, [None, OUTPUT_CLASSES], name='y-input')
    
    hidden1 = nn_layer(x, IMAGE_LENGTH, L1_L2_CONNECTIVITY, 'layer1')

    with tf.name_scope('dropout'):
      keep_prob = tf.placeholder(tf.float32)
      tf.scalar_summary('dropout_keep_probability', keep_prob)
      dropped = tf.nn.dropout(hidden1, keep_prob)

    y = nn_layer(dropped, L1_L2_CONNECTIVITY, OUTPUT_CLASSES, 'layer2', act=tf.nn.softmax)
    # y = nn_layer(x, 1122, 2, 'layer1', tf.nn.softmax)

    with tf.name_scope('cross_entropy'):
      diff = y_ * tf.log(y)
      with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(diff, reduction_indices=[1]))
      tf.scalar_summary('cross entropy', cross_entropy)
    
    with tf.name_scope('train'):
      train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.scalar_summary('accuracy', accuracy)

    # summary
    merged = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/tmp/overwatch-dead-logs', sess.graph)

    # filename_queue = tf.train.string_input_producer([training_file])
    # TODO hardcoded numbers of examples
    NUM_TRAINING_EXAMPLES = 6695
    NUM_VALIDATION_EXAMPLES = 1663
    training_examples_op, training_labels_op = inputs(training_file, NUM_TRAINING_EXAMPLES, 1)
    validation_examples_op, validation_labels_op = inputs(validation_file, NUM_VALIDATION_EXAMPLES, 1)
    training_labels_sliced_op = tf.slice(tf.one_hot(training_labels_op, 3), [0, 1], [-1, -1])
    validation_labels_sliced_op = tf.slice(tf.one_hot(validation_labels_op, 3), [0, 1], [-1, -1])

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # this prevents modification of the graph
    tf.get_default_graph().finalize()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    training_examples = sess.run(training_examples_op)
    # must slice because labels are 1 and 2, 0 is skipped by build_image_data
    training_labels = sess.run(training_labels_sliced_op)

    validation_examples = sess.run(validation_examples_op)
    # must slice because labels are 1 and 2, 0 is skipped by build_image_data
    validation_labels = sess.run(validation_labels_sliced_op)

    # train
    for i in range(NUM_STEPS):
      if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy],
          feed_dict={x: validation_examples, y_:validation_labels, keep_prob:1.0})
        summary_writer.add_summary(summary, i)
        print('Accuracy at step {0}: {1}'.format(i, acc))
      else:
        sess.run(train_step,
          feed_dict={x: training_examples, y_:training_labels, keep_prob:DROPOUT})

    # show loss on training data
    print(sess.run(accuracy, feed_dict={x: training_examples, y_: training_labels, keep_prob:1.0}))

    # show loss on validation data
    print(sess.run(accuracy, feed_dict={x: validation_examples, y_: validation_labels, keep_prob:1.0}))

    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
  if len(sys.argv) != 3:
    print("Usage: {0} <train TFRecord file> <validation TFRecord file>".format(sys.argv[0]))
    exit()

  training_file = sys.argv[1]
  validation_file = sys.argv[2]
  main(training_file, validation_file)

