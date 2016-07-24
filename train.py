#!/usr/bin/env python

from __future__ import print_function
import sys
import tensorflow as tf
import numpy as np
from PIL import Image
import argparse


# parse args
parser = argparse.ArgumentParser()
parser.add_argument("training_file", help="TFRecord file containing training examples")
parser.add_argument("validation_file", help="TFRecord file containing validation examples")
parser.add_argument("--image-width", "-iw", type=int)
parser.add_argument("--image-height", "-ih", type=int)
parser.add_argument("--image-channels", "-ic", type=int, choices=[1,3,4], help="number of color channels in the images")
parser.add_argument("--dropout", "-d", type=float, help="keep probability in the dropout layer")
parser.add_argument("--learning-rate", "-l", type=float)
parser.add_argument("--steps", "-s", type=int, help="number of iterations the model will be trained")
parser.add_argument("--checkpoint-interval", "-ci", type=int, help="interval in steps at which a summary will be output")
parser.add_argument("--num-validation-examples", "-tv", type=int, help="number of examples in the validation file")
parser.add_argument("--batch-size", "-b", type=int, help="number of shuffled examples to use in each training step")
args = parser.parse_args()

IMAGE_WIDTH=args.image_width
IMAGE_HEIGHT=args.image_height
IMAGE_CHANNELS=args.image_channels
IMAGE_SIZE=IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS
OUTPUT_CLASSES=2
DROPOUT=args.dropout # keep probability
LEARNING_RATE=args.learning_rate
NUM_STEPS=args.steps
# TODO hardcoded numbers of examples
NUM_VALIDATION_EXAMPLES = args.num_validation_examples

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
  image = tf.image.decode_jpeg(features['image/encoded'], channels=3)
  height = tf.cast(features['image/height'], tf.int32)
  width = tf.cast(features['image/width'], tf.int32)
  channels = tf.cast(features['image/channels'], tf.int32)
  filename = features['image/filename']

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['image/class/label'], tf.int32)

  # image = tf.reshape(image, tf.pack([height, width, channels]))
  # image.set_shape([IMAGE_WIDTH,IMAGE_HEIGHT,IMAGE_CHANNELS])
  image.set_shape([IMAGE_HEIGHT,IMAGE_WIDTH,IMAGE_CHANNELS])

  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.

  # Convert from [0, 255] -> [-0.5, 0.5] floats.
  # image = tf.cast(image, tf.float32)
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5

  # Flatten
  image = tf.reshape(image, [-1])

  return image, label, filename

# from fully_connected_reader.py
def inputs(filename, batch_size, num_epochs, shuffle=True):
  if not num_epochs: num_epochs = None
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer([filename])

    image, label, filename = read_and_decode(filename_queue)

    if shuffle:
      image_batch, label_batch, filename_batch = tf.train.shuffle_batch(
        [image, label, filename], batch_size=batch_size, capacity=1000 + 3*batch_size,
        min_after_dequeue=1000)
    else:
      image_batch, label_batch, filename_batch = tf.train.batch(
        [image, label, filename], batch_size=batch_size, capacity=1000 + 3*batch_size)

    # tf.image_summary('images/' + filename,
    #   tf.reshape(image_batch, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]))

    return image_batch, label_batch, filename_batch

def weight_variable(shape, stddev=0.1):
  initial = tf.truncated_normal(shape, stddev=stddev)
  return tf.Variable(initial)

def bias_variable(shape, constant=0.1):
  """Create a bias variable with appropriate initialization."""
  initial = tf.constant(constant, shape=shape)
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

def inference_2_layer(x):
  hidden1 = nn_layer(x, IMAGE_SIZE, IMAGE_SIZE, 'layer1')

  # with tf.name_scope('dropout'):
  #   keep_prob = tf.placeholder(tf.float32)
  #   tf.scalar_summary('dropout_keep_probability', keep_prob)
  #   dropped = tf.nn.dropout(hidden1, keep_prob)

  y = nn_layer(hidden1, IMAGE_SIZE, OUTPUT_CLASSES, 'layer2', act=tf.nn.softmax)
  # y = nn_layer(x, 1122, 2, 'layer1', tf.nn.softmax)
  return y

def conv2d(x, W):
  # x/input should have shape [batch, in_height, in_width, in_channels]
  # W/filter should have shape [filter_height, filter_width, in_channels, out_channels]
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# def max_pool_2x2(x):
#   return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def inference_conv(x):
  filter_width = 5
  filter_height = 5
  output_depth_1 = 16
  output_depth_2 = 32
  fc1_dim = 512
  fc2_dim = 256

  W_conv1 = weight_variable([filter_height, filter_width, IMAGE_CHANNELS, output_depth_1])
  b_conv1 = bias_variable([output_depth_1])

  x_image = tf.reshape(x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])

  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  # h_pool1 = max_pool_2x2(h_conv1)
  norm1 = tf.nn.lrn(h_conv1)

  W_conv2 = weight_variable([filter_height, filter_width, output_depth_1, output_depth_2])
  b_conv2 = bias_variable([output_depth_2])

  h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)
  # h_pool2 = max_pool_2x2(h_conv2)
  norm2 = tf.nn.lrn(h_conv2)

  # W_fc1 = weight_variable([IMAGE_HEIGHT * IMAGE_WIDTH / 16 * output_depth_2, fc1_dim])
  W_fc1 = weight_variable([IMAGE_HEIGHT * IMAGE_WIDTH * output_depth_2, fc1_dim])
  b_fc1 = bias_variable([fc1_dim])

  # h_conv2_flat = tf.reshape(h_conv2, [-1, IMAGE_HEIGHT * IMAGE_WIDTH / 16 * output_depth_2])
  h_conv2_flat = tf.reshape(norm2, [-1, IMAGE_HEIGHT * IMAGE_WIDTH * output_depth_2])
  h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([fc1_dim, fc2_dim])
  b_fc2 = bias_variable([fc2_dim])

  h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

  W_fc3 = weight_variable([fc2_dim, OUTPUT_CLASSES])
  b_fc3 = bias_variable([OUTPUT_CLASSES])

  return tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3), keep_prob

# adapted from cifar10
# def inference_conv2(x):
#   # conv1
#   kernel = weight_variable([5,5,IMAGE_CHANNELS,64], 1e-4)
#   conv = conv2d(x, kernel)
#   biases = bias_variable([64], 0.0)
#   bias = tf.nn.bias_add(conv, biases)
#   conv1 = tf.nn.relu(bias)

#   # pool1
#   pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='pool1')

#   # norm1
#   norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

def main(training_file, validation_file):
  with tf.Session() as sess:
    # inputs
    with tf.name_scope('input'):
      x = tf.placeholder(tf.float32, [None, IMAGE_SIZE], name='x-input')
      y_ = tf.placeholder(tf.float32, [None, OUTPUT_CLASSES], name='y-input')

    # y = inference_2_layer(x)
    y, keep_prob = inference_conv(x)

    with tf.name_scope('cross_entropy'):
      diff = y_ * tf.log(y)
      with tf.name_scope('total'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(diff, reduction_indices=[1]))
      tf.scalar_summary('cross entropy', cross_entropy)
    
    with tf.name_scope('train'):
      # train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
      train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
      with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.scalar_summary('accuracy', accuracy)

    # log the incorrectly predicted images
    incorrect_prediction = tf.not_equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    incorrect_prediction_indices = tf.where(incorrect_prediction)
    # the flattened images that were incorrectly predicted
    incorrect_x_flat = tf.gather(x, incorrect_prediction_indices)
    # the correctly shaped images
    incorrect_x = tf.reshape(incorrect_x_flat, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    # the (incorrectly) predicted classes for these images
    #incorrect_y = tf.gather(tf.argmax(y, 1), incorrect_prediction_indices)
    tf.image_summary("incorrect", incorrect_x, max_images=100)

    # TODO inputs is shuffling. we just want to load all validation examples.
    training_examples_op, training_labels_op, training_filename_op = \
      inputs(training_file, args.batch_size, 1)
    validation_examples_op, validation_labels_op, validation_filename_op = \
      inputs(validation_file, NUM_VALIDATION_EXAMPLES, 1, shuffle=False)
    # must slice because labels are 1 and 2, 0 is skipped by build_image_data
    training_labels_sliced_op = tf.slice(tf.one_hot(training_labels_op, 3), [0, 1], [-1, -1])
    validation_labels_sliced_op = tf.slice(tf.one_hot(validation_labels_op, 3), [0, 1], [-1, -1])

    # summary
    merged = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/tmp/overwatch-dead-logs', sess.graph)

    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    # this prevents modification of the graph
    tf.get_default_graph().finalize()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    validation_examples = sess.run(validation_examples_op)
    validation_labels = sess.run(validation_labels_sliced_op)
    validation_filenames = sess.run(validation_filename_op)

    # train
    for i in range(NUM_STEPS):
      training_examples, training_labels, training_filenames = \
        sess.run([training_examples_op, training_labels_sliced_op, training_filename_op])

      if i % args.checkpoint_interval == 0:
        summary, acc = sess.run([merged, accuracy],
          feed_dict={x: training_examples, y_: training_labels, keep_prob: 1.0})
        summary_writer.add_summary(summary, i)
        print()
        print('Training accuracy at step {0}: {1}'.format(i, acc))
      # else:

      sess.run(train_step,
        feed_dict={x: training_examples, y_: training_labels, keep_prob: DROPOUT})
      print('.', end='')
      sys.stdout.flush()

    # show loss on training data
    # print(sess.run(accuracy, feed_dict={x: training_examples, y_: training_labels}))

    # show loss on validation data
    print("Final validation accuracy:")
    print(sess.run(accuracy, feed_dict={x: validation_examples, y_: validation_labels, keep_prob: 1.0}))

    coord.request_stop()
    coord.join(threads)

if __name__ == "__main__":
  main(args.training_file, args.validation_file)

