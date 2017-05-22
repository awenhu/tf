
# coding: utf-8

# ### 1. data helper

# In[1]:

import numpy as np
import re
import itertools
import codecs
from collections import Counter


def clean_str(string):
  """
  Tokenization/string cleaning for all datasets except for SST.
  Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
  """
  string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
  string = re.sub(r"\'s", " \'s", string)
  string = re.sub(r"\'ve", " \'ve", string)
  string = re.sub(r"n\'t", " n\'t", string)
  string = re.sub(r"\'re", " \'re", string)
  string = re.sub(r"\'d", " \'d", string)
  string = re.sub(r"\'ll", " \'ll", string)
  string = re.sub(r",", " , ", string)
  string = re.sub(r"!", " ! ", string)
  string = re.sub(r"\(", " \( ", string)
  string = re.sub(r"\)", " \) ", string)
  string = re.sub(r"\?", " \? ", string)
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip().lower()


def load_data_and_labels():
  """
  Loads MR polarity data from files, splits the data into words and generates labels.
  Returns split sentences and labels.
  """
  # Load data from files
  positive_examples = list(codecs.open("./data/chinese/pos.txt", "r", "utf-8").readlines())
  positive_examples = [s.strip() for s in positive_examples]
  negative_examples = list(codecs.open("./data/chinese/neg.txt", "r", "utf-8").readlines())
  negative_examples = [s.strip() for s in negative_examples]
  # Split by words
  x_text = positive_examples + negative_examples
  # x_text = [clean_str(sent) for sent in x_text]
  x_text = [list(s) for s in x_text]

  # Generate labels
  positive_labels = [[0, 1] for _ in positive_examples]
  negative_labels = [[1, 0] for _ in negative_examples]
  y = np.concatenate([positive_labels, negative_labels], 0)
  return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
  """
  Pads all sentences to the same length. The length is defined by the longest sentence.
  Returns padded sentences.
  """
  sequence_length = max(len(x) for x in sentences)
  padded_sentences = []
  for i in range(len(sentences)):
    sentence = sentences[i]
    num_padding = sequence_length - len(sentence)
    new_sentence = sentence + [padding_word] * num_padding
    padded_sentences.append(new_sentence)
  return padded_sentences


def build_vocab(sentences):
  """
  Builds a vocabulary mapping from word to index based on the sentences.
  Returns vocabulary mapping and inverse vocabulary mapping.
  """
  # Build vocabulary
  word_counts = Counter(itertools.chain(*sentences))
  # Mapping from index to word
  vocabulary_inv = [x[0] for x in word_counts.most_common()]
  # Mapping from word to index
  vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
  return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
  """
  Maps sentencs and labels to vectors based on a vocabulary.
  """
  x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
  y = np.array(labels)
  return [x, y]


def load_data():
  """
  Loads and preprocessed data for the MR dataset.
  Returns input vectors, labels, vocabulary, and inverse vocabulary.
  """
  # Load and preprocess data
  sentences, labels = load_data_and_labels()
  sentences_padded = pad_sentences(sentences)
  vocabulary, vocabulary_inv = build_vocab(sentences_padded)
  x, y = build_input_data(sentences_padded, labels, vocabulary)
  return [x, y, vocabulary, vocabulary_inv]


def batch_iter(data, batch_size, num_epochs):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int(len(data)/batch_size) + 1
  for epoch in range(num_epochs):
    # Shuffle the data at each epoch
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield shuffled_data[start_index:end_index]


# ### 2. text cnn

# In[55]:

import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".
  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (isinstance(args, (list, tuple)) and not args):
    raise ValueError("`args` must be specified")
  if not isinstance(args, (list, tuple)):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  # Now the computation.
  with vs.variable_scope(scope or "Linear"):
    matrix = vs.get_variable("Matrix", [total_arg_size, output_size])
    if len(args) == 1:
      res = math_ops.matmul(args[0], matrix)
    else:
      res = math_ops.matmul(array_ops.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = vs.get_variable(
        "Bias", [output_size],
        initializer=init_ops.constant_initializer(bias_start))
  return res + bias_term

# highway layer that borrowed from https://github.com/carpedm20/lstm-char-cnn-tensorflow
def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
  """Highway Network (cf. http://arxiv.org/abs/1505.00387).

  t = sigmoid(Wy + b)
  z = t * g(Wy + b) + (1 - t) * y
  where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
  """
  output = input_
  for idx in xrange(layer_size):
    output = f(_linear(output, size, True, 0, scope='output_lin_%d' % idx)) #tf.nn.rnn_cell._linear

    transform_gate = tf.sigmoid(
      _linear(input_, size, True, 0, scope='transform_lin_%d' % idx) + bias)  # tf.nn.rnn_cell._linear
    carry_gate = 1. - transform_gate

    output = transform_gate * output + carry_gate * input_

  return output



# class TextCNN(object):
  """
  A CNN for text classification.
  Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
  """
def TextCNN(sequence_length, num_classes, vocab_size,embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
  # Placeholders for input, output and dropout
  input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
  input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
  dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

  # Keeping track of l2 regularization loss (optional)
  l2_loss = tf.constant(0.0)

  # Embedding layer
  with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
        name="W")
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

  # Create a convolution + maxpool layer for each filter size
  pooled_outputs = []
  for filter_size, num_filter in zip(filter_sizes, num_filters):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
      # Convolution Layer
      filter_shape = [filter_size, embedding_size, 1, num_filter]
      W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
      b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
      conv = tf.nn.conv2d(
        embedded_chars_expanded,
        W,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
      # Apply nonlinearity
      h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
      # Maxpooling over the outputs
      pooled = tf.nn.max_pool(
        h,
        ksize=[1, sequence_length - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="pool")
      pooled_outputs.append(pooled)

  # Combine all the pooled features
  num_filters_total = sum(num_filters)
#   print(pooled_outputs)
  h_pool = tf.concat(pooled_outputs, 3)
  h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

  # Add highway
  with tf.name_scope("highway"):
    h_highway = highway(h_pool_flat, h_pool_flat.get_shape()[1], 1, 0)

  # Add dropout
  with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_highway, dropout_keep_prob)

  # Final (unnormalized) scores and predictions
  with tf.name_scope("output"):
    W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
    l2_loss += tf.nn.l2_loss(W)
    l2_loss += tf.nn.l2_loss(b)
    scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")

  # CalculateMean cross-entropy loss
  with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y)
    loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

  # Accuracy
  with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
  
  return loss, accuracy, input_x, input_y, dropout_keep_prob


# ### 3. train

# In[ ]:

#! /usr/bin/env python

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import os
import time
import datetime
# import data_helpers
# from text_cnn import TextCNN

# Parameters
# ==================================================

flags = tf.app.flags
FLAGS = flags.FLAGS

# # Model Hyperparameters
# flags.DEFINE_integer("embedding_size", 200, "embedding size") 
# tf.app.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
# tf.flags.DEFINE_string("filter_sizes", "1,2,3,4,5,6,8", "Comma-separated filter sizes (default: '1,2,3,4,5,6,8')")
# tf.flags.DEFINE_string("num_filters", "50,100,150,150,200,200,200", "Number of filters per filter size (default: 50,100,150,150,200,200,200)")
# tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# # Training parameters
# tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
# tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
# tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# # Misc Parameters
# tf.flags.DEFINE_string("checkpoint", '', "Resume checkpoint")
# tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
# tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

embedding_size = 200
embedding_dim = 128
filter_sizes = "1,2,3,4,5,6,8"
num_filters = "50,100,150,150,200,200,200"
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0
batch_size = 32
num_epochs = 200
evaluate_every = 100
checkpoint_every = 100
checkpoint = ''
allow_soft_placement = True
log_device_placement = False


print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.iteritems()):
  print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = load_data()
# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]
# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-300], x_shuffled[-300:]
y_train, y_dev = y_shuffled[:-300], y_shuffled[-300:]
sequence_length = x_train.shape[1]
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
print("Sequnence Length: {:d}".format(sequence_length))


# Training
# ==================================================

with tf.Graph().as_default():
  session_conf = tf.ConfigProto(
    allow_soft_placement=FLAGS.allow_soft_placement,
    log_device_placement=FLAGS.log_device_placement)
  sess = tf.Session(config=session_conf)
  with sess.as_default():
    loss, accuracy, input_x, input_y, dropout_keep_prob = TextCNN(
      sequence_length=sequence_length,
      num_classes=2,
      vocab_size=len(vocabulary),
      embedding_size=FLAGS.embedding_dim,
      filter_sizes=map(int, FLAGS.filter_sizes.split(",")),
      num_filters=map(int, FLAGS.num_filters.split(",")),
      l2_reg_lambda=FLAGS.l2_reg_lambda)

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(loss, aggregation_method=2)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
      if g is not None:
        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    if FLAGS.checkpoint == "":
      timestamp = str(int(time.time()))
      out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
      print("Writing to {}\n".format(out_dir))
    else:
      out_dir = FLAGS.checkpoint

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", loss)
    acc_summary = tf.summary.scalar("accuracy", accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph_def)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.all_variables())

    # Initialize all variables
    sess.run(tf.initialize_all_variables())

    ckpt = tf.train.get_checkpoint_state(os.path.join( checkpoint, 'checkpoints'))
    if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
      print "Reading model parameters from %s" % ckpt.model_checkpoint_path
      saver.restore(sess, ckpt.model_checkpoint_path)

    def train_step(x_batch, y_batch, loss, accuracy):
      """
      A single training step
      """
      feed_dict = {
        input_x: x_batch,
        input_y: y_batch,
        dropout_keep_prob: FLAGS.dropout_keep_prob
      }
      _, step, summaries, loss, accuracy = sess.run(
        [train_op, global_step, train_summary_op, loss, accuracy],
        feed_dict)
      time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      train_summary_writer.add_summary(summaries, step)

    def dev_step(x_batch, y_batch,loss, accuracy, writer=None):
      """
      Evaluates model on a dev set
      """
      feed_dict = {
        input_x: x_batch,
        input_y: y_batch,
        dropout_keep_prob: 1.0
      }
      step, summaries, loss, accuracy = sess.run(
        [global_step, dev_summary_op, loss, accuracy],
        feed_dict)
      time_str = datetime.datetime.now().strftime("%d, %b %Y %H:%M:%S")
      print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
      if writer:
        writer.add_summary(summaries, step)

    # Generate batches
    batches = batch_iter(
      zip(x_train, y_train), batch_size,  num_epochs)
    # Training loop. For each batch...
    for batch in batches:
      x_batch, y_batch = zip(*batch)
      train_step(x_batch, y_batch, loss, accuracy)
      current_step = tf.train.global_step(sess, global_step)
      if current_step %  evaluate_every == 0:
        print("\nEvaluation:")
        dev_step(x_dev, y_dev, loss, accuracy, writer=dev_summary_writer)
        print("")
      if current_step %  checkpoint_every == 0:
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))


# In[ ]:

import tensorflow as tf
import numpy as np
import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS

# define flags (note that Fomoro will not pass any flags by default)
flags.DEFINE_boolean('skip-training', False, 'If true, skip training the model.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')
print(FLAGS.restore)


# In[ ]:



