
# coding: utf-8

# In[2]:

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
  positive_labels = [0 for _ in positive_examples]
  negative_labels = [1 for _ in negative_examples]
  y = np.concatenate([positive_labels, negative_labels], 0)
    
  return [x_text, y]

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
  x = list([[vocabulary[word] for word in sentence] for sentence in sentences])
  y = list(labels)
  return [x, y]


def load_data():
  """
  Loads and preprocessed data for the MR dataset.
  Returns input vectors, labels, vocabulary, and inverse vocabulary.
  """
  # Load and preprocess data
  sentences, labels = load_data_and_labels()
#   sentences_padded = pad_sentences(sentences)
  vocabulary, vocabulary_inv = build_vocab(sentences)
  x, y = build_input_data(sentences, labels, vocabulary)
  return [x, y, vocabulary, vocabulary_inv]


# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x, y, vocabulary, vocabulary_inv = load_data()


# In[7]:

testNum = 1000

trainX = x[testNum:-testNum]
testX = x[0:testNum] + x[-testNum:]
trainY = y[testNum:-testNum]
testY = y[0:testNum] + y[-testNum:]

print(len(trainX), len(trainY), len(testX), len(testY))

# print(type(x), type(y), type(vocabulary), type(vocabulary_inv))
# print(testY)
# print(len(x), len(y))


# In[6]:

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

tf.reset_default_graph()

## Data preprocessing
## Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
## Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)
# print(trainY, testY)

# Building convolutional network
network = input_data(shape=[None, 100], name='input')
network = tflearn.embedding(network, input_dim=10000, output_dim=128)
branch1 = conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
branch2 = conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
branch3 = conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
network = merge([branch1, branch2, branch3], mode='concat', axis=1)
network = tf.expand_dims(network, 2)
network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch = 10, validation_set=(testX, testY), shuffle=True, show_metric=True, batch_size=32)  


# In[8]:

result = model.predict(testX)
print( testY[:3], result[:3])
print( testY[-3:], result[-3:])


# In[ ]:



