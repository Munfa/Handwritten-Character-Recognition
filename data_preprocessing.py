import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow_datasets as tfds
import tensorflow as tf
 
def fix_and_normalize(image, label):
    image = tf.transpose(image, perm=[1,0,2]) # fix rotation
    image = tf.cast(image, tf.float32)/255.0

    return image, label

def map_labels(train, test):
  mapping = np.loadtxt("emnist-balanced-mapping.txt", dtype=int, usecols=1, unpack=True)
  char_labels = {}
  for _, label in train:
    label = label.numpy().item()
    char_labels[label] = chr(int(mapping[label]))
  for _, label in test:
    label = label.numpy().item()
    char_labels[label] = chr(int(mapping[label]))

  return char_labels

def load_data():
  (ds_train, ds_test), ds_info = tfds.load(
      'emnist/balanced',
      split=['train', 'test'],
      # batch_size=batch,
      shuffle_files=True,
      as_supervised=True,
      with_info=True,
      data_dir='E:/ML tutorials/ML Projects/Handwritten Character Recognition/',
      download=True
  )
  return ds_train, ds_test, ds_info

def preprocess(ds_train, ds_test, ds_info):
  ds_train = ds_train.map(fix_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
  ds_train = ds_train.cache()
  ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
  ds_train = ds_train.batch(128)
  ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

  ds_test = ds_test.map(fix_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
  ds_test = ds_test.batch(128)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

  return ds_train, ds_test