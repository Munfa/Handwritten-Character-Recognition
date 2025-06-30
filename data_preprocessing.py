import numpy as np

#### setting the environment to avoid getting warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow_datasets as tfds
import tensorflow as tf
 
def fix_and_normalize(image, label):
    image = tf.transpose(image, perm=[1,0,2]) # fix rotation [W, H, C] -> [H, W, C]
    image = tf.cast(image, tf.float32)/255.0 # normalize pixel values to [0,1]

    return image, label

def map_labels(train, test):
  mapping = np.loadtxt("emnist-balanced-mapping.txt", dtype=int, usecols=1, unpack=True)
  char_labels = {}
  for _, label in train:
    label = label.numpy().item() # turn it into a Python int
    char_labels[label] = chr(int(mapping[label])) # using ASCII value to convert to char
  for _, label in test:
    label = label.numpy().item()
    char_labels[label] = chr(int(mapping[label]))

  return char_labels

def load_data():
  (ds_train, ds_test), ds_info = tfds.load(
      'emnist/balanced',
      split=['train', 'test'],
      shuffle_files=True,
      as_supervised=True,
      with_info=True,
      data_dir='Your project directory',
      download=True
  )
  return ds_train, ds_test, ds_info

'''
  Preprocessing images using tensorflow pipelines to get faster and smoother training
  map -> every image is normalized and transposed; AUTOTUNE used to optimize pipeline execution
  cache -> to avoid I/O bottlenecks
'''
def preprocess(ds_train, ds_test, ds_info):
  ds_train = ds_train.map(fix_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
  ds_train = ds_train.cache()                                       # saves the dataset in memory after first read
  ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples) # shuffles the training dataset randomly
  ds_train = ds_train.batch(128)                                    # groups images into one batch
  ds_train = ds_train.prefetch(tf.data.AUTOTUNE)                    # loads data in parallel with training

  ds_test = ds_test.map(fix_and_normalize, num_parallel_calls=tf.data.AUTOTUNE)
  ds_test = ds_test.batch(128)
  ds_test = ds_test.cache()
  ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

  return ds_train, ds_test