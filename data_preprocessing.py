import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow_datasets as tfds
import tensorflow as tf
 
def preprocess(image, label):
    image = tf.expand_dims(image, axis=-1)
    image = tf.image.rot90(image, k=1) # fix rotation
    image = tf.image.flip_left_right(image) # fix mirroring
    image = tf.cast(image, tf.float32)/255.0

    return image, label

#load dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    data_dir='E:/ML tutorials/ML Projects/Handwritten Character Recognition/',
    download=True
)

ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


