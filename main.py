#OCR with Keras, TensorFlow and Deep Learning

#import necessary packages
import numpy as np
import cv2
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report

mnist = tf.keras.datasets.mnist

# #load A-Z dataset

def load_az_dataset(datasetPath):
  data = []
  labels = []
  import math
  dataset = pd.read_csv(datasetPath)

  #loop over the rows of the A-Z dataset
  for row in dataset:
      #split label and image from the row
      row = row.split(",")
      
      label = math.floor(float(row[0]))
      image = np.array([int(x) for x in row[1:]], dtype="uint8")

      #reshaping images into 28X28 matrix
      image = np.arange(784).reshape(28,28)
      
      #update list of data and labels
      data.append(image)
      labels.append(label)

  data = np.array(data, dtype="float32")
  labels = np.array(labels, dtype=None)  

  return (data, labels)

 # load MNIST handwritten digit dataset
 def load_mnist_dataset():
    (trainData, trainLabels), (testData, testLabels) = mnist.load_data()
    trainData = tf.keras.utils.normalize(trainData, axis=1)
    testData = tf.keras.utils.normalize(testData, axis=1)
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])

    return (data, labels)
#loading both datasets and combining them into one
azPath = "A_Z Handwritten Data.csv"
(azData, azLabels) = load_az_dataset(azPath)
(digitData, digitLabels) = load_mnist_dataset()

# as MNIST has labels 0-9, we are going to add 10 to ever A-Z labels so that they are not labeled as digits
azLabels += 10

data = np.vstack([azData, digitData])
labels = np.hstack([azLabels, digitLabels])

data = [cv2.resize(image, (28, 28)) for image in data]
data = np.array(data, dtype="float32")

#add a channel dimension to every image in the dataset and normalize them
data = np.expand_dims(data, axis=-1)
data /= 255.0
