#OCR with Keras, TensorFlow and Deep Learning
#Image input

#import necessary packages
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

#load A-Z dataset
def load_az_dataset(datasetPath):
  data = []
  labels = []

  #loop over the rows of the A-Z dataset
  for row in open(datasetPath):
    #split label and image from the row
    row = row.split(",")
    label = int(row[0])
    image = np.array([int(x) for x in row[1:]], dtype=np.uint8)

    #reshaping images into 28X28 matrix
    image = image.reshape((28,28))

    #update list of data and labels
    data.append(image)
    labels.append(label)

  data = np.array(data, dtype="float32")
  labels = np.array(labels, dtype="int")

  return (data, labels)

 # load MNIST handwritten digit dataset
 def load_mnist_dataset():
    (trainData, trainLabels), (testData, testLabels) = mnist.load_data()
    trainData = tf.keras.utils.normalize(trainData, axis=1)
    testData = tf.keras.utils.normalize(testData, axis=1)
    data = np.vstack([trainData, testData])
    labels = np.hstack([trainLabels, testLabels])

    return (data, labels)

# loading both datasets and combining them into one
azPath = "A_Z Handwritten Data.csv"
(azData, azLabels) = load_az_dataset(azPath)
(digitData, digitLabels) = load_mnist_dataset()

# as MNIST has labels 0-9, we are going to add 10 to ever A-Z labels so that they are not labeled as digits
azLabels += 10

data = np.vstack([azData, digitData])
labels = np.hstack([azLabels, digitLabels])
