#OCR with Keras, TensorFlow and Deep Learning
#Image input

#import necessary packages
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib as plt
from tf.keras.datasets import mnist

#load A-Z dataset
datasetPath = pd.read_csv(r'A_Z Handwritten Data.csv')
def load_az_dataset(datasetPath):
  data = []
  labels = []

  #loop over the rows of the A-Z dataset
  for row in open(datasetPath):
    #split label and image from the row
    row = row.split(",")
    label = int(row[0])
    image = np.array([int(x) for x in row[1:]], dtype="unit8")

    #reshaping images into 28X28 matrix
    image = image.reshape((28,28))

    #update list of data and labels
    data.append(image)
    labels.append(label)

    data = np.array(data, dtype="float32")
    labels = np.array(labels, dtype="int")

    return (data, labels)

#load MNIST handwritten digit dataset
def load_mnist_dataset():
  ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
  data = np.vstack([trainData, testData])
  labels = np.hstack([trainLabels, testLabels])

  return (data, labels)



