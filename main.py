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

#convert labels to binary class matrix
labels = tf.keras.utils.to_categorical(labels, num_classes=None, dtype='float32')

data = data.reshape(-1, 28, 28, 1)

#spliting 80% of the data for training and 20% for testing
(trainX, testX, trainy, testy) = train_test_split(data, labels, test_size=0.20, random_state=42)

#constructing the image data generator for augmentation
aug = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    fill_mode="nearest"
)

#The number of epochs to train for and batch size
Epochs = 10
BS = 128

#compile our deep neural network
print("[INFO] compiling model..")

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (5,5), activation=tf.nn.relu, input_shape=trainX.shape[1:]))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(64, (3,3), kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu, padding='same'))
model.add(tf.keras.layers.Conv2D(128, (3,3), kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu, padding='valid'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.001), activation=tf.nn.relu))

model.add(tf.keras.layers.Dropout(0.25))

model.add(tf.keras.layers.Dense(266, activation=tf.nn.softmax))
model.summary()

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=["accuracy"])
h = model.fit(aug.flow(trainX, trainy, batch_size=BS), 
          validation_data=(testX, testy), 
          steps_per_epoch=len(trainX)//BS,
          epochs=Epochs,
          verbose=1)

# evaluate the network and print the accuracy and loss of the model
score = model.evaluate(testX, testy, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

#define the list of label names
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(np.argmax(testy, axis=1),
  np.argmax(predictions, axis=1), target_names=None))
# save the model to disk
model.save("model.h5")
img = cv2.imread(r' image_name')
img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (400,440))

img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
_, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

img_final = cv2.resize(img_thresh, (28,28))
img_final =np.reshape(img_final, (1,28,28,1))

img_pred = labelNames[np.argmax(model.predict(img_final))]
# plt.imshow(img, cmap='gray')

from google.colab.patches import cv2_imshow
cv2.putText(img, " _ _ _ ", (20,25), cv2.FONT_HERSHEY_TRIPLEX, 0.7, color = (0,0,230))
cv2.putText(img, "Prediction: " + img_pred, (20,410), cv2.FONT_HERSHEY_DUPLEX, 1.3, color = (255,0,30))
cv2_imshow(img)
