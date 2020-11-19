#OCR with Keras, TensorFlow and Deep Learning
#Image input

#import necessary packages
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf

#train model
mnist = tf.keras.datasets.mnist
(trainData, trainLabels),(testData,testLabels) = mnist.load_data()

trainData = tf.keras.utils.normalize(trainData, axis=1)
testData = tf.keras.utils.normalize(testData, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(trainData,trainLabels, epochs=1)

loss, accuracy = model.evaluate(testData, testLabels)
print(accuracy)
print(loss)

model.save('digits.model')

#predict image
for x in range(1,6):
  img = cv.imread(f'{x}.png')[:,:,0]
  img = np.invert(np.array([img]))
  prediction = model.predict(img)
  print(f'The result is probably: {np.argmax(prediction)}')
  plt.imshow(img[0], cmap= plt.cm.binary)
  plt.show()
