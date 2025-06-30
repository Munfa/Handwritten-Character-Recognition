from keras.models import Sequential
from keras import Input
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D

'''
    Input = defines the shape of intput images. tells the model what to expect
    Conv2D = scans the image to extract patterns. 
    MaxPooling2D = shrinks the image while keeping important features. (2,2) means height and width cut in half
    Dropout = randomly turns off some neurons to prevent model from memorizing the training data. 0.3 means 30% neurons turned off during each training step
    Flatten = turns 2D feature map into a 1D vector for Dense layers
    Dense = fully connected layer. each input is connected to each output neuron. 'softmax' turns outputs into the probabilities of classes
    BatchNormalization = makes sure the outputs of each layer stay steady while the model learns. makes training faster
'''
def create_model():
    model = Sequential()
    model.add(Input(shape=(28,28,1)))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())

    #### hidden Dense layers. they decide how to use features extracted by Conv layers to classify
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    #### output layer
    model.add(Dense(47, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    
    return model