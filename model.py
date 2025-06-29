from keras.models import Sequential
from keras import Input
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D

def create_model():
    model = Sequential()
    model.add(Input(shape=(28,28,1)))
    model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(47, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    
    return model