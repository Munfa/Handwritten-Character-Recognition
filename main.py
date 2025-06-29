'''
    creating separate files for cleaner code 
    combining all functions in this main.py file
    importing the functions from other files
'''
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras
from data_preprocessing import load_data, map_labels, preprocess
from view_samples import show_images
from model import create_model
from predict_and_plot import plot_training_curves, show_preds

#### load the dataset
train, test, info = load_data()

#### view sample images before preprocessing 
show_images(train, "Before Preprocessing") 

#### map the character labels
char_labels = map_labels(train, test)

#### preprocess the data
pre_train, pre_test = preprocess(train, test, info)

#### images after preprocessing
show_images(pre_train.unbatch(), "After Preprocessing", char_labels)

model = create_model()

checkpoint = keras.callbacks.ModelCheckpoint(
    "best_model.keras", save_best_only=True, monitor="val_loss"
)
history = model.fit(
    pre_train,
    epochs=5,
    validation_data=pre_test,
    callbacks=checkpoint
)
test_loss, test_acc = model.evaluate(pre_test)
print("Test Accuracy: ", test_acc)             

plot_training_curves(history)

model = keras.models.load_model("best_model.keras")
folder_path = "images"
image_paths = [os.path.join(folder_path, f_name) for f_name in os.listdir(folder_path)
               if f_name.lower().endswith((".png", ".jpg", ".jpeg"))]
show_preds(model, image_paths, char_labels)
