import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#### setting the environment to avoid getting warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

#### getting accuracy and loss from history and plotting the curves
def plot_training_curves(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc)+1)

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ax.plot(epochs, acc, label="Accuracy")
    ax.plot(epochs, loss, label="Loss")

    plt.xlabel("Epochs")
    plt.title("Accuracy vs Loss")
    plt.legend(loc='upper right')
    plt.show()

'''
    open = opening the image. make sure it's grayscale; L = grayscale
    point = changes every pixel value to white(255) or black(0). greater than 30 becomes white and less than is black
    getbbox = return [left, upper, right, lower] coordinates of just the box around the character
    crop = crops image to remove unnecessary black space and zoom in on the character
    max = returns the larger value between width and height
    new = creating new grayscale balck canvas
    paste = pastes image on the black canvas at the specified position (this (//) ensures the result is integer not float)
    resize = resize without making character blurry or distorted. high-quality resampling method -> resampling.LANCZOS
'''    
#### preprocessing downloaded images for prediction     
def preprocess_img(img_path):
    img = Image.open(img_path).convert("L") 
    img = img.point(lambda x:255 if x>30 else 0, 'L') 

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox) 

    max_side = max(img.size) 
    padding = 20
    padded_img = Image.new("L", (max_side+padding, max_side+padding), 0) 
    padded_img.paste(img, ((padded_img.size[0] - img.size[0])//2, (padded_img.size[1] - img.size[1])//2)) 

    final_img = padded_img.resize((28,28), Image.Resampling.LANCZOS)
    final_img = np.array(final_img).astype("float32")/255.0 # normalizing and turning PIL image to numpy array to expand dimensions 
    img_tensor = tf.expand_dims(final_img, axis=-1) # add batch (1, 28, 28)
    img_tensor = tf.expand_dims(final_img, axis=0) # add channel (1, 28, 28, 1) -> the shape the model expects

    return img_tensor, final_img

def show_preds(model, img_paths, char_labels):
    plt.figure(figsize=(8,5))

    for i, img_path in enumerate(img_paths):
        img_tensor, final_img = preprocess_img(img_path)
        pred = model.predict(img_tensor)    # returns probabilities of classes
        pred_label = np.argmax(pred)    # gets the most probability
        pred_char = char_labels[pred_label] # finds the label 

        plt.subplot(3, 3, i+1)
        plt.imshow(tf.squeeze(final_img), cmap="gray") # turn (1,28,28,1) to (28,28) -> easier to visualize
        plt.title(f"Predicted: {pred_char}")
        plt.axis("off")

    plt.suptitle("Model Predictions on Downloaded Images")
    plt.tight_layout()
    plt.show()