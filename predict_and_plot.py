import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

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
    
def preprocess_img(img_path):
    img = Image.open(img_path).convert("L") # make sure it's grayscale; L = grayscale
    img = img.point(lambda x:255 if x>30 else 0, 'L')

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    max_side = max(img.size)
    padding = 20
    padded_img = Image.new("L", (max_side+padding, max_side+padding), 0)
    padded_img.paste(img, ((padded_img.size[0] - img.size[0])//2, (padded_img.size[1] - img.size[1])//2))

    final_img = padded_img.resize((28,28), Image.Resampling.LANCZOS)
    final_img = np.array(final_img).astype("float32")/255.0
    img_tensor = tf.expand_dims(final_img, axis=-1)
    img_tensor = tf.expand_dims(final_img, axis=0)

    return img_tensor, final_img

def show_preds(model, img_paths, char_labels):
    plt.figure(figsize=(8,5))

    for i, img_path in enumerate(img_paths):
        img_tensor, final_img = preprocess_img(img_path)
        pred = model.predict(img_tensor)
        pred_label = np.argmax(pred)
        pred_char = char_labels[pred_label]

        plt.subplot(3, 3, i+1)
        plt.imshow(tf.squeeze(final_img), cmap="gray")
        plt.title(f"Predicted: {pred_char}")
        plt.axis("off")

    plt.suptitle("Model Predictions on Downloaded Images")
    plt.tight_layout()
    plt.show()