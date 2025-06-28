import matplotlib.pyplot as plt
from data_preprocessing import load_data, map_labels
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def show_images(data, plot_title, char_labels=None):
    samples = list(data.take(9))
    plt.figure(figsize=(8,5))
    for i, (image, label) in enumerate(samples):
        # for i in range(9):
        # img = images[i]
        img = tf.squeeze(image)
        label = label.numpy()
        char = char_labels[label] if char_labels else str(label)
        # if char_labels is not None:
        #     char = char_labels[label]
        #     title = f"{char}"
        # else:
        #     title = f"Label: {label}"
        plt.subplot(3,3,i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{char}")
        plt.axis('off')
    plt.suptitle(plot_title)
    plt.tight_layout()
    plt.show()