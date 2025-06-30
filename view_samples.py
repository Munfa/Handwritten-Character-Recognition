import matplotlib.pyplot as plt

#### setting the environment to avoid getting warnings
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def show_images(data, plot_title, char_labels=None):
    samples = list(data.take(9))    # load images in a list 
    plt.figure(figsize=(8,5))
    for i, (image, label) in enumerate(samples):
        img = tf.squeeze(image)     # turn (1,28,28,1) to (28,28) -> easier to visualize
        label = label.numpy()       # converts it to scalar
        char = char_labels[label] if char_labels else str(label)
        plt.subplot(3,3,i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"{char}")
        plt.axis('off')
    plt.suptitle(plot_title)
    plt.tight_layout()
    plt.show()