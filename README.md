# EMNIST Handwritten Character Recognition Using CNN

### Project Overview
This project builds a Convolutional Neural Network(CNN) to recognize handwritten letters and digits using the EMNIST Balanced dataset. The model is trained on grayscale 28x28 images and can predict handwritten characters, including custom images drawn using online tools. The project demonstrates a complete ML workflow: data loading, preprocessing, training, evaluation and prediction. 

### Install Dependencies
Install tensorflow, tensorflow_datasets, keras, and pillow
<pre>
  pip install tensorflow tensorflow_dataset keras pillow
</pre>

### Dataset Used
<ul>
  <li><b>Name</b>: EMNIST Balanced</li>
  <li><b>Source</b>: Tensorflow datasets</li>
  <li><b>Classes</b>: 47 (digits + uppercase/lowercase letters)</li>
</ul>
Get the dataset in the same directory as your project<br>
<pre>
  (ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
    data_dir='Your project directory',
    download=True
  )
</pre>

### Data Preprocessing
The EMNIST dataset is preprocessed using a TensorFlow data pipeline to ensure efficient training and accurate input formatting:
<br>
<ul>
  <b>Rotation</b>: Images are transposed and rotated to match EMNIST's character orientation<br>
  <b>Normalization</b>: Pixel values are scaled to [0, 1]<br>
  <b>Batching</b>: Dataset is batched (e.g., batch_size=128)<br>
  <b>Caching</b>: Training data is cached to avoid I/O bottlenecks<br>
  <b>Shuffling</b>: The dataset is shuffled to prevent learning order-based patterns<br>
  <b>Prefetching</b>: 'tf.data.AUTOTUNE' is used to optimize pipeline execution<br>
  <b>Mapping</b>: Mapping character labels to the images
</ul>

### Features
<ul>
  <li>Custom CNN model built with Keras (Conv2D, MaxPooling, Dense, Dropout)</li>
  <li>Visual output with matplotlib before and after preprocessing dataset images</li>
  <li>Trained and evaluated on EMNIST dataset and saved the best model</li>
  <li>Preprocessing pipeline with centering, padding and resizing</li>
  <li>Predicting custom handwritten images (drawn via Kleki)</li>
  <li>Visualize downloaded image with predicted labels</li>
</ul>

### Challenges & Lessons Learned
<ul>
  <li>Learned to manage image preprocessing, including padding and center alignment</li>
  <li>Understood the importance of image format consistency (channel dims, grascale)</li>
  <li>Faced a model prediction issue that turned out to be due to forgetting to load the trained model</li>
  <li>Learned how to visualize and debug predictions and improve model input quality</li>
</ul>

### Future Work
<ul>
  <li>Add support for drawing in-app and real-time prediction</li>
  <li>Try transfer learning using pretrained vision models</li>
  <li>Improve generalization with fine-tuning on external data</li>
</ul>

### References
<ul>
  <li>TensorFlow Example: [https://www.tensorflow.org/datasets/keras_example] </li>
  <li>Kaggle Notebook: [https://www.kaggle.com/code/atamish/emnist-cnn#CNN-Model]</li>
  <li>ChatGpt - Used for visualization, debugging code, and understanding concepts clearly</li>
</ul>





