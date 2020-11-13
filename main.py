{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Untitled0.ipynb",
      "provenance": [],
      "authorship_tag": "ABX9TyOmoUJfZkBFCy6dgc+UJwx5",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Munfa/HCR/blob/main/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Oaixvva9EWpb"
      },
      "source": [
        "#OCR with Keras, TensorFlow and Deep Learning\n",
        "#Image input"
      ],
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "y62XffrcEYwd"
      },
      "source": [
        "#import necessary packages\n",
        "from tensorflow.keras.datasets import mnist\n",
        "import numpy as np\n"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "4CPBg4BKbbc5"
      },
      "source": [
        "def load_mnist_dataset():\n",
        "    ((trainData, trainLabels),(testData,testLabels)) = mnist.load_data()\n",
        "    data = np.vstack([trainData,testData])\n",
        "    labels = np.hstack([trainLabels,testLabels])\n",
        "\n",
        "    return (data,labels)\n",
        "    "
      ],
      "execution_count": 3,
      "outputs": []
    }
  ]
}