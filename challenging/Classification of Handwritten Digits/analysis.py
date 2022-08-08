import tensorflow as tf
import numpy as np


def main():
    """
    Objective Stage 1:
    - 1 Import tensorflow and numpy
    - 2 Load MNIST data from tensorflow.keras.datasets
    - 3 Reshape the features array to the 2D array with n rows and m columns
      (n: number of images in the dataset; m: number of pixels in each image)
      --> flatten the features and target arrays
    - 4 Provide the unique target classes' names, the shape of the features array
      and the shape of the target variable in following format: (n, m).
      Print the min and max values of the features array.
    """
    # 2 - Load MNIST datasets from tensorflow.keras.datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # 3 - Flatten features array with n images and m pixels/image
    n = np.shape(x_train)[0]
    m = np.shape(x_train)[1] * np.shape(x_train)[2]
    x_train = x_train.reshape(n, m)
    # 4 - Print unique target classes' names, features and target shapes,
    # and min & max values of the features array
    print("Classes:", np.unique(y_train))
    print("Features' shape:", np.shape(x_train))
    print("Target's shape:", np.shape(y_train))
    print(f"min: {x_train.min()}, max: {x_train.max()}")


if __name__ == '__main__':
    main()
