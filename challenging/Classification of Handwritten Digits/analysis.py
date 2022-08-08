import pandas as pd
import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split


def main():
    """
    Objective Stage 2:
    - 1 Import necessary tool from sklearn to split datasets
    - 2 Use the first 6000 rows of the dataset.
      Set the test_set_size = 0.3 & random_seed = 40
    - 3 Print new data shapes and the proportions of samples per class in the training
      set
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
    # Model settings
    rows_limit = 6000
    seed = 40
    test_size = 0.3  # 30% of dataset

    # Load MNIST datasets from tensorflow.keras.datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    X = np.concatenate((x_train, x_test))  # combined features array
    Y = np.concatenate((y_train, y_test))  # combined target array

    # Flatten features array with n images and m pixels/image
    n = np.shape(X)[0]
    m = np.shape(X)[1] * np.shape(X)[2]
    X = X.reshape(n, m)

    # 2a - Use the first 6000 rows of the dataset
    # 2b - Set the test_size = 0.3 & random_seed 40
    X = X[:rows_limit]
    Y = Y[:rows_limit]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    # 3 - Print shapes of new data and the proportions of sample per class in the training set
    # print("Classes:", np.unique(y_train))
    print("x_train shape:", np.shape(X_train))
    print("x_test shape:", np.shape(X_test))
    print("y_train shape:", np.shape(Y_train))
    print("y_test shape:", np.shape(Y_test))
    print("Proportion of samples per class in train set:")
    print(pd.Series(Y_train).value_counts(normalize=True))


if __name__ == '__main__':
    main()
