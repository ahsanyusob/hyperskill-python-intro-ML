import pandas as pd
import tensorflow as tf
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# 2
def fit_predict_eval(dictionary, model, features_train, features_test, target_train, target_test, print_val):
    # here you fit the model
    model.fit(features_train, target_train)
    # make a prediction
    predicted_target = model.predict(features_test)
    # calculate accuracy and save it to score
    score = accuracy_score(target_test, predicted_target)
    if print_val:
        print(f'Model: {model}\nAccuracy: {score: .4f}\n')
    dictionary.update({type(model).__name__: score})
    return dictionary


def main():
    """
    Objective Stage 4:
    - 1 Import sklearn.preprocessing.Normalizer - done
    - 2 Initialize the normalizer, transform the features and save the output to features_norm
    - 3 Repeat steps 2-4 from stage 3
    - 4 Answer the following questions:
      - Does the normalization have a positive impact in general?
      - Which two models show the best scores? Round the result to the 3rd d.p.

    Objective Stage 3:
    - 1 Import sklearn implementations of the classifiers and the accuracy scorer for:
      (KNN, dt, logistic regression, Random Forest)
    - 2 Implement separate function to make it easier for training a lot of models as follows:
      def fit_predict_eval(model, features_train, features_test, target_train, target_test):
            # fit the model
            # make prediction
            # calculate accuracy and save to score

      example of function implementation:
      --> fit_predict_eval(
                            model=KNeighborsClassifier(),
                            features_train=x_train,
                            features_test=x_test,
                            target_train=y_train,
                            target_test=y_test
                          )
    - 3 Initialize the models and set random_state=40 when needed
    - 4 Fit the models
    - 5 Make predictions and print the accuracies in the following order:
      (KNN, dt, logistic regression, random forest)
    - 6 Determine which model performs the best.

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
    # Initialization
    result_dict = dict()
    result_dict_norm = dict()
    q1_yes_no_list = []

    # Model settings
    rows_limit = 6000
    seed = 40

    # Load MNIST datasets from tensorflow.keras.datasets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    X = np.concatenate((x_train, x_test))  # combined features array
    Y = np.concatenate((y_train, y_test))  # combined target array

    # Flatten features array with n images and m pixels/image
    n = np.shape(X)[0]
    m = np.shape(X)[1] * np.shape(X)[2]
    X = X.reshape(n, m)

    # Use the first 6000 rows of the dataset
    # Set the test_size = 0.3 & random_seed 40
    X = X[:rows_limit]
    Y = Y[:rows_limit]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

    # 2 - Initialize the normalizer and transform the features
    X_train_norm = Normalizer().transform(X_train)
    X_test_norm = Normalizer().transform(X_test)

    # Print shapes of new data and the proportions of sample per class in the training set
    # print("Classes:", np.unique(y_train))
    # print("x_train shape:", np.shape(X_train))
    # print("x_test shape:", np.shape(X_test))
    # print("y_train shape:", np.shape(Y_train))
    # print("y_test shape:", np.shape(Y_test))
    # print("Proportion of samples per class in train set:")
    # print(pd.Series(Y_train).value_counts(normalize=True))

    # 3 - Repeat steps 2-4 from stage 3
    # ii - Initialize the models
    KNN_model = KNeighborsClassifier()
    dt_model = DecisionTreeClassifier(random_state=seed)
    lr_model = LogisticRegression(random_state=seed, solver="liblinear")
    RF_model = RandomForestClassifier(random_state=seed)
    models = (KNN_model, dt_model, lr_model, RF_model)

    # iii-a, iv-a - Fit the models, make predictions and print the accuracies
    for model in models:
        result_dict = fit_predict_eval(
                result_dict,
                model=model,
                features_train=X_train,
                features_test=X_test,
                target_train=Y_train,
                target_test=Y_test,
                print_val=False
        )

    # iii-b, iv-b - Fit the models, make predictions and print the accuracies
    for model in models:
        result_dict_norm = fit_predict_eval(
            result_dict_norm,
            model=model,
            features_train=X_train_norm,
            features_test=X_test_norm,
            target_train=Y_train,
            target_test=Y_test,
            print_val=True
        )

    # 4a - Answer to Q1: "Does the normalization have a positive impact in general?"
    for model, result_norm in result_dict_norm.items():
        result = result_dict[model]
        q1_yes_no_list.append("Y") if result_norm > result else q1_yes_no_list.append("N")
    print(f"The answer to the 1st question:", "yes" if q1_yes_no_list.count('Y') >= q1_yes_no_list.count('N') else "no")

    # 4b - Answer to Q2: "Which two models show the best scores? Round the result to the 3rd d.p."
    best_model, best_model_score = '', 0
    second_best_model, second_best_model_score = '', 0
    for model, result in result_dict_norm.items():
        if result > best_model_score and result > second_best_model_score:
            best_model_score = result
            best_model = model
        elif second_best_model_score < result < best_model_score:
            second_best_model_score = result
            second_best_model = model

    print(f"The answer to the 2nd question: {best_model}-{best_model_score:.3f}, "
          f"{second_best_model}-{second_best_model_score:.3f}")


if __name__ == '__main__':
    main()
