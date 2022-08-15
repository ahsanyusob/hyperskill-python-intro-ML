import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def fit_predict_eval(dictionary, model, features_train, features_test, target_train, target_test, print_val):
    # here you fit the model
    model.fit(features_train, target_train)
    # make a prediction
    predicted_target = model.predict(features_test)
    # calculate accuracy and save it to score
    score = accuracy_score(target_test, predicted_target)
    if print_val:
        print(f'Model: {model}\nAccuracy: {score: .4f}\n')
    dictionary.update({model: score})
    return dictionary


def main():
    """
    #### Objective Stage 5:
    - 1 - Choose two best models from previous stage
    - 2 - Initialize GridSearchCV(estimator=..., param_grid=..., scoring='accuracy', n_jobs=-1)
      - For KNN: n_neighbors = [3, 4], weights = ['uniform', 'distance'], algorithm = ['auto', 'brute']
      - For RF: n_estimators = [300, 500], max_features = ['auto', 'log2'],
        class_weight = ['balanced', 'balanced_subsample']
    - 3 - Run the fit method of GridSearchCV to find the best estimator
    - 4 - Print the best sets of parameters for both algorithms and their accuracies.
    """
    # Initialization
    result_dict = dict()
    
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

    # Initialize the normalizer and transform the features
    X_train_norm = Normalizer().transform(X_train)
    X_test_norm = Normalizer().transform(X_test)

    # (Stage 5)
    # 1 - KNN and RF Classifier models with normalized features give the highest two accuracies
    # 2 - Initialize GridSearchCV() to search over the following parameters:
    #     n_neighbours, weights, algorithm for KNN-Classifier
    #     n_estimators, max_features, class_weight for RF-Classifier
    knn_estimator = KNeighborsClassifier()
    knn_param_grid = dict(n_neighbors=[3, 4], weights=['uniform', 'distance'],
                          algorithm=['auto', 'brute'])
    knn_grid_search = GridSearchCV(estimator=knn_estimator, param_grid=knn_param_grid,
                                   scoring='accuracy', n_jobs=-1)

    rf_estimator = RandomForestClassifier(random_state=seed)
    rf_param_grid = dict(n_estimators=[300, 500], max_features=['auto', 'log2'],
                         class_weight=['balanced', 'balanced_subsample'])
    rf_grid_search = GridSearchCV(estimator=rf_estimator, param_grid=rf_param_grid,
                                  scoring='accuracy', n_jobs=-1)

    # 3 - Run the fit method for GridSearchCV (use train set only)
    knn_grid_search.fit(X_train_norm, Y_train)
    knn_best_estimator = knn_grid_search.best_estimator_
    rf_grid_search.fit(X_train_norm, Y_train)
    rf_best_estimator = rf_grid_search.best_estimator_
    models = (knn_best_estimator, rf_best_estimator)

    # 4 - Print the best sets of parameters for both algorithms.
    #     (Get the info from attribute called best_estimator_ of each algorithm)
    #     Train two best estimators on the test set and print their accuracies.
    for model in models:
        result_dict = fit_predict_eval(
                    result_dict,
                    model=model,
                    features_train=X_train_norm,
                    features_test=X_test_norm,
                    target_train=Y_train,
                    target_test=Y_test,
                    print_val=False
            )
    print("K-nearest neighbours algorithm")
    print("best estimator:", knn_best_estimator)
    print(f"accuracy: {result_dict[knn_best_estimator]: .3f}\n")
    print("Random forest algorithm")
    print("best estimator:", rf_best_estimator)
    print(f"accuracy: {result_dict[rf_best_estimator]: .3f}\n")


if __name__ == '__main__':
    main()
