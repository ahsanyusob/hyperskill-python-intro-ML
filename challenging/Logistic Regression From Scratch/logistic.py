import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = np.array([[]])

    def sigmoid(self, t):
        return [1 / (1 + math.exp(-t[i])) for i in range(len(t))]

    def predict_proba(self, row, coef_):
        t = row @ coef_
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        n_row = np.shape(X_train)[0]
        if self.fit_intercept:
            try:
                X_train.insert(0, 'ones', np.ones((n_row, 1)), False)  # allow_duplicates = False
            except ValueError:
                pass
        n_coefficient = np.shape(X_train)[1]
        self.coef_ = np.zeros((n_coefficient, 1))  # initialized weights and/or bias 1 x (n_coefficient)
        X_train = np.array(X_train)  # (n_row) x (n_feature)
        y_train = np.array([y_train])  # (n_row) x (n_target)

        for iter_ in range(1, self.n_epoch + 1):
            # if iter_ % 100 == 0:
            #     print(f"\nIteration: {iter_}\n------------\n")
            #     print(f"\nX_train: {X_train}\n")
            #     print(f"\nCoefficients: {self.coef_}\n")
            for i, row in enumerate(X_train):
                row = np.array([row])  # 1 x (n_feature)
                y_hat = self.predict_proba(row, self.coef_)  # 1x1
                y_hat = np.array([y_hat])
                # update all weights
                self.coef_ = self.coef_ - self.l_rate * (y_hat - y_train[0, i]) * y_hat * (1 - y_hat) * row.T

    def predict(self, X_test, cut_off=0.5):
        n_row = np.shape(X_test)[0]
        predictions = [0 for _ in range(n_row)]
        if self.fit_intercept:
            try:
                X_test.insert(0, 'ones', np.ones((n_row, 1)), False)  # allow_duplicates = False
            except ValueError:
                pass
        X_test = np.array(X_test)
        for i, row in enumerate(X_test):
            y_hat = self.predict_proba(row, self.coef_)
            y_hat = 1 if y_hat[0] >= cut_off else 0
            predictions[i] = y_hat
        return predictions  # predictions are binary values - 0 or 1


def standardize_z(x):
    """
    Take a nx1-vector, then perform z-standardization using:
    "z[i] = (x[i] - mu) / sigma" where:

    - z[i]  --> i-th sample standard score
    - x[i]  --> i-th sample value
    - mu    --> mean of x
    - sigma --> standard deviation of x

    :param x: feature data in dataframe (569 x 1 columns)
              --> i.e. 'worst concave points', 'worst perimeter'
    :return: standardized feature data in numpy array (569 x 1 columns)
             --> i.e. 'std worst concave points', 'std worst perimeter'
    """
    z = list(map(lambda x_: (x_ - np.mean(x)) / np.std(x), x))
    return z


def main():
    """
    #### Objective TASK 2:

    - 1 - Implement the fit_mse method [CLASS]
    - 2 - Implement the predict method [CLASS]
    - 3 - Load the dataset. Independant variables: 'worst concave points', 'worst perimeter',
          'worst radius' [MAIN] - done
    - 4 - Standardize X [MAIN] - done
    - 5 - Instantiate CustomLogisticRegression where fit_intercept=True, l_rate=0.01, n_epoch=1000) [MAIN] - done
    - 6 - Fit the model with training set using fit_mse & Stochastic Gradient Descent algorithm
    - 7 - Predict y_hat values
    - 8 - Calculate accuracy score
    - 9 - Print coef_ array and accuracy score as a Python dictionary

    #### Objectives TASK 1:

    - 1 - Create the CustomLogisticRegression class [CLASS] - done
    - 2 - Create the __init__ method [CLASS] - done
    - 3 - Create the sigmoid method [CLASS] - done
    - 4 - Create the predict_proba method [CLASS] - done
    - 5 - Load the Breast Cancer Wisconsin dataset. - done
      Select worst concave points and worst perimeter X as features - done
      and target y as the target variable [MAIN] - done
    - 6 - Standardize X [FUNCTION standardize_z] - done
    - 7 - Split the dataset including the target variable into training and test sets.
      Set train_size=0.8 and random_state=43 [MAIN] - done
    - 8 - Given the coefficients below, calculate the probabilities of the first 10 rows
      in the test set. (Don't need the training set in this stage) [MAIN] - done
    - 9 - Print these probabilities as a Python list [MAIN] - done
    """
    # # # model settings
    intercept_flag = True
    n_epoch, learning_rate = 1000, 0.01
    train_size, random_state = 0.8, 43
    cut_off = 0.5

    # 3 - Load Breast Cancer Wisconsin dataset
    x = load_breast_cancer(as_frame=True)
    # # # Select 'worst concave points', 'worst perimeter' and 'worst radius' as features X
    X = x.data
    X = pd.DataFrame(X[['worst concave points', 'worst perimeter', 'worst radius']])
    # # # Select 'target' variable as target Y
    Y = x.target

    # 4 - standardize X using z-standardization
    X['worst concave points'] = standardize_z(X['worst concave points'])
    X['worst perimeter'] = standardize_z(X['worst perimeter'])
    X['worst radius'] = standardize_z(X['worst radius'])

    # 5 - Initialize custom logistic regression model
    # # # instantiate CustomLogisticRegression with settings provided
    custom_lr_model = CustomLogisticRegression(fit_intercept=intercept_flag, l_rate=learning_rate, n_epoch=n_epoch)

    # # # split the data 80-20 with random_state=43
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_size, random_state=random_state)

    # 6 - Fit the model
    custom_lr_model.fit_mse(X_train, y_train)

    # # 7 - Predict yhat value
    y_predicted = custom_lr_model.predict(X_test, cut_off=cut_off)

    # # 8 - Calculate model accuracy
    acc_score = accuracy_score(np.array(y_test), np.array(y_predicted))

    # 9 - print coefficient array and accuracy score as Python dictionary
    result_dictionary = {'coef_': custom_lr_model.coef_.T.tolist()[0], 'accuracy': acc_score}
    print(result_dictionary)


if __name__ == "__main__":
    main()
