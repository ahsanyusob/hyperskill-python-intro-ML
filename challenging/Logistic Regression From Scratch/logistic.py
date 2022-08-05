import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        self.coef_ = np.array([[]])
        self.mse_error_first = np.array([[]])
        self.mse_error_last = np.array([[]])
        self.logloss_error_first = np.array([[]])
        self.logloss_error_last = np.array([[]])

    def sigmoid(self, t):
        return np.array([1 / (1 + math.exp(-t[i])) for i in range(len(t))])

    def predict_proba(self, row, coef_):
        t = row @ coef_
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        n_row = np.shape(X_train)[0]
        squared_err_list = [0 for _ in range(n_row)]
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
            for i, row in enumerate(X_train):
                row = np.array([row])  # 1 x (n_feature)
                y_hat = self.predict_proba(row, self.coef_)  # 1x1
                y_hat = np.array([y_hat])
                squared_err = ((y_hat - y_train[0, i]) ** 2) / n_row
                squared_err_list[i] = squared_err[0, 0]
                # update all weights
                self.coef_ = self.coef_ - self.l_rate * (y_hat - y_train[0, i]) * y_hat * (1 - y_hat) * row.T
            if iter_ == 1:
                self.mse_error_first = np.array(squared_err_list)
            elif iter_ == self.n_epoch:
                self.mse_error_last = np.array(squared_err_list)

    def fit_log_loss(self, X_train, y_train):
        n_row = np.shape(X_train)[0]
        logloss_err_list = [0 for _ in range(n_row)]
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
            for i, row in enumerate(X_train):
                row = np.array([row])  # 1 x (n_feature)
                y_hat = self.predict_proba(row, self.coef_)  # 1x1
                y_hat = np.array([y_hat])
                logloss_error = -(y_train[0, i] * math.log(y_hat) + (1 - y_train[0, i]) * math.log(1 - y_hat)) / n_row
                logloss_err_list[i] = logloss_error
                # update all weights
                self.coef_ = self.coef_ - self.l_rate * ((y_hat - y_train[0, i]) / n_row) * row.T
            if iter_ == 1:
                self.logloss_error_first = np.array(logloss_err_list)
            elif iter_ == self.n_epoch:
                self.logloss_error_last = np.array(logloss_err_list)

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
    #### Objective Stage 4:
    - 1 - Load the dataset as in stage 2 - done
    - 2 - Standardize X as in stage 2 - done
    - 3 - Instantiate the CustomLogisticRegression as in stage 2 - done
    - 4 - Split the dataset as in stage 1 - done
    - 5 - fit a model with fit_log_loss method
    - 6 - fit a model with fit_mse method
    - 7 - Import LogisticRegression from sklearn.linear_model & fit the model
    - 8 - Determine the errors during the first epoch of training the model with
      both fit_log_loss and fit_mse
    - 9 - Repeat the same operation for the last epoch
    - 10 - Predict y_hat values with all three models
    - 11 - Calculate the accuracy scores for all models
    - 12 - Print the accuracy scores and the errors from 1st and last epoch as
      a Python dictionary

    #### Objective Stage 3:
    - 1 - Implement fit_log_loss - done
    - 2 - Load the dataset as in previous stage - done
    - 3 - Standardize X - done
    - 4 - Instantiate the CustomLogisticRegression like in previous stage - done
    - 5 - Fit the model with the training set from Stage 1 using fit_log_loss - done
    - 6 - Predict y_hat values - done
    - 7 - Calculate the accuracy score - done
    - 8 - Print coef_ array & accuracy as a Python dictionary - done

    #### Objective Stage 2:
    - 1 - Implement the fit_mse method - done
    - 2 - Implement the predict method - done
    - 3 - Load the dataset. Independant variables: 'worst concave points', 'worst perimeter',
          'worst radius'  done
    - 4 - Standardize X - done
    - 5 - Instantiate CustomLogisticRegression where fit_intercept=True, l_rate=0.01, n_epoch=1000) - done
    - 6 - Fit the model with training set using fit_mse & Stochastic Gradient Descent algorithm - done
    - 7 - Predict y_hat values - done
    - 8 - Calculate accuracy score - done
    - 9 - Print coef_ array and accuracy score as a Python dictionary - done

    #### Objectives Stage 1:
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
    # 0 - model settings
    intercept_flag = True
    n_epoch, learning_rate = 1000, 0.01
    train_size, random_state = 0.8, 43
    cut_off = 0.5

    # 1 - Load Breast Cancer Wisconsin dataset
    x = load_breast_cancer(as_frame=True)
    # # # Select 'worst concave points', 'worst perimeter' and 'worst radius' as features X
    X = x.data
    X = pd.DataFrame(X[['worst concave points', 'worst perimeter', 'worst radius']])
    # # # Select 'target' variable as target Y
    Y = x.target

    # 2 - standardize X using z-standardization
    X['worst concave points'] = standardize_z(X['worst concave points'])
    X['worst perimeter'] = standardize_z(X['worst perimeter'])
    X['worst radius'] = standardize_z(X['worst radius'])

    # 3 - Instantiate CustomLogisticRegression with settings provided
    custom_lr_ll = CustomLogisticRegression(fit_intercept=intercept_flag, l_rate=learning_rate, n_epoch=n_epoch)
    custom_lr_mse = CustomLogisticRegression(fit_intercept=intercept_flag, l_rate=learning_rate, n_epoch=n_epoch)

    # 4 - split the data 80-20 with random_state=43
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=train_size, random_state=random_state)

    # 5 - Fit a model with fit_log_loss
    custom_lr_ll.fit_log_loss(X_train, y_train)

    # 6 - Fit a model with fit_mse
    custom_lr_mse.fit_mse(X_train, y_train)

    # 7 - Fit a model from sklearn using LogisticRegression
    sklearn_lr = LogisticRegression(fit_intercept=intercept_flag)
    sklearn_lr.fit(X_train, y_train)

    # 8 - Determine the errors during the first epoch of training custom lr for both fit_log_loss and fit_mse
    logloss_err_first = custom_lr_ll.logloss_error_first
    mse_err_first = custom_lr_mse.mse_error_first

    # 9 - Determine the errors during the last epoch of training custom lr for both fit_log_loss and fit_mse
    logloss_err_last = custom_lr_ll.logloss_error_last
    mse_err_last = custom_lr_mse.mse_error_last

    # 10 - Predict yhat values for all three models
    y_hat_ll = custom_lr_ll.predict(X_test, cut_off=cut_off)
    y_hat_mse = custom_lr_mse.predict(X_test, cut_off=cut_off)
    y_hat_sk = sklearn_lr.predict(X_test)

    # 11 - Calculate accuracy scores for all three models
    acc_score_ll = accuracy_score(np.array(y_test), np.array(y_hat_ll))
    acc_score_mse = accuracy_score(np.array(y_test), np.array(y_hat_mse))
    acc_score_sk = accuracy_score(np.array(y_test), np.array(y_hat_sk))

    # 12 - print coefficient array and accuracy score as Python dictionary
    res_dict = {'mse_accuracy': acc_score_mse, 'logloss_accuracy': acc_score_ll, 'sklearn_accuracy': acc_score_sk,
                'mse_error_first': mse_err_first.tolist(), 'mse_error_last': mse_err_last.tolist(),
                'logloss_error_first': logloss_err_first.tolist(), 'logloss_error_last': logloss_err_last.tolist()}
    print(res_dict)

    # 13 - answer 6 questions:
    print("Answers to the questions:")
    # i)   What is the minimum MSE value for the first epoch?
    print("1)", format(min(mse_err_first), '.5f'))
    # ii)  What is the minimum MSE value for the last epoch?
    print("2)", format(min(mse_err_last), '.5f'))
    # iii) What is the maximum Log-loss value for the first epoch?
    print("3)", format(max(logloss_err_first), '.5f'))
    # iv)  What is the maximum Log-loss value for the last epoch?
    print("4)", format(max(logloss_err_last), '.5f'))
    # v)   Has the range of the MSE values expanded or narrowed? (expanded/narrowed)
    print("5)", "expanded" if max(mse_err_first) - min(mse_err_first) <
                        max(mse_err_last) - min(mse_err_last) else "narrowed")
    # vi)  Has the range of the Log-loss values expanded or narrowed? (expanded/narrowed)
    print("6)", "expanded" if max(logloss_err_first) - min(logloss_err_first) <
                        max(logloss_err_last) - min(logloss_err_last) else "narrowed")


if __name__ == "__main__":
    main()
