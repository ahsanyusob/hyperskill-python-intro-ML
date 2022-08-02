import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch

    def sigmoid(self, t):
        return [1 / (1 + math.exp(-t[i])) for i in range(len(t))]

    def predict_proba(self, row, coef_):
        if self.fit_intercept:
            row = np.array([[1 for _ in range(len(row))],
                            [row[i, 0] for i in range(len(row))],
                            [row[i, 1] for i in range(len(row))]])
            row = row.T
        else:
            coef_ = coef_[1:]
        t = row @ coef_
        return self.sigmoid(t)


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
    Objectives:

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
    # 1 - Initialize custom logistic regression model
    custom_log_reg_model = CustomLogisticRegression(fit_intercept=True)

    # 5a - Load Breast Cancer Wisconsin dataset
    x = load_breast_cancer(as_frame=True)
    # 5b - Select 'worst concave points' and 'worst perimeter' as features X
    X = x.data
    X = pd.DataFrame(X[['worst concave points', 'worst perimeter']])
    # 5c - Select 'target' variable as target Y
    Y = x.target

    # 6 - standardize X using z-standardization
    X['worst concave points'] = standardize_z(X['worst concave points'])
    X['worst perimeter'] = standardize_z(X['worst perimeter'])

    # 7 - split the data 80-20 with random_state=43
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=43)
    row_ = np.array(X_test)
    row_ = row_[:10, ::]

    # 8 - calculate the probabilities of the first 10 rows in the test set (coef is given with bias)
    coefficient = np.array([[0.77001597],
                            [-2.12842434],
                            [-2.39305793]])
    prob_list = custom_log_reg_model.predict_proba(row_, coefficient)

    # 9 - print the probabilities as a Python List
    print(prob_list)


if __name__ == "__main__":
    main()
