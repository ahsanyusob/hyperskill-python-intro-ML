import pandas as pd
import numpy as np


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0

    def fit(self, x, y):
        if self.fit_intercept:
            X = np.array([[1.0, x[i]] for i in range(len(x))])  # create nx2-(system matrix)
            Y = np.array([[y[i]] for i in range(len(y))])  # create nx1-(output matrix)
            beta_0, beta_1 = np.linalg.inv(X.T @ X) @ X.T @ Y  # 2xn x nx2 x 2xn x nx1 = 2x1
            self.intercept = beta_0[0]
            self.coefficient = beta_1
        else:
            X = np.array([[x[i]] for i in range(len(x))])  # create nx1-(system matrix)
            Y = np.array([[y[i]] for i in range(len(y))])  # create nx1-(output matrix)
            beta = np.linalg.inv(X.T @ X) @ X.T @ Y  # 1xn x nx1 x 1xn x nx1 = 1x1
            self.intercept = None
            self.coefficient = beta

        return self.intercept, self.coefficient


def main():
    # Data x and y
    x = [4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0]
    y = [33, 42, 45, 51, 53, 61, 62]
    # x = [4.0, 7.0]
    # y = [10.0, 16.0]
    # x = [1.0, 2.0, 3.0, 4.0, 5.0]
    # y = [0.0, 0.0, 0.0, 0.0, 0.0]
    # x = [1.0, 4.5, 14.0, 3.8, 7.0, 19.4]
    # y = [106.0, 150.7, 200.9, 115.8, 177, 156]

    # 1 - LOAD a pandas DataFrame containing x and y
    data_frame = pd.DataFrame(zip(x, y), columns=['x', 'y'])

    # 2 - INITIALIZE CustomLinearRegression class
    linear_regression_model_1 = CustomLinearRegression()

    # 3 - Implement the fit() method
    b_0_1, b_1_1 = linear_regression_model_1.fit(data_frame['x'], data_frame['y'])
    beta_dict_1 = {'Intercept': b_0_1, 'Coefficient': b_1_1}

    # 4 - INITIALIZE CustomLinearRegression class with fit_intercept=True
    linear_regression_model_2 = CustomLinearRegression(fit_intercept=False)

    # 5 - Implement the fit() method
    b_0_2, b_1_2 = linear_regression_model_2.fit(data_frame['x'], data_frame['y'])
    beta_dict_2 = {'Intercept': b_0_2, 'Coefficient': b_1_2}

    # 6 - Print a dictionary containing the intercept and coefficient values.
    print(beta_dict_1)
    # print(beta_dict_2)


if __name__ == '__main__':
    main()
