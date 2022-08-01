import pandas as pd
import numpy as np


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0

    def fit(self, x, y):
        if self.fit_intercept:
            X = np.array([[1.0, x['x'][i], x['w'][i], x['z'][i]] for i in range(len(x))])  # create nx4-(system matrix)
            Y = np.array([[y[i]] for i in range(len(y))])  # create nx1-(output matrix)
            beta = np.linalg.inv(X.T @ X) @ X.T @ Y  # 4xn x nx4 x 4xn x nx1 = 4x1
            self.intercept = beta[0]
            self.coefficient = beta[1:]
        else:
            X = np.array([[x['x'][i], x['w'][i], x['z'][i]] for i in range(len(x))])  # create nx3-(system matrix)
            Y = np.array([[y[i]] for i in range(len(y))])  # create nx1-(output matrix)
            beta = np.linalg.inv(X.T @ X) @ X.T @ Y  # 3xn x nx3 x 3xn x nx1 = 3x1
            self.intercept = None
            self.coefficient = beta
        return self.intercept, self.coefficient, beta

    def predict(self, x, beta):
        if self.fit_intercept:
            X = np.array([[1.0, x['x'][i], x['w'][i], x['z'][i]] for i in range(len(x))])
        else:
            X = np.array([[x['x'][i], x['w'][i], x['z'][i]] for i in range(len(x))])
        return X @ beta


def main():
    # Data x and y
    x = [4.0, 4.5, 5, 5.5, 6.0, 6.5, 7.0]
    w = [1, -3, 2, 5, 0, 3, 6]
    z = [11, 15, 12, 9, 18, 13, 16]
    y = [33, 42, 45, 51, 53, 61, 62]

    # Intercept exist?
    intercept_flag = False

    # 1 - LOAD a pandas DataFrame containing x and y
    data_frame = pd.DataFrame(zip(x, w, z, y), columns=['x', 'w', 'z', 'y'])

    # 2 - INITIALIZE CustomLinearRegression class
    linear_regression_model = CustomLinearRegression(fit_intercept=intercept_flag)

    # 3.1 - Implement the fit() method
    intercept, coefficients, beta_hat = linear_regression_model.fit(data_frame[['x', 'w', 'z']], data_frame['y'])
    beta_dict = {'Intercept': intercept, 'Coefficient': coefficients}

    # 3.2 - Print a dictionary containing the intercept and coefficient values.
    # print(beta_dict)

    # 4 - Predict y for the other dataset xwz --> y_hat
    y_predicted = linear_regression_model.predict(data_frame[['x', 'w', 'z']], beta_hat)

    # 5 - Print y_hat
    print(y_predicted)


if __name__ == '__main__':
    main()


