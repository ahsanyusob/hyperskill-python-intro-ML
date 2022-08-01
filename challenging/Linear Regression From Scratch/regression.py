import math
import pandas as pd
import numpy as np


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0

    def fit(self, x, y):
        if self.fit_intercept:
            X = np.array([[1.0, x['Capacity'][i], x['Age'][i]] for i in range(len(x))])  # create nx3-(sys matrix)
            Y = np.array([[y[i]] for i in range(len(y))])  # create nx1-(output matrix)
            beta = np.linalg.inv(X.T @ X) @ X.T @ Y  # 3xn x nx3 x 3xn x nx1 = 3x1
            self.intercept = beta[0, 0]
            self.coefficient = beta[1:]
        else:
            X = np.array([[x['Capacity'][i], x['Age'][i]] for i in range(len(x))])  # create nx2-(system matrix)
            Y = np.array([[y[i]] for i in range(len(y))])  # create nx1-(output matrix)
            beta = np.linalg.inv(X.T @ X) @ X.T @ Y  # 2xn x nx2 x 2xn x nx1 = 2x1
            self.intercept = None
            self.coefficient = beta
        return self.intercept, self.coefficient, beta

    def predict(self, x, beta):
        if self.fit_intercept:
            X = np.array([[1.0, x['Capacity'][i], x['Age'][i]] for i in range(len(x))])  # create nx3-(sys matrix)
        else:
            X = np.array([[x['Capacity'][i], x['Age'][i]] for i in range(len(x))])  # create nx2-(system matrix)
        return X @ beta

    def r2_score(self, y, y_hat):
        Y = np.array([[y[i]] for i in range(len(y))])  # create nx1-(output matrix)
        y_mean = Y.mean()
        num = 0
        den = 0
        for i in range(len(Y)):
            num += (Y[i] - y_hat[i]) ** 2
            den += (Y[i] - y_mean) ** 2
        res = 1 - num/den
        return res[0]

    def rmse(self, y, y_hat):
        Y = np.array([[y[i]] for i in range(len(y))])  # create nx1-(output matrix)
        square_diff = 0
        for i in range(len(Y)):
            square_diff += (Y[i] - y_hat[i]) ** 2
        res = (1/len(Y)) * square_diff
        res = math.sqrt(res[0])
        return res


def main():
    # Data x and y
    capacity = [0.9, 0.5, 1.75, 2.0, 1.4, 1.5, 3.0, 1.1, 2.6, 1.9]
    age = [11, 11, 9, 8, 7, 7, 6, 5, 5, 4]
    cost_per_ton = [21.95, 27.18, 16.9, 15.37, 16.03, 18.15, 14.22, 18.72, 15.4, 14.69]

    # Intercept exist?
    intercept_flag = True

    # 1 - LOAD a pandas DataFrame containing x and y
    data_frame = pd.DataFrame(zip(capacity, age, cost_per_ton), columns=['Capacity', 'Age', 'Cost/ton'])

    # 2 - INITIALIZE CustomLinearRegression class
    linear_regression_model = CustomLinearRegression(fit_intercept=intercept_flag)

    # 3 - Implement the fit() method
    intercept, coefficients, beta_hat = linear_regression_model.fit(data_frame[['Capacity', 'Age']],
                                                                    data_frame['Cost/ton'])

    # 4 - Predict y for the other dataset xwz and print the result (yhat)
    y_predicted = linear_regression_model.predict(data_frame[['Capacity', 'Age']], beta_hat)
    # print(y_predicted)

    # 5 - Calculate RMSE & R2 metrics
    r2_metrics = linear_regression_model.r2_score(data_frame['Cost/ton'], y_predicted)
    rmse_metrics = linear_regression_model.rmse(data_frame['Cost/ton'], y_predicted)

    # 6 - Print the intercept, coefficient, RMSE, and R2 values as a Python dictionary
    result_dict = {'Intercept': intercept, 'Coefficient': coefficients, 'R2': r2_metrics, 'RMSE': rmse_metrics}
    print(result_dict)


if __name__ == '__main__':
    main()

