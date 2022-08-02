import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = np.array([])
        self.intercept = 0.0
        self.beta = np.array([])

    def fit(self, x, y):
        if self.fit_intercept:
            x.insert(0, 'ones', np.ones(shape=(len(x), 1)), False)  # allow_duplicates = False
            X = np.array(x)  # create mxn-(system matrix) where m = number of columns and n = number of rows
            Y = np.array(y)  # create nx1-(output matrix)
            self.beta = np.linalg.inv(X.T @ X) @ X.T @ Y  # nxm x mxn x nxm x mx1 = nx1
            self.intercept = self.beta[0]
            self.coefficient = self.beta.T
        else:
            X = np.array(x)  # create mxn-(system matrix) where m = number of columns and n = number of rows
            Y = np.array(y)  # create nx1-(output matrix)
            self.beta = np.linalg.inv(X.T @ X) @ X.T @ Y  # nxm x mxn x nxm x mx1 = nx1
            self.intercept = 0.0
            self.coefficient = self.beta.T

    def predict(self, x):
        if self.fit_intercept:
            try:
                x.insert(0, 'ones', np.ones(shape=(len(x), 1)), False)  # allow_duplicates = False
            except ValueError:
                pass
            X = np.array(x)  # create mxn-(system matrix) where m = number of columns and n = number of rows
        else:
            X = np.array(x)  # create mxn-(system matrix) where m = number of columns and n = number of rows
        return X @ self.beta

    def r2_score(self, y, y_hat):
        Y = np.array(y)  # create nx1-(output matrix)
        y_mean = Y.mean()
        squared_error, squared_deviation = 0, 0
        for i in range(len(Y)):
            squared_error += (Y[i] - y_hat[i]) ** 2
            squared_deviation += (Y[i] - y_mean) ** 2
        r2 = 1 - squared_error/squared_deviation
        return r2[0]

    def rmse(self, y, y_hat):
        Y = np.array(y)  # create nx1-(output matrix)
        squared_error = 0
        for i in range(len(Y)):
            squared_error += (Y[i] - y_hat[i]) ** 2
        mse = (1/len(Y)) * squared_error
        rmse = math.sqrt(mse[0])
        return rmse


def main():
    # Intercept exist?
    intercept_flag = True

    # 1 - LOAD a pandas DataFrame containing x and y
    # Data x and y
    # f1 = [2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87]
    # f2 = [65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3]
    # f3 = [15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2]
    # y = [24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0]
    # dataframe_x = pd.DataFrame(zip(f1, f2, f3), columns=['f1', 'f2', 'f3'])
    # dataframe_y = pd.DataFrame(y, columns=['y'])

    # Or read from CSV
    dataframe_x = pd.read_csv('data_stage4.csv', usecols=[0, 1, 2])
    dataframe_y = pd.read_csv('data_stage4.csv', usecols=[3])

    # 2 - INITIALIZE CustomLinearRegression class
    custom_regression_model = CustomLinearRegression(fit_intercept=intercept_flag)
    scikit_regression_model = LinearRegression(fit_intercept=intercept_flag)

    # 3 - Fit the data by passing the X DataFrame and y Series to LinearRegression and CustomLinearRegression
    custom_regression_model.fit(dataframe_x, dataframe_y)
    scikit_regression_model.fit(dataframe_x, dataframe_y)
    intercept_crm = custom_regression_model.intercept[0]
    intercept_srm = scikit_regression_model.intercept_[0]
    coefficients_crm = custom_regression_model.coefficient[0][1:]
    coefficients_srm = scikit_regression_model.coef_[0][1:]

    # 4 - Predict y for the other dataset xwz and print the result (yhat)
    y_predicted_crm = custom_regression_model.predict(dataframe_x)
    y_predicted_srm = scikit_regression_model.predict(dataframe_x)

    # 5 - Calculate RMSE & R2 metrics
    r2_score_crm = custom_regression_model.r2_score(dataframe_y, y_predicted_crm)
    r2_score_srm = r2_score(dataframe_y, y_predicted_srm)
    rmse_crm = custom_regression_model.rmse(dataframe_y, y_predicted_crm)
    rmse_srm = math.sqrt(mean_squared_error(dataframe_y, y_predicted_srm))

    # 6 - Print the differences between intercept, coefficient, RMSE, and R2 values of
    # LinearRegression and CustomLinearRegression as a Python dictionary
    result_dictionary = {'Intercept': intercept_srm - intercept_crm, 'Coefficient': coefficients_srm - coefficients_crm,
                         'R2': r2_score_srm - r2_score_crm, 'RMSE': rmse_srm - rmse_crm}
    print(result_dictionary)


if __name__ == '__main__':
    main()

