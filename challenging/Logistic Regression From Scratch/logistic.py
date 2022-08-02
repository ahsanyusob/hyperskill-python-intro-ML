import numpy as np
import pandas as pd


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = ...
        self.l_rate = ...
        self.n_epoch = ...

    def sigmoid(self, t):
        return ...

    def predict_proba(self, row, coef_):
        t = ...
        return self.sigmoid(t)
