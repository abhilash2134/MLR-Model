# -*- coding: utf-8 -*-

"""
@author Abhilash
August 2021
"""

import os
import pandas as pd
import pickle
import numpy

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

import matplotlib.pyplot as plt

class ModelComparison:
    def __init__(self):
        self.x_y = None
        self.x_transform_y = None
        self.y = None
        self.scores = {}
        self.metrics = {}

    def load_data(self):
        """ Method which loads the important parameters from csv file """
        data = pd.read_csv(path_data)
        
        Predictors = ['X1', 'X3', 'X4', 'X5']
        Response   = ['Y2']
        
        data_relevant = data[[*Predictors, *Response]].dropna()
        self.x_y = data_relevant[Predictors]
        self.y = data_relevant[Response]
        self.x_y, self.y = self.x_y.to_numpy(), self.y.to_numpy()

        self.preprocessing()

    def preprocessing(self):
        """ Method generates polynomial features """
        poly = PolynomialFeatures(degree=3)
        self.x_transform_y = poly.fit_transform(self.x_y)

    def linear_regression(self):
        """ Least squares Linear Regression """
        reg = linear_model.LinearRegression(fit_intercept=True)
        reg.fit(self.x_transform_y, self.y)

        score = reg.score(self.x_transform_y, self.y)
        self.scores['Standard'] = score

        y_predict = reg.predict(self.x_transform_y)
        mae, mse, median = self.get_metrics(y_predict)
        self.metrics['Linear Regression'] = (mae, mse, median)

        self.save_model(reg, 'linear')
        return y_predict

    def linear_regression_lasso(self):
        """ Linear Model trained with L1 prior as regularizer """
        reg = linear_model.Lasso(alpha=0.4, max_iter=100000000)
        reg.fit(self.x_transform_y, self.y)
        score = reg.score(self.x_transform_y, self.y)
        self.scores['Lasso\t'] = score

        y_predict = reg.predict(self.x_transform_y)
        mae, mse, median = self.get_metrics(y_predict)
        self.metrics['Linear regression Lasso'] = (mae, mse, median)

        self.save_model(reg, 'lasso')
        return y_predict

    def linear_regression_bayesian(self):
        """  """
        reg = linear_model.BayesianRidge(fit_intercept=True)
        reg.fit(self.x_transform_y, self.y)
        score = reg.score(self.x_transform_y, self.y)
        self.scores['Bayesian'] = score

        y_predict = reg.predict(self.x_transform_y)
        mae, mse, median = self.get_metrics(y_predict)
        self.metrics['Linear Regression Bayesian'] = (mae, mse, median)

        self.save_model(reg, 'bayesian')
        return y_predict

    def linear_regression_ridge(self):
        """  """
        reg = linear_model.Ridge(alpha=.5)
        reg.fit(self.x_transform_y, self.y)
        score = reg.score(self.x_transform_y, self.y)
        self.scores['Ridge\t'] = score

        y_predict = reg.predict(self.x_transform_y)
        mae, mse, median = self.get_metrics(y_predict)
        self.metrics['Linear Regression Ridge'] = (mae, mse, median)

        self.save_model(reg, 'ridge')
        return y_predict

    def svr(self):
        """ Method for support vector regression """
        regr = make_pipeline(StandardScaler(), SVR(C=2, epsilon=0.2, kernel='rbf'))
        regr.fit(self.x_transform_y, self.y)
        score_svr = regr.score(self.x_transform_y, self.y)
        self.scores['SVR\t\t'] = score_svr

        y_predict = regr.predict(self.x_transform_y)
        mae, mse, median = self.get_metrics(y_predict)
        self.metrics['Support vector regression'] = (mae, mse, median)

        self.save_model(regr, 'svr')
        return y_predict


    def gpr(self):
        """ Method for support vector regression """
        
        kernel = DotProduct() + WhiteKernel()
        regr = GaussianProcessRegressor(kernel=kernel,
                                        random_state=0)
        regr.fit(self.x_transform_y, self.y)
        score_gpr= regr.score(self.x_transform_y, self.y)
        self.scores['GPR\t\t'] = score_gpr

        y_predict = regr.predict(self.x_transform_y)
        mae, mse, median = self.get_metrics(y_predict)
        self.metrics['Gaussian Process Regression'] = (mae, mse, median)

        self.save_model(regr, 'gpr')
        return y_predict

    def get_metrics(self, y_predict):
        mae = metrics.mean_absolute_error(self.y, y_predict)
        mse = metrics.mean_squared_error(self.y, y_predict)
        median = metrics.median_absolute_error(self.y, y_predict)

        mae, mse, median = round(mae, 4), round(mse, 4), round(median, 4)

        return mae, mse, median

    def plot_regression(self, y_pred_lin, y_pred_lasso, y_pred_bayesian, y_pred_ridge):
        """ Method which shows four scatter plots for each regression type one """
        fig, axs = plt.subplots(2, 2, figsize=(9, 9))

        axs[0, 0].scatter(self.y, y_pred_lin, color='lightseagreen')
        m, b = numpy.polyfit(self.y.reshape((1,-1))[0], y_pred_lin.reshape((1,-1))[0], 1)
        axs[0, 0].plot(self.y, m*self.y+b, color='red')
        axs[0, 0].set_xlabel("Labels")
        axs[0, 0].set_ylabel("Predictions")
        axs[0, 0].set_title("Linear")

        axs[1, 0].scatter(self.y, y_pred_lasso, color='lightseagreen')
        m, b = numpy.polyfit(self.y.reshape((1,-1))[0], y_pred_lasso.reshape((1,-1))[0], 1)
        axs[1, 0].plot(self.y, m*self.y+b, color='red')
        axs[1, 0].set_xlabel("Labels")
        axs[1, 0].set_ylabel("Predictions")
        axs[1, 0].set_title("LAR")

        axs[0, 1].scatter(self.y, y_pred_bayesian, color='lightseagreen')
        m, b = numpy.polyfit(self.y.reshape((1,-1))[0], y_pred_bayesian.reshape((1,-1))[0], 1)
        axs[0, 1].plot(self.y, m*self.y+b, color='red')
        axs[0, 1].set_xlabel("Labels")
        axs[0, 1].set_ylabel("Predictions")
        axs[0, 1].set_title("Bayesian")

        axs[1, 1].scatter(self.y, y_pred_ridge, color='lightseagreen')
        m, b = numpy.polyfit(self.y.reshape((1,-1))[0], y_pred_ridge.reshape((1,-1))[0], 1)
        axs[1, 1].plot(self.y, m*self.y+b, color='red')
        axs[1, 1].set_xlabel("Labels")
        axs[1, 1].set_ylabel("Predictions")
        axs[1, 1].set_title("Ridge")
        
        plt.tight_layout()
        plt.show()

    def plot_single(self, y_pred, title='Regression'):
        """ Method which shows scatter plot for a single regression type """
        plt.scatter(self.y, y_pred)
        plt.xlabel("Labels")
        plt.ylabel("Predictions")
        plt.title(title)
        plt.show()

    def print_scores(self):
        """ Prints achieved scores for all models """
        for metric in self.scores:
            value = self.scores[metric]
            print(metric, '\t:\t', value)
        print()

    def print_metrics(self):
        """ Prints achieved metrics for all models """
        for metric in self.metrics:
            print(metric)
            mae, mse, median = self.metrics[metric]
            print('MAE : ', mae, ',\tMSE : ', mse, ',\tMedian : ', median, '\n')

    def save_model(self, model, model_name):
        """ Saves trained model in sub-directory 'Models'"""
        pickle.dump(model, open(os.path.join(RUNNING_DIR, 'Models', model_name), 'wb'))
        pass


if __name__ == "__main__":
    RUNNING_DIR = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(RUNNING_DIR, 'Data.csv')

    modelling = ModelComparison()
    modelling.load_data()

    y_pred_lin      = modelling.linear_regression()
    y_pred_lasso    = modelling.linear_regression_lasso()
    y_pred_bayesian = modelling.linear_regression_bayesian()
    y_pred_ridge    = modelling.linear_regression_ridge()

    modelling.plot_regression(y_pred_lin, y_pred_lasso, y_pred_bayesian, y_pred_ridge)
    modelling.print_scores()
    modelling.print_metrics()