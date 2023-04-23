"""
@author Abhilash: abhilash@doscon.no 
October 2021
Version 2: updated on 02/02/2023
"""

import os
import pandas as pd
import pickle

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


class ModelComparison:
    def __init__(self):
        self.x_tpi = None
        self.x_transform_tpi = None
        self.y_tpi = None
        self.scores = {}
        self.metrics = {}

    def load_data(self):
        """ Method which loads the important parameters from csv file """
        data = pd.read_csv(path_data)
        # remove all rows which contain a None: performs slightly better than refilling with mean
        data_relevant = data[['X1', 'X2', 'X3', 'X4', 'Y1']]
        
        self.x_tpi = data_relevant[['X1', 'X2', 'X3', 'X4']]
        self.y_tpi = data_relevant['Y1']
        self.x_tpi, self.y_tpi = self.x_tpi.to_numpy(), self.y_tpi.to_numpy()

        self.preprocessing()

    def preprocessing(self):
        """ Method generates polynomial features """
        poly = PolynomialFeatures(degree=3)
        self.x_transform_tpi = poly.fit_transform(self.x_tpi)

    def linear_regression(self):
        """ Least squares Linear Regression """
        regTPI = linear_model.LinearRegression(fit_intercept=False)
        regTPI.fit(self.x_transform_tpi, self.y_tpi)

        scoreTPI = regTPI.score(self.x_transform_tpi, self.y_tpi)
        self.scores['Standard'] = scoreTPI

        y_predictTPI = regTPI.predict(self.x_transform_tpi)
        mae, mse, median = self.get_metrics(y_predictTPI)
        self.metrics['Linear Regression'] = (mae, mse, median)

        self.save_model(regTPI, 'linear')
        return y_predictTPI

    def linear_regression_lasso(self):
        """ Linear Model trained with L1 prior as regularizer """
        regTPI = linear_model.Lasso(alpha=0.4, max_iter=100000000)
        regTPI.fit(self.x_transform_tpi, self.y_tpi)
        scoreTPI = regTPI.score(self.x_transform_tpi, self.y_tpi)
        self.scores['Lasso\t'] = scoreTPI

        y_predictTPI = regTPI.predict(self.x_transform_tpi)
        mae, mse, median = self.get_metrics(y_predictTPI)
        self.metrics['Linear regression Lasso'] = (mae, mse, median)

        self.save_model(regTPI, 'lasso')
        return y_predictTPI

    def linear_regression_bayesian(self):
        """  """
        regTPI = linear_model.BayesianRidge(fit_intercept=True)
        regTPI.fit(self.x_transform_tpi, self.y_tpi)
        scoreTPI = regTPI.score(self.x_transform_tpi, self.y_tpi)
        self.scores['Bayesian'] = scoreTPI

        y_predictTPI = regTPI.predict(self.x_transform_tpi)
        mae, mse, median = self.get_metrics(y_predictTPI)
        self.metrics['Linear Regression Bayesian'] = (mae, mse, median)

        self.save_model(regTPI, 'bayesian')
        return y_predictTPI

    def linear_regression_ridge(self):
        """  """
        regTPI = linear_model.Ridge(alpha=.5)
        regTPI.fit(self.x_transform_tpi, self.y_tpi)
        scoreTPI = regTPI.score(self.x_transform_tpi, self.y_tpi)
        self.scores['Ridge\t'] = scoreTPI

        y_predictTPI = regTPI.predict(self.x_transform_tpi)
        mae, mse, median = self.get_metrics(y_predictTPI)
        self.metrics['Linear Regression Ridge'] = (mae, mse, median)

        self.save_model(regTPI, 'ridge')
        return y_predictTPI

    def svr(self):
        """ Method for support vector regression """
        regr = make_pipeline(StandardScaler(), SVR(C=2, epsilon=0.2, kernel='rbf'))
        regr.fit(self.x_transform_tpi, self.y_tpi)
        score_svr = regr.score(self.x_transform_tpi, self.y_tpi)
        self.scores['SVR\t\t'] = score_svr

        y_predictTPI = regr.predict(self.x_transform_tpi)
        mae, mse, median = self.get_metrics(y_predictTPI)
        self.metrics['Support vector regression'] = (mae, mse, median)

        self.save_model(regr, 'svr')
        return y_predictTPI

    def mlp_regression(self):
        """ Method to train a multi layer perceptron (ANN) - not useful, predicts only mean! """
        X_train, X_test, y_train, y_test = train_test_split(self.x_transform_tpi, self.y_tpi, test_size=0.30,
                                                            random_state=40)
        mlp = MLPRegressor(hidden_layer_sizes=(8, 8, 8, 8, 4), activation='identity', max_iter=300,
                           learning_rate='adaptive', learning_rate_init=0.2, solver='adam', random_state=3)
        mlp.fit(X_train, y_train)

        predict_train = mlp.predict(X_train)
        predict_test = mlp.predict(X_test)
        scoreTrain = mlp.score(X_train, y_train)
        scoreTest = mlp.score(X_test, y_test)

        self.plot_single(predict_train, title='Multi-layer perceptron (Train)')
        self.plot_single(predict_test, title='Multi-layer perceptron (Test)')

        self.scores["MLP"] = (scoreTrain, scoreTest)

    def get_metrics(self, y_predict):
        mae = metrics.mean_absolute_error(self.y_tpi, y_predict)
        mse = metrics.mean_squared_error(self.y_tpi, y_predict)
        median = metrics.r2_score(self.y_tpi, y_predict)

        mae, mse, median = round(mae, 4), round(mse, 4), round(median, 4)

        return mae, mse, median

    def plot_regression(self, y_pred_lin, y_pred_lasso, y_pred_bayesian, y_pred_ridge):
        """ Method which shows four scatter plots for each regression type one """
        
        
        fig, axes = plt.subplots(1, 4, figsize=(17, 5))

        axes[0].scatter(self.y_tpi, y_pred_lin, color='red', s=20, label='Prediction')
        axes[0].plot(self.y_tpi, self.y_tpi, '#1fbdbd',
                        label='Ideal Prediction')
        axes[0].set(xlabel='True Data',
                    ylabel='Predicted Values', title= 'Least-Square: MSE={}'.format(self.metrics['Linear Regression'][0]))
        axes[0].grid
        axes[0].legend()
        
        axes[1].scatter(self.y_tpi, y_pred_lasso, color='red', s=20, label='Prediction')
        axes[1].plot(self.y_tpi, self.y_tpi, '#1fbdbd',
                        label='Ideal Prediction')
        axes[1].set(xlabel='True Data',
                    ylabel='Predicted Values', title='LAR: MSE={}'.format(self.metrics['Linear regression Lasso'][0]))
        axes[1].grid
        axes[1].legend()
        
        axes[2].scatter(self.y_tpi, y_pred_bayesian, color='red', s=20, label='Prediction')
        axes[2].plot(self.y_tpi, self.y_tpi, '#1fbdbd',
                        label='Ideal Prediction')
        axes[2].set(xlabel='True Data',
                    ylabel='Predicted Values', title='Bayesian: MSE={}'.format(self.metrics['Linear Regression Bayesian'][0]))
        axes[2].grid
        axes[2].legend()
        
        axes[3].scatter(self.y_tpi, y_pred_svr, color='red', s=20, label='Prediction')
        axes[3].plot(self.y_tpi, self.y_tpi, '#1fbdbd',
                        label='Ideal Prediction')
        axes[3].set(xlabel='True Data',
                    ylabel='Predicted Values', title='SVR: MSE={}'.format(self.metrics['Support vector regression'][0]))
        axes[3].grid
        axes[3].legend()

        plt.show()

    def plot_single(self, y_pred, title='Regression'):
        """ Method which shows scatter plot for a single regression type """
        plt.scatter(self.y_tpi, y_pred)
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
        """ Saves trained model in sub-directory 'TPI models'"""
        pickle.dump(model, open(os.path.join(RUNNING_DIR, 'Models', model_name), 'wb'))
        pass


if __name__ == "__main__":
    RUNNING_DIR = os.path.dirname(os.path.abspath(__file__))
    path_data = os.path.join(RUNNING_DIR, 'Data.csv')
    data = pd.read_csv(path_data)
    
    modelling = ModelComparison()
    modelling.load_data()

    y_pred_lin = modelling.linear_regression()
    y_pred_lasso = modelling.linear_regression_lasso()
    y_pred_bayesian = modelling.linear_regression_bayesian()
    y_pred_ridge = modelling.linear_regression_ridge()
    y_pred_svr = modelling.svr()

    modelling.plot_regression(y_pred_lin, y_pred_lasso, y_pred_bayesian, y_pred_ridge)
    modelling.print_scores()
    modelling.print_metrics()
    

    data['Y1_Predict'] = y_pred_lin

    data.to_excel('DataEstimate.xlsx', index=False)

    # modelling.mlp_regression() ### not useful at all
