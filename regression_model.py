import pickle

import numpy as np
import pandas as pd
from keras.losses import mean_squared_logarithmic_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

import datasets
from model_interface import ModelInterface


class RegressionModel(ModelInterface):
    NORMALIZE = False

    def __init__(self, model_name):
        self.reg_model = LinearRegression()
        if model_name == 'linear':
            self.reg_model = LinearRegression()
        elif model_name == 'decision_tree':
            self.reg_model = DecisionTreeRegressor(max_depth=6)
        elif model_name == 'random_forest':
            self.reg_model = RandomForestRegressor(max_depth=6, random_state=0)

    def train(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
              histogram_path=None) -> None:
        """
        Train a regression model for spatial join cost estimator, then save the trained model to file
        """

        # Extract train and test data, but only use train data
        X_train, y_train, X_test, y_test = datasets.load_tabular_features(join_result_path, tabular_path, RegressionModel.NORMALIZE)

        # Fit and save the model
        model = self.reg_model.fit(X_train, y_train)
        pickle.dump(model, open(model_path, 'wb'))

    def test(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
             histogram_path=None) -> (float, float, float, float):
        """
        Evaluate the accuracy metrics of a trained  model for spatial join cost estimator
        :return mean_squared_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, mean_absolute_error
        """

        # Extract train and test data, but only use test data
        X_train, y_train, X_test, y_test = datasets.load_tabular_features(join_result_path, tabular_path, RegressionModel.NORMALIZE)

        # Load the model and use it for prediction
        loaded_model = pickle.load(open(model_path, 'rb'))
        y_pred = loaded_model.predict(X_test)

        # TODO: delete this dumping action. This is just for debugging
        test_df = pd.DataFrame()
        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred
        test_df.to_csv('data/temp/test_df.csv')

        # Compute accuracy metrics
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        msle = np.mean(mean_squared_logarithmic_error(y_test, y_pred))

        return mae, mape, mse, msle
