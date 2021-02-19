import pickle

import numpy as np
import pandas as pd
from keras.losses import mean_squared_logarithmic_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

import datasets
from model_interface import ModelInterface


class LinearRegressionModel(ModelInterface):
    def train(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
              histogram_path=None) -> None:
        """
        Train a regression model for spatial join cost estimator, then save the trained model to file
        """

        # Extract train and test data, but only use train data
        X_train, y_train, X_test, y_test = datasets.load_tabular_features(join_result_path, tabular_path)

        # Fit and save the model
        model = LinearRegression().fit(X_train, y_train)
        pickle.dump(model, open(model_path, 'wb'))

    def test(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
             histogram_path=None) -> (float, float, float, float):
        """
        Evaluate the accuracy metrics of a trained  model for spatial join cost estimator
        :return mean_squared_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, mean_absolute_error
        """

        # Extract train and test data, but only use test data
        X_train, y_train, X_test, y_test = datasets.load_tabular_features(join_result_path, tabular_path)

        # Load the model and use it for prediction
        loaded_model = pickle.load(open(model_path, 'rb'))
        y_pred = loaded_model.predict(X_test)

        # TODO: delete this dumping action. This is just for debugging
        test_df = pd.DataFrame()
        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred
        test_df.to_csv('data/temp/test_df.csv')

        # Compute accuracy metrics
        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        msle = np.mean(mean_squared_logarithmic_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        return mse, mape, msle, mae
