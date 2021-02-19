import datasets
from model_interface import ModelInterface

import numpy as np
import pandas as pd
from keras.losses import mean_squared_logarithmic_error
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


class LinearRegressionModel(ModelInterface):
    def train(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
              histogram_path=None) -> None:
        """
        Train a regression model for spatial join cost estimator, then save the trained model to file
        """

        tabular_features_df = datasets.load_datasets_feature(tabular_path)
        cols = ['dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']
        join_df = pd.read_csv(join_result_path, delimiter=',', header=None, names=cols)
        join_df = join_df[join_df.result_size != 0]
        join_df = pd.merge(join_df, tabular_features_df, left_on='dataset1', right_on='dataset_name')
        join_df = pd.merge(join_df, tabular_features_df, left_on='dataset2', right_on='dataset_name')

        cardinality_x = join_df[' cardinality_x']
        cardinality_y = join_df[' cardinality_y']
        result_size = join_df['result_size']

        join_selectivity = result_size / (cardinality_x * cardinality_y)

        join_df = join_df.drop(
            columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', ' cardinality_x',
                     ' cardinality_y', 'mbr_tests', 'duration'])

        join_df['cardinality_x'] = cardinality_x
        join_df['cardinality_y'] = cardinality_y

        join_df = join_df.rename(columns={x: y for x, y in zip(join_df.columns, range(0, len(join_df.columns)))})

        join_df.insert(len(join_df.columns), 'join_selectivity', join_selectivity, True)

        join_df.to_csv('data/temp/join_df.csv')

        # Split input data to train and test
        target = 'join_selectivity'
        train_data, test_data = train_test_split(join_df, test_size=0.20, random_state=42)
        num_features = len(join_df.columns) - 1
        X_train = pd.DataFrame.to_numpy(train_data[[i for i in range(num_features)]])
        X_test = pd.DataFrame.to_numpy(test_data[[i for i in range(num_features)]])
        y_train = train_data[target]
        y_test = test_data[target]

        model = LinearRegression().fit(X_train, y_train)
        pickle.dump(model, open(model_path, 'wb'))

    def test(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
             histogram_path=None) -> (float, float, float, float):
        """
        Evaluate the accuracy metrics of a trained  model for spatial join cost estimator
        :return mean_squared_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, mean_absolute_error
        """
        print('LinearRegressionModel.test')
        tabular_features_df = datasets.load_datasets_feature(tabular_path)
        cols = ['dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']
        join_df = pd.read_csv(join_result_path, delimiter=',', header=None, names=cols)
        join_df = join_df[join_df.result_size != 0]
        join_df = pd.merge(join_df, tabular_features_df, left_on='dataset1', right_on='dataset_name')
        join_df = pd.merge(join_df, tabular_features_df, left_on='dataset2', right_on='dataset_name')

        cardinality_x = join_df[' cardinality_x']
        cardinality_y = join_df[' cardinality_y']
        result_size = join_df['result_size']

        join_selectivity = result_size / (cardinality_x * cardinality_y)

        join_df = join_df.drop(
            columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', ' cardinality_x',
                     ' cardinality_y', 'mbr_tests', 'duration'])

        join_df['cardinality_x'] = cardinality_x
        join_df['cardinality_y'] = cardinality_y

        join_df = join_df.rename(columns={x: y for x, y in zip(join_df.columns, range(0, len(join_df.columns)))})

        join_df.insert(len(join_df.columns), 'join_selectivity', join_selectivity, True)

        join_df.to_csv('data/temp/join_df.csv')

        # Split input data to train and test
        target = 'join_selectivity'
        train_data, test_data = train_test_split(join_df, test_size=0.20, random_state=42)
        num_features = len(join_df.columns) - 1
        X_train = pd.DataFrame.to_numpy(train_data[[i for i in range(num_features)]])
        X_test = pd.DataFrame.to_numpy(test_data[[i for i in range(num_features)]])
        y_train = train_data[target]
        y_test = test_data[target]

        loaded_model = pickle.load(open(model_path, 'rb'))
        y_pred = loaded_model.predict(X_test)

        test_df = pd.DataFrame()
        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred
        test_df.to_csv('data/temp/test_df.csv')

        mse = mean_squared_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        msle = np.mean(mean_squared_logarithmic_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        return mse, mape, msle, mae
