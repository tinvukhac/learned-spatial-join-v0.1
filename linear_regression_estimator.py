import numpy as np
import pandas as pd
from keras.losses import mean_squared_logarithmic_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

import datasets


def main():
    print('Linear regression model for spatial join estimator')
    tabular_path = 'data/tabular/tabular_all.csv'
    join_result_path = 'data/join_results/train/join_results_small_x_small_uniform.csv'
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

    reg = LinearRegression().fit(X_train, y_train)
    score = reg.score(X_test, y_test)
    print('Score: {}'.format(score))

    y_pred = reg.predict(X_test)

    test_df = pd.DataFrame()
    test_df['y_test'] = y_test
    test_df['y_pred'] = y_pred
    test_df.to_csv('data/temp/test_df.csv')

    print(f'MSE {mean_squared_error(y_test, y_pred)}')
    print(f'MAPE {mean_absolute_percentage_error(y_test, y_pred)}')
    print(f'Mean Square Logarithmic Error {np.mean(mean_squared_logarithmic_error(y_test, y_pred))}')
    print(f'MAE {mean_absolute_error(y_test, y_pred)}')


if __name__ == '__main__':
    main()
