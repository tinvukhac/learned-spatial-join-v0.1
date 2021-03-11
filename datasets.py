import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import copy
import math

# A shared random state will ensure that data is split in a same way in both train and test function
RANDOM_STATE = 42


def load_tabular_features_hadoop(distribution='all', matched=False, scale='all', minus_one=False):
    tabular_path = 'data/join_results/train/join_cardinality_data_points_sara.csv'
    print(tabular_path)
    tabular_features_df = pd.read_csv(tabular_path, delimiter='\\s*,\\s*', header=0)

    if distribution != 'all':
        tabular_features_df = tabular_features_df[tabular_features_df['label'].str.contains('_{}'.format(distribution))]

    if matched:
        tabular_features_df = tabular_features_df[tabular_features_df['label'].str.contains('_Match')]

    if scale != all:
        tabular_features_df = tabular_features_df[tabular_features_df['label'].str.contains(scale)]

    if minus_one:
        tabular_features_df['join_sel'] = 1 - tabular_features_df['join_sel']

    tabular_features_df = tabular_features_df.drop(columns=['label', 'coll1', 'D1', 'coll2', 'D2'])
    tabular_features_df = tabular_features_df.rename(columns={x: y for x, y in zip(tabular_features_df.columns, range(0, len(tabular_features_df.columns)))})

    # Get train and test data
    train_data, test_data = train_test_split(tabular_features_df, test_size=0.20, random_state=RANDOM_STATE)

    num_features = len(tabular_features_df.columns) - 1

    X_train = pd.DataFrame.to_numpy(train_data[[i for i in range(num_features)]])
    y_train = train_data[num_features]
    X_test = pd.DataFrame.to_numpy(test_data[[i for i in range(num_features)]])
    y_test = test_data[num_features]

    return X_train, y_train, X_test, y_test


def load_tabular_features(join_result_path, tabular_path, normalize=False):
    tabular_features_df = pd.read_csv(tabular_path, delimiter='\\s*,\\s*', header=0)
    cols = ['dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']
    join_df = pd.read_csv(join_result_path, delimiter=',', header=None, names=cols)
    join_df = join_df[join_df.result_size != 0]
    join_df = pd.merge(join_df, tabular_features_df, left_on='dataset1', right_on='dataset_name')
    join_df = pd.merge(join_df, tabular_features_df, left_on='dataset2', right_on='dataset_name')

    cardinality_x = join_df['cardinality_x']
    cardinality_y = join_df['cardinality_y']
    result_size = join_df['result_size']

    join_selectivity = 1 - result_size / (cardinality_x * cardinality_y)

    join_df = join_df.drop(
        columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', 'mbr_tests', 'duration'])

    if normalize:
        column_groups = [
            ['AVG area_x', 'AVG area_y'],
            ['AVG x_x', 'AVG y_x', 'AVG x_y', 'AVG y_y'],
            ['E0_x', 'E2_x', 'E0_y', 'E2_y'],
            ['cardinality_x', 'cardinality_y'],
        ]
        for column_group in column_groups:
            input_data = join_df[column_group].to_numpy()
            original_shape = input_data.shape
            reshaped = input_data.reshape(input_data.size, 1)
            reshaped = preprocessing.minmax_scale(reshaped)
            join_df[column_group] = reshaped.reshape(original_shape)

    # Rename the column's names to numbers for easier access
    join_df = join_df.rename(columns={x: y for x, y in zip(join_df.columns, range(0, len(join_df.columns)))})

    # Save the number of features in order to extract (X, y) correctly
    num_features = len(join_df.columns)

    # Append the target to the right of data frame
    join_df.insert(len(join_df.columns), 'join_selectivity', join_selectivity, True)

    # TODO: delete this dumping action. This is just for debugging
    join_df.to_csv('data/temp/join_df.csv')

    # Split join data to train and test data
    target = 'join_selectivity'
    train_data, test_data = train_test_split(join_df, test_size=0.20, random_state=RANDOM_STATE)

    X_train = pd.DataFrame.to_numpy(train_data[[i for i in range(num_features)]])
    y_train = train_data[target]
    X_test = pd.DataFrame.to_numpy(test_data[[i for i in range(num_features)]])
    y_test = test_data[target]

    return X_train, y_train, X_test, y_test


def load_datasets_feature(filename):
    features_df = pd.read_csv(filename, delimiter=',', header=0)
    return features_df


def load_join_data(features_df, result_file, histograms_path, num_rows, num_columns):
    cols = ['dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']
    result_df = pd.read_csv(result_file, delimiter=',', header=None, names=cols)
    result_df = result_df[result_df.result_size != 0]
    # result_df = result_df.sample(frac=1)
    result_df = pd.merge(result_df, features_df, left_on='dataset1', right_on='dataset_name')
    result_df = pd.merge(result_df, features_df, left_on='dataset2', right_on='dataset_name')

    # Load histograms
    ds1_histograms, ds2_histograms, ds1_original_histograms, ds2_original_histograms, ds_all_histogram, ds_bops_histogram = load_histograms(
        result_df, histograms_path, num_rows, num_columns)

    # Compute BOPS
    bops = np.multiply(ds1_original_histograms, ds2_original_histograms)
    # print (bops)
    bops = bops.reshape((bops.shape[0], num_rows * num_columns))
    bops_values = np.sum(bops, axis=1)
    bops_values = bops_values.reshape((bops_values.shape[0], 1))
    # result_df['bops'] = bops_values
    cardinality_x = result_df[' cardinality_x']
    cardinality_y = result_df[' cardinality_y']
    result_size = result_df['result_size']
    mbr_tests = result_df['mbr_tests']

    join_selectivity = 1 - result_size / (cardinality_x * cardinality_y)
    # join_selectivity = join_selectivity * math.pow(10, 9)
    join_selectivity_log = copy.deepcopy(join_selectivity)
    join_selectivity_log = join_selectivity_log.apply(lambda x: (-1) * math.log10(x))

    # print(join_selectivity)
    # join_selectivity = -math.log10(join_selectivity)

    mbr_tests_selectivity = mbr_tests / (cardinality_x * cardinality_y)
    mbr_tests_selectivity = mbr_tests_selectivity * math.pow(10, 9)

    duration = result_df['duration']

    dataset1 = result_df['dataset1']
    dataset2 = result_df['dataset2']

    # result_df = result_df.drop(columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', ' cardinality_x', ' cardinality_y'])
    # result_df = result_df.drop(
    #     columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y'])

    result_df = result_df.drop(
        columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', ' cardinality_x',
                 ' cardinality_y', 'mbr_tests', 'duration'])

    x = result_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    result_df = pd.DataFrame(x_scaled)

    result_df['cardinality_x'] = cardinality_x
    result_df['cardinality_y'] = cardinality_y
    result_df['bops'] = bops_values
    result_df['dataset1'] = dataset1
    result_df['dataset2'] = dataset2
    result_df.insert(len(result_df.columns), 'result_size', result_size, True)
    result_df.insert(len(result_df.columns), 'join_selectivity', join_selectivity, True)
    result_df.insert(len(result_df.columns), 'join_selectivity_log', join_selectivity_log, True)
    result_df.insert(len(result_df.columns), 'mbr_tests', mbr_tests, True)
    result_df.insert(len(result_df.columns), 'mbr_tests_selectivity', mbr_tests_selectivity, True)
    result_df.insert(len(result_df.columns), 'duration', duration, True)

    result_df.to_csv('data/temp/result_df.csv')

    return result_df, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram


def load_join_data2(features_df, result_file, histograms_path, num_rows, num_columns):
    cols = ['count', 'dataset1', 'dataset2', 'result_size', 'mbr_tests', 'duration']
    result_df = pd.read_csv(result_file, delimiter=',', header=None, names=cols)

    # result_df = result_df.sample(frac=1)
    result_df = pd.merge(result_df, features_df, left_on='dataset1', right_on='dataset_name')
    result_df = pd.merge(result_df, features_df, left_on='dataset2', right_on='dataset_name')

    # Load histograms
    ds1_histograms, ds2_histograms, ds1_original_histograms, ds2_original_histograms, ds_all_histogram, ds_bops_histogram = load_histograms2(
        result_df, histograms_path, num_rows, num_columns)

    # Compute BOPS
    bops = np.multiply(ds1_original_histograms, ds2_original_histograms)
    # print (bops)
    bops = bops.reshape((bops.shape[0], num_rows * num_columns))
    bops_values = np.sum(bops, axis=1)
    bops_values = bops_values.reshape((bops_values.shape[0], 1))
    # result_df['bops'] = bops_values
    cardinality_x = result_df[' cardinality_x']
    cardinality_y = result_df[' cardinality_y']
    result_size = result_df['result_size']
    mbr_tests = result_df['mbr_tests']
    join_selectivity = result_size / (cardinality_x * cardinality_y)
    join_selectivity = join_selectivity * math.pow(10, 9)

    dataset1 = result_df['dataset1']
    dataset2 = result_df['dataset2']

    # result_df = result_df.drop(columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', ' cardinality_x', ' cardinality_y'])
    # result_df = result_df.drop(
    #     columns=['result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y'])

    result_df = result_df.drop(
        columns=['count', 'result_size', 'dataset1', 'dataset2', 'dataset_name_x', 'dataset_name_y', ' cardinality_x',
                 ' cardinality_y', 'mbr_tests', 'duration'])

    x = result_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    result_df = pd.DataFrame(x_scaled)

    result_df['cardinality_x'] = cardinality_x
    result_df['cardinality_y'] = cardinality_y
    result_df['bops'] = bops_values
    result_df['dataset1'] = dataset1
    result_df['dataset2'] = dataset2
    result_df.insert(len(result_df.columns), 'result_size', result_size, True)
    result_df.insert(len(result_df.columns), 'join_selectivity', join_selectivity, True)
    result_df.insert(len(result_df.columns), 'mbr_tests', join_selectivity, True)

    # print (len(result_df))
    # result_df.to_csv('result_df.csv')

    return result_df, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram


def load_histogram(histograms_path, num_rows, num_columns, dataset):
    hist = np.genfromtxt('{}/{}x{}/{}'.format(histograms_path, num_rows, num_columns, dataset), delimiter=',')
    # normalized_hist = hist / hist.max() # divide by max value
    normalized_hist = hist / hist.sum() # divide by sum of all value
    normalized_hist = normalized_hist.reshape((hist.shape[0], hist.shape[1], 1))
    hist = hist.reshape((hist.shape[0], hist.shape[1], 1))
    return normalized_hist, hist


def load_histogram2(histograms_path, num_rows, num_columns, count, dataset):
    hist = np.genfromtxt('{}/{}x{}/{}/{}'.format(histograms_path, num_rows, num_columns, count, dataset), delimiter=',')
    normalized_hist = hist / hist.max()
    normalized_hist = normalized_hist.reshape((hist.shape[0], hist.shape[1], 1))
    hist = hist.reshape((hist.shape[0], hist.shape[1], 1))
    return normalized_hist, hist


def load_histograms(result_df, histograms_path, num_rows, num_columns):
    ds1_histograms = []
    ds2_histograms = []
    ds1_original_histograms = []
    ds2_original_histograms = []
    ds_all_histogram = []
    ds_bops_histogram = []

    for dataset in result_df['dataset1']:
        normalized_hist, hist = load_histogram(histograms_path, num_rows, num_columns, dataset)
        ds1_histograms.append(normalized_hist)
        ds1_original_histograms.append(hist)

    for dataset in result_df['dataset2']:
        normalized_hist, hist = load_histogram(histograms_path, num_rows, num_columns, dataset)
        ds2_histograms.append(normalized_hist)
        ds2_original_histograms.append(hist)

    for i in range(len(ds1_histograms)):
        hist1 = ds1_original_histograms[i]
        hist2 = ds2_original_histograms[i]
        combined_hist = np.dstack((hist1, hist2))
        combined_hist = combined_hist / combined_hist.max()
        ds_all_histogram.append(combined_hist)

    for i in range(len(ds1_histograms)):
        hist1 = ds1_original_histograms[i]
        hist2 = ds2_original_histograms[i]
        bops_hist = np.multiply(hist1, hist2)
        if bops_hist.max() > 0:
            bops_hist = bops_hist / bops_hist.max()
        ds_bops_histogram.append(bops_hist)

    return np.array(ds1_histograms), np.array(ds2_histograms), np.array(ds1_original_histograms), np.array(
        ds2_original_histograms), np.array(ds_all_histogram), np.array(ds_bops_histogram)


def load_histograms2(result_df, histograms_path, num_rows, num_columns):
    ds1_histograms = []
    ds2_histograms = []
    ds1_original_histograms = []
    ds2_original_histograms = []
    ds_all_histogram = []
    ds_bops_histogram = []

    for index, row in result_df.iterrows():
        count = row['count']
        dataset1 = row['dataset1']
        dataset2 = row['dataset2']

        normalized_hist, hist = load_histogram2(histograms_path, num_rows, num_columns, count, dataset1)
        ds1_histograms.append(normalized_hist)
        ds1_original_histograms.append(hist)

        normalized_hist, hist = load_histogram2(histograms_path, num_rows, num_columns, count, dataset2)
        ds2_histograms.append(normalized_hist)
        ds2_original_histograms.append(hist)

    # count = 0
    # for dataset in result_df['dataset1']:
    #     count += 1
    #     normalized_hist, hist = load_histogram2(histograms_path, num_rows, num_columns, count, dataset)
    #     ds1_histograms.append(normalized_hist)
    #     ds1_original_histograms.append(hist)
    #
    # count = 0
    # for dataset in result_df['dataset2']:
    #     count += 1
    #     normalized_hist, hist = load_histogram2(histograms_path, num_rows, num_columns, count, dataset)
    #     ds2_histograms.append(normalized_hist)
    #     ds2_original_histograms.append(hist)

    for i in range(len(ds1_histograms)):
        hist1 = ds1_original_histograms[i]
        hist2 = ds2_original_histograms[i]
        combined_hist = np.dstack((hist1, hist2))
        combined_hist = combined_hist / combined_hist.max()
        ds_all_histogram.append(combined_hist)

    for i in range(len(ds1_histograms)):
        hist1 = ds1_original_histograms[i]
        hist2 = ds2_original_histograms[i]
        bops_hist = np.multiply(hist1, hist2)
        if bops_hist.max() > 0:
            bops_hist = bops_hist / bops_hist.max()
        ds_bops_histogram.append(bops_hist)

    return np.array(ds1_histograms), np.array(ds2_histograms), np.array(ds1_original_histograms), np.array(
        ds2_original_histograms), np.array(ds_all_histogram), np.array(ds_bops_histogram)


def main():
    print('Dataset utils')

    load_tabular_features_hadoop(distribution='Uniform', matched=True)

    # features_df = load_datasets_feature('data/uniform_datasets_features.csv')
    # load_join_data(features_df, 'data/uniform_result_size.csv', 'data/histogram_uniform_values', 16, 16)

    # features_df = load_datasets_feature('data/data_aligned/aligned_small_datasets_features.csv')
    # join_data, ds1_histograms, ds2_histograms, ds_all_histogram = load_join_data(features_df,
    #                                                                              'data/data_aligned/join_results_small_datasets.csv',
    #                                                                              'data/data_aligned/histograms/small_datasets', 32,
    #                                                                              32)
    # print (join_data)


if __name__ == '__main__':
    main()
