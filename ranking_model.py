import pickle

import pandas as pd
from sklearn import metrics

import datasets
from model_interface import ModelInterface
import lightgbm as lgb
from sklearn.model_selection import train_test_split


class RankingModel(ModelInterface):
    NORMALIZE = False
    DISTRIBUTION = 'all'
    MATCHED = False
    SCALE = 'all'
    MINUS_ONE = False
    # TARGET = 'join_selectivity'
    # TARGET = 'mbr_tests_selectivity'
    TARGET = 'best_algorithm'
    # Descriptors
    drop_columns_feature_set1 = ['dataset1', 'dataset2', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y',
                                 'join_selectivity', 'mbr_tests_selectivity', 'E0_x', 'E2_x', 'total_area_x', 'total_margin_x',
                                 'total_overlap_x', 'size_std_x', 'block_util_x', 'total_blocks_x', 'total_area_y', 'total_margin_y',
                                 'total_overlap_y', 'size_std_y', 'block_util_y', 'total_blocks_y', 'intersection_area1', 'intersection_area2', 'jaccard_similarity',
                                 'cardinality_x', 'cardinality_y', 'e0', 'e2']
    # Descriptors + histograms
    drop_columns_feature_set2 = ['dataset1', 'dataset2', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y',
                                 'join_selectivity', 'mbr_tests_selectivity', 'total_area_x', 'total_margin_x',
                                 'total_overlap_x', 'size_std_x', 'block_util_x', 'total_blocks_x', 'total_area_y', 'total_margin_y',
                                 'total_overlap_y', 'size_std_y', 'block_util_y', 'total_blocks_y', 'cardinality_x', 'cardinality_y']
    # Descriptors + histograms + partitioning features
    drop_columns_feature_set3 = ['dataset1', 'dataset2', 'x1_x', 'y1_x', 'x2_x', 'y2_x', 'x1_y', 'y1_y', 'x2_y', 'y2_y',
                                 'join_selectivity', 'mbr_tests_selectivity', 'cardinality_x', 'cardinality_y']
    DROP_COLUMNS = drop_columns_feature_set3

    def __init__(self, model_name):
        self.params = {
            "task": "train",
            "num_leaves": 31,
            "min_data_in_leaf": 1,
            "min_sum_hessian_in_leaf": 100,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1, 2, 3, 4],
            "learning_rate": .1,
            "num_threads": 2
        }

    def train(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
              histogram_path=None) -> None:
        """
        Train a classification model for spatial join cost estimator, then save the trained model to file
        """

        # Extract train and test data, but only use train data
        X_train, y_train = datasets.load_data(tabular_path, RankingModel.TARGET, RankingModel.DROP_COLUMNS)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

        query_train = [X_train.shape[0]]
        query_val = [X_val.shape[0]]

        gbm = lgb.LGBMRanker()
        model = gbm.fit(X_train, y_train, group=query_train,
                eval_set=[(X_val, y_val)], eval_group=[query_val],
                eval_at=[1, 2], early_stopping_rounds=50)

        # Fit and save the model
        # model = self.rnk_model.fit(X_train, y_train)
        pickle.dump(model, open(model_path, 'wb'))

    def test(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
             histogram_path=None) -> (float, float, float, float):
        """
        Evaluate the accuracy metrics of a trained  model for spatial join cost estimator
        :return mean_squared_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, mean_absolute_error
        """

        # Extract train and test data, but only use test data
        # X_train, y_train, X_test, y_test = datasets.load_tabular_features_hadoop(RegressionModel.DISTRIBUTION, RegressionModel.MATCHED, RegressionModel.SCALE, RegressionModel.MINUS_ONE)
        # X_train, y_train, X_test, y_test = datasets.load_tabular_features(join_result_path, tabular_path, RegressionModel.NORMALIZE, RegressionModel.MINUS_ONE, RegressionModel.TARGET)
        X_test, y_test = datasets.load_data(tabular_path, RankingModel.TARGET, RankingModel.DROP_COLUMNS)

        # Load the model and use it for prediction
        loaded_model = pickle.load(open(model_path, 'rb'))
        y_pred = loaded_model.predict(X_test)

        # TODO: delete this dumping action. This is just for debugging
        test_df = pd.DataFrame()
        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred
        test_df.to_csv('data/temp/test_df.csv')

        # Compute accuracy metrics
        # ndcg = metrics.ndcg_score(y_test, y_pred)
        # acc = metrics.accuracy_score(y_test, y_pred)
        # print('Accuracy:', metrics.accuracy_score(y_test, y_pred))
        ndcg = 0
        return ndcg, ndcg, ndcg, ndcg
