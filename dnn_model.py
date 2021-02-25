import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense
from keras.losses import mean_squared_logarithmic_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn import metrics

import datasets
from model_interface import ModelInterface

BATCH_SIZE = 256
EPOCHS = 100
VAL_SIZE = 0.2


class DNNModel(ModelInterface):
    NORMALIZE = False
    DISTRIBUTION = 'all'
    MATCHED = True
    SCALE = 'Small'
    MINUS_ONE = False

    def train(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
              histogram_path=None) -> None:
        """
        Train a regression model for spatial join cost estimator, then save the trained model to file
        """

        # Extract train and test data, but only use train data
        X_train, y_train, X_test, y_test = datasets.load_tabular_features_hadoop(DNNModel.DISTRIBUTION, DNNModel.MATCHED, DNNModel.SCALE, DNNModel.MINUS_ONE)
        # X_train, y_train, X_test, y_test = datasets.load_tabular_features(join_result_path, tabular_path, DNNModel.NORMALIZE)

        # Define a sequential deep neural network model
        model = Sequential()
        model.add(Dense(16, input_dim=X_train.shape[1], activation='relu'))
        # model.add(Dense(4, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile and fit the model
        LR = 1e-2
        opt = Adam(lr=LR)
        model.compile(
            optimizer=opt, loss=mean_absolute_percentage_error, metrics=[mean_absolute_error, mean_squared_error]
        )
        early_stopping = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        history = model.fit(
            x=X_train,
            y=y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            callbacks=[early_stopping],
            validation_split=VAL_SIZE,
        )

        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()

        test_loss = model.evaluate(X_test, y_test)
        print(test_loss)
        # print('Accuracy: %.2f' % (test_loss * 100))

        y_pred = model.predict(X_test)
        # TODO: delete this dumping action. This is just for debugging
        test_df = pd.DataFrame()
        test_df['y_test'] = y_test
        test_df['y_pred'] = y_pred
        test_df.to_csv('data/temp/test_df.csv')

        # Convert back to 1 - y if need
        if DNNModel.MINUS_ONE:
            y_test, y_pred = 1 - y_test, 1 - y_pred

        # Compute accuracy metrics
        mse = metrics.mean_squared_error(y_test, y_pred)
        mape = metrics.mean_absolute_percentage_error(y_test, y_pred)
        msle = np.mean(mean_squared_logarithmic_error(y_test, y_pred))
        mae = metrics.mean_absolute_error(y_test, y_pred)
        print('mae: {}\nmape: {}\nmse: {}\nmlse: {}'.format(mae, mape, mse, msle))
        print('{}\t{}\t{}\t{}'.format(mae, mape, mse, msle))

    def test(self, tabular_path: str, join_result_path: str, model_path: str, model_weights_path=None,
             histogram_path=None) -> (float, float, float, float):
        """
        Evaluate the accuracy metrics of a trained  model for spatial join cost estimator
        :return mean_squared_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, mean_absolute_error
        """
        pass
