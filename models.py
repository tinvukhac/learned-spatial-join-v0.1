from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import concatenate
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import datasets


def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(4, input_dim=dim, activation="relu"))
    model.add(Dense(2, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))

    # return our model
    return model


def create_cnn(width, height, depth, filters=(4, 8, 16), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    input_shape = (height, width, depth)
    chan_dim = -1

    # define the model input
    inputs = Input(shape=input_shape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chan_dim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(8)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chan_dim)(x)
    x = Dropout(0.5)(x)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model


def run(tabular_path, histogram_path, join_result_path, model_path, model_weights_path, is_train=True):
    print ('Training the join cardinality estimator')
    print ('Tabular data: {}'.format(tabular_path))
    print ('Histogram path: {}'.format(histogram_path))
    print ('Join result data: {}'.format(join_result_path))

    target = 'join_selectivity'
    num_rows, num_columns = 16, 16
    tabular_features_df = datasets.load_datasets_feature(tabular_path)
    join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram = datasets.load_join_data(
        tabular_features_df, join_result_path, histogram_path, num_rows, num_columns)

    num_features = len(join_data.columns) - 10

    if is_train:
        train_attributes, test_attributes, ds1_histograms_train, ds1_histograms_test, ds2_histograms_train, ds2_histograms_test, ds_all_histogram_train, ds_all_histogram_test, ds_bops_histogram_train, ds_bops_histogram_test = train_test_split(
            join_data, ds1_histograms, ds2_histograms, ds_all_histogram, ds_bops_histogram, test_size=0.20,
            random_state=42)
        X_train = pd.DataFrame.to_numpy(train_attributes[[i for i in range(num_features)]])
        X_test = pd.DataFrame.to_numpy(test_attributes[[i for i in range(num_features)]])
        y_train = train_attributes[target]
        y_test = test_attributes[target]
    else:
        X_test = pd.DataFrame.to_numpy(join_data[[i for i in range(num_features)]])
        y_test = join_data[target]
        ds_bops_histogram_test = ds_bops_histogram

    mlp = create_mlp(X_test.shape[1], regress=False)
    cnn1 = create_cnn(num_rows, num_columns, 1, regress=False)
    # cnn2 = models.create_cnn(num_rows, num_columns, 1, regress=False)
    # cnn3 = models.create_cnn(num_rows, num_columns, 1, regress=False)

    # combined_input = concatenate([mlp.output, cnn1.output, cnn2.output, cnn3.output])
    combined_input = concatenate([mlp.output, cnn1.output])

    x = Dense(4, activation="relu")(combined_input)
    x = Dense(1, activation="linear")(x)

    # model = Model(inputs=[mlp.input, cnn1.input, cnn2.input, cnn3.input], outputs=x)
    model = Model(inputs=[mlp.input, cnn1.input], outputs=x)

    EPOCHS = 40
    LR = 1e-2
    opt = Adam(lr=LR, decay=LR / EPOCHS)
    model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

    if is_train:
        print ('Training the model')
        model.fit(
            [X_train, ds_bops_histogram_train], y_train,
            validation_data=([X_test, ds_bops_histogram_test], y_test),
            epochs=EPOCHS, batch_size=256)

        model.save(model_path)
        model.save_weights(model_weights_path)
    else:
        print ('Loading the saved model and model weights')
        model = load_model(model_path)
        model.load_weights(model_weights_path)

    print ('Testing')
    y_pred = model.predict([X_test, ds_bops_histogram_test])

    print ('r2 score: {}'.format(r2_score(y_test, y_pred)))

    diff = y_pred.flatten() - y_test
    percent_diff = (diff / y_test)
    abs_percent_diff = np.abs(percent_diff)

    # Compute the mean and standard deviation of the absolute percentage difference
    mean = np.mean(abs_percent_diff)
    std = np.std(abs_percent_diff)

    # NOTICE: mean is the MAPE value, which is the target we want to minimize
    print ('mean = {}, std = {}'.format(mean, std))
