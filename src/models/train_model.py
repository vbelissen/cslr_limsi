import numpy as np
from pandas import DataFrame
import os, os.path
import csv
from itertools import groupby
from time import time
import sys

import tensorflow as tf
v0 = tf.__version__[0]
if v0 == '2':
    # For tensorflow 2, keras is included in tf
    import tensorflow.keras.backend as K
    from tensorflow.keras import optimizers
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input, Dense, Conv1D, Dropout, GlobalAveragePooling1D, multiply
    from tensorflow.python.keras.layers.core import *
    from tensorflow.keras.models import *
    from tensorflow.keras.utils import to_categorical, plot_model
elif v0 == '1':
    #For tensorflow 1.2.0
    import keras.backend as K
    from keras import optimizers
    from keras.callbacks import TensorBoard
    from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input, Dense, Conv1D, Dropout, GlobalAveragePooling1D, merge
    from keras.layers.core import *
    from keras.models import *
    from keras.utils import to_categorical, plot_model
else:
    sys.exit('Tensorflow version should be 1.X or 2.X')


def generator(features, annot, batch_size, seq_length, output_form):
    """
    Generator function for batch training models
    """
    total_length_round = (features.shape[1]//seq_length)*seq_length
    batch_size_time = np.min([batch_size*seq_length, total_length_round])
    feature_number = features.shape[2]

    batch_features = np.zeros((1, batch_size_time, feature_number))

    if output_form == 'mixed':
        batch_labels = []
        labels_number = len(annot)
        labels_shape = []
        for i_label_cat in range(labels_number):
            labels_shape.append(annot[i_label_cat].shape[2])
            batch_labels.append(np.zeros((1, batch_size_time, labels_shape[i_label_cat])))
    elif output_form == 'sign_types':
        labels_shape = annot.shape[2]
        batch_labels = np.zeros((1, batch_size_time, labels_shape))
    else:
        sys.exit('Wrong annotation format')

    while True:
        # Random start
        random_ini = np.random.randint(0, total_length_round)
        end = random_ini + batch_size_time
        end_modulo = np.mod(end, total_length_round)

        # Fill in batch features
        batch_features = batch_features.reshape(1, batch_size_time, feature_number)
        if end <= total_length_round:
            batch_features = features[0, random_ini:end, :].reshape(-1, seq_length, feature_number)
        else:
            batch_features[0, :(total_length_round - random_ini), :] = features[0, random_ini:total_length_round, :]
            batch_features[0, (total_length_round - random_ini):, :] = features[0, 0:end_modulo, :]
            batch_features = batch_features.reshape(-1, seq_length, feature_number)

        # Fill in batch annotations
        if output_form == 'mixed':
            for i_label_cat in range(labels_number):
                batch_labels[i_label_cat] = batch_labels[i_label_cat].reshape(1, batch_size_time, labels_shape[i_label_cat])
                if end <= total_length_round:
                    batch_labels[i_label_cat] = annot[i_label_cat][0, random_ini:end, :].reshape(-1, seq_length, labels_shape[i_label_cat])
                else:
                    batch_labels[i_label_cat][0, :(total_length_round - random_ini), :] = annot[i_label_cat][0, random_ini:total_length_round, :]
                    batch_labels[i_label_cat][0, (total_length_round - random_ini):, :] = annot[i_label_cat][0, 0:end_modulo, :]
                    batch_labels[i_label_cat] = batch_labels[i_label_cat].reshape(-1, seq_length, labels_shape[i_label_cat])
        elif output_form == 'sign_types':
            batch_labels = batch_labels.reshape(1, batch_size_time, labels_shape)
            if end <= total_length_round:
                batch_labels = annot[0, random_ini:end, :].reshape(-1, seq_length, labels_shape)
            else:
                batch_labels[0, :(total_length_round - random_ini), :] = annot[0, random_ini:total_length_round, :]
                batch_labels[0, (total_length_round - random_ini):, :] = annot[0, 0:end_modulo, :]
                batch_labels = batch_labels.reshape(-1, seq_length, labels_shape)

        yield batch_features, batch_labels


def train_model(model, features_train, annot_train, features_valid, annot_valid, batch_size, epochs, seq_length):
    """
        Trains a keras model.

        Inputs:
            model: keras model
            features_train: numpy array of features [1, time_steps_train, features]
            annot_train: either list of annotation arrays (output_form: 'mixed')
                            or one binary array (output_form: 'sign_types')
            features_valid: numpy array of features [1, time_steps_valid, features]
            annot_valid: either list of annotation arrays (output_form: 'mixed')
                             or one binary array (output_form: 'sign_types')
            batch_size

        Outputs:
            ?
    """
    if type(annot_train) == list:
        output_form = 'mixed'
    elif type(annot_train) == np.ndarray:
        output_form = 'sign_types'
    else:
        sys.exit('Wrong annotation format')

    if output_form == 'mixed':
        annot_categories_number = len(annot_train)

    time_steps_train = features_train.shape[1]
    time_steps_valid = features_valid.shape[1]

    total_length_train_round = (features_train.shape[1] // seq_length) * seq_length
    batch_size_time = np.min([batch_size * seq_length, total_length_train_round])

    hist = model.fit_generator(generator(features_train, annot_train, batch_size, seq_length, output_form),
                               epochs=epochs,
                               steps_per_epoch=np.ceil(time_steps_train/batch_size_time),
                               validation_data=generator(features_valid, annot_valid, batch_size, seq_length, output_form),
                               validation_steps=1)

