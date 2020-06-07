import numpy as np
from pandas import DataFrame
import os, os.path
import csv
from itertools import groupby
from time import time
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
v0 = tf.__version__[0]
if v0 == '2':
    # For tensorflow 2, keras is included in tf
    import tensorflow.keras.backend as K
    from tensorflow.keras import optimizers
    from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input, Dense, Conv1D, Dropout, GlobalAveragePooling1D, multiply
    from tensorflow.python.keras.layers.core import *
    from tensorflow.keras.models import *
    from tensorflow.keras.utils import to_categorical, plot_model
elif v0 == '1':
    #For tensorflow 1.2.0
    import keras.backend as K
    from keras import optimizers
    from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input, Dense, Conv1D, Dropout, GlobalAveragePooling1D, merge
    from keras.layers.core import *
    from keras.models import *
    from keras.utils import to_categorical, plot_model
else:
    sys.exit('Tensorflow version should be 1.X or 2.X')


def generator(features, annot, batch_size, seq_length, output_form, output_class_weights):
    """
    Generator function for batch training models
    """
    total_length_round = (features.shape[1]//seq_length)*seq_length
    batch_size_time = np.min([batch_size*seq_length, total_length_round])
    feature_number = features.shape[2]

    batch_features = np.zeros((1, batch_size_time, feature_number))
    batch_labels_weight = np.zeros((1, batch_size_time))

    if output_class_weights != []:
        if output_form == 'mixed':
            annot_labels_weight = np.ones(1, annot[0].shape[1])
            labels_number = len(annot)
            for i_label_cat in range(labels_number):
                annot_labels_weight_tmp = np.zeros(1, annot[i_label_cat].shape[1])
                nClasses = annot[i_label_cat].shape[2]
                for iClass in range(nClasses):
                    annot_labels_weight_tmp[0, np.argmax(annot[i_label_cat][0,:,:],axis=1)==iClass] = output_class_weights[i_label_cat][iClass]
                annot_labels_weight = annot_labels_weight*annot_labels_weight_tmp
        elif output_form == 'sign_types':
            nClasses = annot.shape[2]
            annot_labels_weight=np.zeros((1, annot.shape[1]))
            for iClass in range(nClasses):
                annot_labels_weight[0, np.argmax(annot[0,:,:],axis=1)==iClass] = output_class_weights[0][iClass]

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

        # Fill in batch weights
        if output_class_weights != []:
            batch_labels_weight = batch_labels_weight.reshape(1, batch_size_time)
            if end <= total_length_round:
                batch_labels_weight = annot_labels_weight[0, random_ini:end].reshape(-1, seq_length)
            else:
                batch_labels_weight[0, :(total_length_round - random_ini)] = annot_labels_weight[0, random_ini:total_length_round]
                batch_labels_weight[0, (total_length_round - random_ini):] = annot_labels_weight[0, 0:end_modulo]
                batch_labels_weight = batch_labels_weight.reshape(-1, seq_length)


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

        if output_class_weights != []:
            yield batch_features, batch_labels, batch_labels_weight
        else:
            yield batch_features, batch_labels


def train_model(model,
                features_train,
                annot_train,
                features_valid,
                annot_valid,
                batch_size,
                epochs,
                seq_length,
                output_class_weights=[],
                earlyStopping=False,
                save='no',
                saveMonitor='val_loss',
                saveMonitorMode='min',
                saveBestName='',
                reduceLrOnPlateau=False,
                reduceLrMonitor='val_loss',
                reduceLrMonitorMode='min',
                reduceLrPatience=7,
                reduceLrFactor=0.8):
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
            output_class_weights: list of vector of weights for each class of each output
            save: for saving the models ('no' or 'best' or 'all')


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

    callbacksPerso = []
    if earlyStopping:
        callbacksPerso.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'))
    if save=='all':
        callbacksPerso.append(ModelCheckpoint(filepath=saveBestName+'-best.hdf5',#+'-epoch-{epoch:02d}-val_loss-{val_loss:.2f}.hdf5',
                                              save_best_only=False,
                                              save_weights_only=False,
                                              monitor=saveMonitor,
                                              mode=saveMonitorMode),
                                              verbose=1)
    elif save=='best':
        callbacksPerso.append(ModelCheckpoint(filepath=saveBestName+'-best.hdf5',#+'-epoch-{epoch:02d}-val_loss-{val_loss:.2f}.hdf5',
                                              save_best_only=True,
                                              save_weights_only=False,
                                              monitor=saveMonitor,
                                              mode=saveMonitorMode),
                                              verbose=1)
    if reduceLrOnPlateau:
        callbacksPerso.append(ReduceLROnPlateau(monitor=reduceLrMonitor, factor=reduceLrFactor, patience=reduceLrPatience, verbose=1, epsilon=1e-4, mode=reduceLrMonitorMode))

    hist = model.fit_generator(generator(features_train, annot_train, batch_size, seq_length, output_form, output_class_weights),
                               epochs=epochs,
                               steps_per_epoch=np.ceil(time_steps_train/batch_size_time),
                               validation_data=generator(features_valid, annot_valid, batch_size, seq_length, output_form, output_class_weights),
                               validation_steps=1,
                               callbacks=callbacksPerso)
