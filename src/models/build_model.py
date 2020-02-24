import numpy as np
from pandas import DataFrame
import os, os.path
import csv
from itertools import groupby
from time import time
import sys

import keras.backend as K
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input, Dense, Conv1D, Dropout, GlobalAveragePooling1D, merge
from keras.layers.core import *
from keras.models import *
from keras.utils import to_categorical, plot_model

from attention.attention_utils import get_activations, get_data_recurrent



def attention_timewise(inputs, single=False, attention_layer_descriptor=''):
    """
        Timewise attention block (Keras API).
        Applies this block to inputs (inputs.shape = (batch_size, time_steps, input_dim)).

        Inputs:
            inputs (Keras layer)
            single (bool): if True, attention is shared across features
            attention_layer_descriptor (string): describes where attention is applied

        Output: A Keras layer
    """
    time_steps = int(inputs.shape[1])
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    if single:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction_'+attention_layer_descriptor)(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec_'+attention_layer_descriptor)(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul_timewise_'+attention_layer_descriptor, mode='mul')
    return output_attention_mul

def attention_featurewise(inputs, single=False, attention_layer_descriptor=''):
    """
        Featurewise attention block (Keras API).
        Applies this block to inputs (inputs.shape = (batch_size, time_steps, input_dim)).

        Inputs:
            inputs (Keras layer)
            single (bool): if True, attention is shared across timesteps
            attention_layer_descriptor (string): describes where attention is applied

        Output: A Keras layer
    """
    input_dim = int(inputs.shape[2])
    a = Dense(input_dim, activation='softmax')(inputs)
    if single:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction_'+attention_layer_descriptor)(a)
        a = RepeatVector(input_dim)(a)
    output_attention_mul = merge([inputs, a], name='attention_mul_featurewise_'+attention_layer_descriptor, mode='mul')
    return output_attention_mul

def get_model(output_names,
              output_classes,
              conv=True,
              conv_filt=200,
              conv_ker=3,
              conv_strides=1,
              rnn_number=2,
              rnn_type='lstm',
              dropout=0,
              attention_in_rnn=False,
              att_in_rnn_single = False,
              att_out_rnn=False,
              att_out_rnn_single=False,
              rnn_return_sequences=True,
              classif_local=True,
              optimizer='rms',
              learning_rate=0.005):
    """
        Returns Keras model

        Inputs:
            output_names: list of outputs (strings)
            output_classes: list of number of classes of each output type
            conv (bool): if True, applies convolution on input
            conv_filt: number of convolution filters
            conv_ker: size of convolution kernel
            conv_strides: number of convolution strides
            rnn_number: number of recurrent layers
            rnn_type: type of recurrent layers (string)
            dropout: how much dropout (0 to 1)
            attention_in_rnn: if True, applies attention layer before recurrent layers
            att_in_rnn_single: single (shared) attention layer or not
            att_out_rnn: if True, applies attention layer after recurrent layers
            att_out_rnn_single: single (shared) attention layer or not
            rnn_return_sequences: if False, only last timestep of recurrent layers is returned
            classif_local (bool): whether classification is for each timestep (local) of globally for the sequence
            optimizer: gradient optimizer type (string)
            learning_rate: learning rate (float)

        Output: A Keras model
    """
    return 0#model