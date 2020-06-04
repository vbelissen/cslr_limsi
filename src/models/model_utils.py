import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

from models.perf_utils import *

def attention_timewise(inputs, time_steps, single=False, attention_layer_descriptor=''):
    """
        Timewise attention block (Keras API).
        Applies this block to inputs (inputs.shape = (batch_size, time_steps, input_dim)).

        Inputs:
            inputs (Keras layer)
            time_steps
            single (bool): if True, attention is shared across features
            attention_layer_descriptor (string): describes where attention is applied

        Output: A Keras layer
    """
    input_dim = int(inputs.shape[-1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    if single:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction_'+attention_layer_descriptor)(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec_'+attention_layer_descriptor)(a)
    if v0 == '2':
        output_attention_mul = multiply([inputs, a_probs], name='attention_mul_timewise_'+attention_layer_descriptor)
    else:
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
    input_dim = int(inputs.shape[-1])
    a = Dense(input_dim, activation='softmax')(inputs)
    if single:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction_'+attention_layer_descriptor)(a)
        a = RepeatVector(input_dim)(a)
    if v0 == '2':
        output_attention_mul = multiply([inputs, a], name='attention_mul_featurewise_'+attention_layer_descriptor)
    else:
        output_attention_mul = merge([inputs, a], name='attention_mul_featurewise_'+attention_layer_descriptor, mode='mul')
    return output_attention_mul

def get_model(output_names,
              output_classes,
              output_weights=[],
              output_class_weights=[],
              conv=True,
              conv_filt=200,
              conv_ker=3,
              conv_strides=1,
              rnn_number=2,
              rnn_type='lstm',
              rnn_hidden_units=55,
              dropout=0,
              att_in_rnn=False,
              att_in_rnn_single = False,
              att_in_rnn_type='timewise',
              att_out_rnn=False,
              att_out_rnn_single=False,
              att_out_rnn_type='timewise',
              rnn_return_sequences=True,
              classif_local=True,
              mlp_layers_number=0,
              mlp_layers_size=30,
              optimizer='rms',
              metrics=['acc'],
              learning_rate=0.005,
              time_steps=100,
              features_number=420,
              print_summary=True):
    """
        Keras recurrent neural network model builder.
        It can include a convolutional layer, attention on the input, several RNN layers, attention on RNN output, additional dense layers.

        Inputs:
            output_names: list of outputs (strings)
            output_classes: list of number of classes of each output type
            output_weights: list of weights for each_output
            output_class_weights: list of vector of weights for each class of each output
            conv (bool): if True, applies convolution on input
            conv_filt: number of convolution filters
            conv_ker: size of convolution kernel
            conv_strides: size of convolution strides
            rnn_number: number of recurrent layers
            rnn_type: type of recurrent layers (string)
            rnn_hidden_units: number of hidden units
            dropout: how much dropout (0 to 1)
            att_in_rnn: if True, applies attention layer before recurrent layers
            att_in_rnn_single: single (shared) attention layer or not
            att_in_rnn_type (string): timewise or featurewise attention layer
            att_out_rnn: if True, applies attention layer after recurrent layers
            att_out_rnn_single: single (shared) attention layer or not
            att_out_rnn_type (string): timewise or featurewise attention layer
            rnn_return_sequences: if False, only last timestep of recurrent layers is returned
            classif_local (bool): whether classification is for each timestep (local) of globally for the sequence
            mlp_layers_number: number of additional dense layers
            mlp_layers_size: size of additional dense layers
            optimizer: gradient optimizer type (string)
            learning_rate: learning rate (float)
            time_steps: length of sequences (int)
            features_number: number of features (int)
            print_summary (bool)

        Output: A Keras model
    """

    # input
    main_input = Input(shape=(time_steps, features_number))
    input_transfo = main_input

    # convolution input
    if conv:
        input_transfo = Conv1D(filters=conv_filt, kernel_size=conv_ker, strides=conv_strides, padding='same', activation='relu')(input_transfo)

    # attention before RNNs
    if att_in_rnn:
        if att_in_rnn_type == 'timewise':
            input_transfo = attention_timewise(input_transfo, time_steps=time_steps, single=att_in_rnn_single, attention_layer_descriptor='before_rnn')
        elif att_in_rnn_type == 'featurewise':
            input_transfo = attention_featurewise(input_transfo, single=att_in_rnn_single, attention_layer_descriptor='before_rnn')
        else:
            sys.exit('Invalid attention type')

    # recurrent layers
    if rnn_number == 1:
        if rnn_type == 'lstm':
            rnn_n = Bidirectional(LSTM(rnn_hidden_units, return_sequences=rnn_return_sequences, dropout=dropout, recurrent_dropout=dropout))(input_transfo)
        else:
            sys.exit('Invalid RNN type')
    elif rnn_number == 2:
        if rnn_type == 'lstm':
            rnn_1 = Bidirectional(LSTM(rnn_hidden_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))(input_transfo)
            rnn_n = Bidirectional(LSTM(rnn_hidden_units, return_sequences=rnn_return_sequences, dropout=dropout, recurrent_dropout=dropout))(rnn_1)
        else:
            sys.exit('Invalid RNN type')
    elif rnn_number >= 3:
        if rnn_type == 'lstm':
            rnn_i = Bidirectional(LSTM(rnn_hidden_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))(input_transfo)
        else:
            sys.exit('Invalid RNN type')
        for i_rnn in range(1, rnn_number - 1):
            if rnn_type == 'lstm':
                rnn_i = Bidirectional(LSTM(rnn_hidden_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout))(rnn_i)
            else:
                sys.exit('Invalid RNN type')
        if rnn_type == 'lstm':
            rnn_n = Bidirectional(LSTM(rnn_hidden_units, return_sequences=rnn_return_sequences, dropout=dropout, recurrent_dropout=dropout))(rnn_i)
        else:
            sys.exit('Invalid RNN type')
    else:
        sys.exit('Invalid RNN number')
    # attention after RNNs
    if att_out_rnn:
        if rnn_return_sequences:
            if att_out_rnn_type == 'timewise':
                rnn_n = attention_timewise(rnn_n, time_steps=time_steps, single=att_out_rnn_single, attention_layer_descriptor='after_rnn')
            elif att_out_rnn_type == 'featurewise':
                rnn_n = attention_featurewise(rnn_n, single=att_out_rnn_single, attention_layer_descriptor='after_rnn')
            else:
                sys.exit('Invalid attention type')
        elif att_out_rnn_type == 'featurewise':
            rnn_n = attention_featurewise(rnn_n, single=att_out_rnn_single, attention_layer_descriptor='after_rnn')
        elif att_out_rnn_type == 'timewise':
            sys.exit('Cannot apply attention after RNN when RNN does not return sequences')
        else:
            sys.exit('Invalid attention type')

    # Final layers
    output_intermed = rnn_n
    output_number = len(output_names)
    output_intermed_list = []
    output_list = []
    for i_output in range(output_number):
        output_intermed_list.append(output_intermed)
    if rnn_return_sequences and not classif_local:
        for i_output in range(output_number):
            output_intermed_list[i_output] = GlobalAveragePooling1D()(output_intermed_list[i_output])
    if rnn_return_sequences and classif_local:
        if mlp_layers_number > 0:
            for i_add in range(mlp_layers_number):
                for i_output in range(output_number):
                    output_intermed_list[i_output] = TimeDistributed(Dense(mlp_layers_size, activation='relu'))(output_intermed_list[i_output])
                if dropout > 0:
                    for i_output in range(output_number):
                        output_intermed_list[i_output] = TimeDistributed(Dropout(dropout))(output_intermed_list[i_output])
        for i_output in range(output_number):
            output_list.append(TimeDistributed(Dense(output_classes[i_output], activation='softmax', name='output_' + output_names[i_output]))(output_intermed_list[i_output]))
    else:
        if mlp_layers_number > 0:
            for i_add in range(mlp_layers_number):
                for i_output in range(output_number):
                    output_intermed_list[i_output] = Dense(mlp_layers_size, activation='relu')(output_intermed_list[i_output])
                if dropout > 0:
                    for i_output in range(output_number):
                        output_intermed_list[i_output] = Dropout(dropout)(output_intermed_list[i_output])
        for i_output in range(output_number):
            output_list.append(Dense(output_classes[i_output], activation='softmax', name='output_' + output_names[i_output])(output_intermed_list[i_output]))

    # Create model
    model = Model(inputs=main_input, outputs=output_list)
    # framewise weights:
    if classif_local:
        weight_mode_sequence = 'temporal'
    else:
        weight_mode_sequence = None
    if optimizer == 'sgd':
        opt = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimizer == 'rms':
        opt = optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=None, decay=0.0)
    elif optimizer == 'ada':
        opt = optimizers.Adagrad(lr=learning_rate, epsilon=None, decay=0.0)
    else:
        sys.exit('Invalid gradient optimizer')
    if output_weights == []:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics, sample_weight_mode=weight_mode_sequence)
    else:
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=metrics, loss_weights=output_weights, sample_weight_mode=weight_mode_sequence)
    if print_summary:
        model.summary()
    return model
