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
    # For tensorflow 2, keras in included in tf
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

import data_utils
import model_utils
