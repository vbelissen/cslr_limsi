from models.data_utils import *
from models.model_utils import *
from models.train_model import *
from models.perf_utils import *

import math
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
v0 = tf.__version__[0]
if v0 == '2':
    # For tensorflow 2, keras is included in tf
    from tensorflow.keras.models import *
elif v0 == '1':
    #For tensorflow 1.2.0
    from keras.models import *

np.random.seed(17)

## PARAMETERS
# Categories names
corpus = 'NCSLGR'
outputNames = ['fls-FS-DS']
catNames = ['fls', 'FS', 'DS']
catDetails = [
              ['lexical_with_ns_not_fs'],
              ['fingerspelling', 'fingerspelled_loan_signs'],
              [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
             ]
batch_size=200
epochs=200
seq_length=100
separation=0
dropout=0
rnn_number=1
mlp_layers_number=0
rnn_hidden_units=5
learning_rate=0.001
earlyStopping=True
saveBest=True
saveBestName='Yanovich'
reduceLrOnPlateau=True


# Data split
fractionValid = 0.20
fractionTest = 0.2
videosToDelete = ['dorm_prank_1053_small_0_1.mov', 'DSP_DeadDog.mov', 'DSP_Immigrants.mov', 'DSP_Trip.mov']
lengthCriterion = 300
includeLong=True
includeShort=True


## GET VIDEO INDICES
idxTrain, idxValid, idxTest = getVideoIndicesSplitNCSLGR(fractionValid=fractionValid,
                                                         fractionTest=fractionTest,
                                                         videosToDelete=videosToDelete,
                                                         lengthCriterion=lengthCriterion,
                                                         includeLong=True,
                                                         includeShort=True)


# A model with 1 output matrix:
# [other, Pointing, Depicting, Lexical]
model_2 = get_model(outputNames,[4],[1],
                    dropout=dropout,
                    rnn_number=rnn_number,
                    rnn_hidden_units=rnn_hidden_units,
                    mlp_layers_number=mlp_layers_number,
                    time_steps=seq_length,
                    learning_rate=learning_rate)
features_2_train, annot_2_train = get_data_concatenated(corpus,
                                                        'sign_types',
                                                        catNames, catDetails,
                                                        video_indices=idxTrain,
                                                        separation=separation)
features_2_valid, annot_2_valid = get_data_concatenated(corpus,
                                                        'sign_types',
                                                        catNames, catDetails,
                                                        video_indices=idxValid,
                                                        separation=separation)
train_model(model_2,
            features_2_train,
            annot_2_train,
            features_2_valid,
            annot_2_valid,
            batch_size=batch_size,
            epochs=epochs,
            seq_length=seq_length,
            saveBest=saveBest,
            saveBestName=saveBestName,
            reduceLrOnPlateau=reduceLrOnPlateau)



# Test
model_2.load_weights('Yanovich-best.hdf5')

predict_2_test = np.zeros((annot_2_test.shape[1],annot_2_test.shape[2]))
nRound=annot_2_test.shape[1]//seq_length
for i in range(nRound):
    predict_2_test[i*seq_length:(i+1)*seq_length,:]=model_2.predict(features_2_test[:,i*seq_length:(i+1)*seq_length,:])[0]

acc = framewiseAccuracy(annot_2_test[0,:nRound*seq_length,:],predict_2_test[:nRound*seq_length,:],True,True)
accYanovich, accYanovichPerClass = framewiseAccuracyYanovich(annot_2_test[0,:nRound*seq_length,:],predict_2_test[:nRound*seq_length,:],True)
pStarTp, pStarTr, rStarTp, rStarTr, fStarTp, fStarTr = prfStar(annot_2_test[0,:nRound*seq_length,:],predict_2_test[:nRound*seq_length,:],True,True)
P,R,F1 = integralValues(fStarTp, fStarTr)

print('Accuracy : ' + str(acc))
print('Accuracy Yanovich : ' + str(accYanovich))
print('Accuracy Yanovich per class :')
print(accYanovichPerClass)
print('P, R, F1 (star) = ' + star(P) + ', ' + star(R) + ', ' + star(F1))
