'''
This script is intended to compare results with Yanovich's paper
It is ran on the NCSLGR corpus, with FLS, FS and DS
'''

from models.data_utils import *
from models.model_utils import *
from models.train_model import *
from models.perf_utils import *

import math
import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')

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
separation=50
dropout=0
rnn_number=1
mlp_layers_number=0
rnn_hidden_units=5
learning_rate=0.001
earlyStopping=True
save='best'
saveMonitor='val_accYanovichK'
saveMonitorMode='max'
saveBestName='Yanovich'
reduceLrOnPlateau=True
reduceLrMonitor='val_accYanovichK'
reduceLrMonitorMode='max'
reduceLrPatience=7
reduceLrFactor=0.8
optimizer='rms'
metrics=['acc', accYanovichK]


# Data split
fractionValid = 0.2
fractionTest = 0.2
videosToDelete = ['dorm_prank_1053_small_0_1.mov', 'DSP_DeadDog.mov', 'DSP_Immigrants.mov', 'DSP_Trip.mov']
lengthCriterion = 300
includeLong=True
includeShort=True

# Wolf metrics
stepWolf=0.01

#classWeights = np.array([1, 1, 1, 1])

## GET VIDEO INDICES
idxTrain, idxValid, idxTest = getVideoIndicesSplitNCSLGR(fractionValid=fractionValid,
                                                         fractionTest=fractionTest,
                                                         videosToDelete=videosToDelete,
                                                         lengthCriterion=lengthCriterion,
                                                         includeLong=True,
                                                         includeShort=True)
features_train, annot_train = get_data_concatenated(corpus,
                                                        'sign_types',
                                                        catNames, catDetails,
                                                        video_indices=idxTrain,
                                                        separation=separation)
features_valid, annot_valid = get_data_concatenated(corpus,
                                                        'sign_types',
                                                        catNames, catDetails,
                                                        video_indices=idxValid,
                                                        separation=separation)
features_test, annot_test = get_data_concatenated(corpus,
                                                        'sign_types',
                                                        catNames, catDetails,
                                                        video_indices=idxTest,
                                                        separation=separation)

classWeights, classWeights_dict = weightVectorImbalancedDataOneHot(annot_train[0, :, :])

#classWeights = np.array([1, 1, 1, 1])
classWeights[0] = 0.01



# A model with 1 output matrix:
# [other, Pointing, Depicting, Lexical]
model = get_model(outputNames,[4],[1],
                    dropout=dropout,
                    rnn_number=rnn_number,
                    rnn_hidden_units=rnn_hidden_units,
                    mlp_layers_number=mlp_layers_number,
                    time_steps=seq_length,
                    learning_rate=learning_rate,
                    optimizer=optimizer,
                    metrics=metrics)

train_model(model,
            features_train,
            annot_train,
            features_valid,
            annot_valid,
            output_class_weights=[classWeights],
            batch_size=batch_size,
            epochs=epochs,
            seq_length=seq_length,
            save=save,
            saveMonitor=saveMonitor,
            saveMonitorMode=saveMonitorMode,
            saveBestName=saveBestName,
            reduceLrOnPlateau=reduceLrOnPlateau,
            reduceLrMonitor=reduceLrMonitor,
            reduceLrMonitorMode=reduceLrMonitorMode,
            reduceLrPatience=reduceLrPatience,
            reduceLrFactor=reduceLrFactor)

# Test
model.load_weights(saveBestName+'-best.hdf5')

#predict_test = np.zeros((annot_test.shape[1],annot_test.shape[2]))
nRound=annot_test.shape[1]//seq_length
timestepsRound = nRound*seq_length
predict_test = model.predict(features_test[:,:timestepsRound,:].reshape(-1, seq_length, features_test.shape[2])).reshape(1, timestepsRound, 4)
predict_test = predict_test[0]
#    predict_test[i*seq_length:(i+1)*seq_length,:]=model.predict(features_test[:,i*seq_length:(i+1)*seq_length,:])[0]

acc = framewiseAccuracy(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True)
accYanovich, accYanovichPerClass = framewiseAccuracyYanovich(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True)
pStarTp, pStarTr, rStarTp, rStarTr, fStarTp, fStarTr = prfStar(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,step=stepWolf)
Ip, Ir, Ipr = integralValues(fStarTp, fStarTr,step=stepWolf)

print('Accuracy : ' + str(acc))
print('Accuracy Yanovich : ' + str(accYanovich))
print('Accuracy Yanovich per class :')
print(accYanovichPerClass)
print('Ip, Ir, Ipr (star) = ' + str(Ip) + ', ' + str(Ir) + ', ' + str(Ipr))
np.savez('reports/corpora/'+corpus+'/compareYanovich/compareYanovich_prf1.npz',pStarTp=pStarTp, pStarTr=pStarTr, rStarTp=rStarTp, rStarTr=rStarTr, fStarTp=fStarTp, fStarTr=fStarTr)

np.savez('reports/corpora/'+corpus+'/compareYanovich/compareYanovich_annot_predict_test.npz',annot=annot_test,predict=predict_test)


t = np.arange(0,1+stepWolf,stepWolf)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(t,pStarTp,label='pStar')
ax.plot(t,rStarTp,label='rStar')
ax.plot(t,fStarTp,label='f1Star')
ax.set_title('tr=0')
ax.set_xlabel('tp')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.legend()
plt.savefig('reports/corpora/'+corpus+'/compareYanovich/compareYanovich_prf1_tp_tr0')
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(t,pStarTr,label='pStar')
ax.plot(t,rStarTr,label='rStar')
ax.plot(t,fStarTr,label='f1Star')
ax.set_title('tp=0')
ax.set_xlabel('tr')
ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.legend()
plt.savefig('reports/corpora/'+corpus+'/compareYanovich/compareYanovich_prf1_tr_tp0')
