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
corpus = 'DictaSign'
outputName = 'PT'
flsBinary = True
flsKeep = []
signerIndependent=False
batch_size=500
epochs=100
seq_length=100
separation=0
dropout=0.3
rnn_number=1
mlp_layers_number=0
rnn_hidden_units=100
learning_rate=0.005
earlyStopping=True
saveBest=True
saveMonitor='val_f1K'
saveMonitorMode='max'
saveBestName='recognitionUniqueDictaSign'+outputName
reduceLrOnPlateau=False
reduceLrMonitor='val_loss'
reduceLrMonitorMode='min'
reduceLrPatience=3
reduceLrFactor=0.5
optimizer='rms'
metrics=['acc', f1K, precisionK, recallK]


# Data split
sessionsTrain=[2,3,4,5,6,9]
sessionsValid=[8]
sessionsTest=[7] # session 7
tasksTrain=[]
tasksValid=[]
tasksTest=[]
signersTrain=[]
signersValid=[]
signersTest=[]



# Wolf metrics
stepWolf=0.1

#classWeights = np.array([1, 1, 1, 1])

## GET VIDEO INDICES
idxTrain, idxValid, idxTest = getVideoIndicesSplitDictaSign([sessionsTrain,sessionsValid,sessionsTest],
                                                            [tasksTrain,tasksValid,tasksTest],
                                                            [signersTrain,signersValid,signersTest])

if outputName=='fls' and not flsBinary:
    output_form='mixed'
    output_categories_or_names_original=[flsKeep]
else:
    output_form='sign_types'
    output_categories_or_names_original=[[outputName]]
features_train, annot_train = get_data_concatenated(corpus=corpus,
                                                    output_form=output_form,
                                                    output_names_final=[outputName],
                                                    output_categories_or_names_original=output_categories_or_names_original,
                                                    video_indices=idxTrain,
                                                    separation=separation)
features_valid, annot_valid = get_data_concatenated(corpus=corpus,
                                                    output_form=output_form,
                                                    output_names_final=[outputName],
                                                    output_categories_or_names_original=output_categories_or_names_original,
                                                    video_indices=idxValid,
                                                    separation=separation)
features_test, annot_test = get_data_concatenated(corpus=corpus,
                                                    output_form=output_form,
                                                    output_names_final=[outputName],
                                                    output_categories_or_names_original=output_categories_or_names_original,
                                                    video_indices=idxTest,
                                                    separation=separation)


#classWeights, classWeights_dict = weightVectorImbalancedDataOneHot(annot_train[0, :, :])

classWeights = np.array([1, 1])
#classWeights[0] = 0.01


model = get_model([outputName],[2],[1],
                    output_class_weights=[classWeights],
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
            saveBest=saveBest,
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
predict_test = model.predict(features_test[:,:timestepsRound,:].reshape(-1, seq_length, features_test.shape[2])).reshape(1, timestepsRound, 2)
predict_test = predict_test[0]
#    predict_test[i*seq_length:(i+1)*seq_length,:]=model.predict(features_test[:,i*seq_length:(i+1)*seq_length,:])[0]

acc = framewiseAccuracy(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True)
#accYanovich, accYanovichPerClass = framewiseAccuracyYanovich(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True)
pStarTp, pStarTr, rStarTp, rStarTr, fStarTp, fStarTr = prfStar(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,step=stepWolf)
Ip, Ir, Ipr = integralValues(fStarTp, fStarTr,step=stepWolf)

print('Accuracy : ' + str(acc))
#print('Accuracy Yanovich : ' + str(accYanovich))
#print('Accuracy Yanovich per class :')
#print(accYanovichPerClass)
print('Ip, Ir, Ipr (star) = ' + str(Ip) + ', ' + str(Ir) + ', ' + str(Ipr))
np.savez('reports/corpora/'+corpus+'/recognitionUnique/'+outputName+'_prf1.npz',pStarTp=pStarTp, pStarTr=pStarTr, rStarTp=rStarTp, rStarTr=rStarTr, fStarTp=fStarTp, fStarTr=fStarTr)

np.savez('reports/corpora/'+corpus+'/recognitionUnique/'+outputName+'_annot_predict_test.npz',annot=annot_test,predict=predict_test)


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
plt.savefig('reports/corpora/'+corpus+'/recognitionUnique/'+outputName+'_prf1_tp_tr0')
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
plt.savefig('reports/corpora/'+corpus+'/recognitionUnique/'+outputName+'_prf1_tr_tp0')
