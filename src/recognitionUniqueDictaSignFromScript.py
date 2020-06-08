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
#from matplotlib import pyplot as plt
#plt.switch_backend('agg')

import argparse

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

parser = argparse.ArgumentParser(description='Trains a Keras-TF model for the recognition of a unique type of annotation, on the DictaSign-LSF-v2 corpus')
#group = parser.add_mutually_exclusive_group()
#group.add_argument("-v", "--verbose", action="store_true")
#group.add_argument("-q", "--quiet", action="store_true")

# Output type
parser.add_argument('--outputName', type=str,    default='PT', help='The output type that the model is trained to recognize')
parser.add_argument('--flsBinary',  type=int,    default=1,    help='If the output is FLS, if seen as binary',              choices=[0, 1])
parser.add_argument('--flsKeep',    type=int,    default=[],   help='If the output is FLS, list of FLS indices to consider', nargs='*')

# Training global setting
parser.add_argument('--videoSplitMode',    type=str,    default='manual', choices=['manual', 'auto'], help='Split mode for videos (auto or manually specified)')
parser.add_argument('--fractionValid',     type=float,  default=0.10,                                 help='Fraction of valid data wrt total (if auto split mode)')
parser.add_argument('--fractionTest',      type=float,  default=0.05,                                 help='Fraction of test data wrt total (if auto split mode)')
parser.add_argument('--signerIndependent', type=int,    default=0,        choices=[0, 1],             help='Signer independent train/valid/test random shuffle')
parser.add_argument('--taskIndependent',   type=int,    default=0,        choices=[0, 1],             help='Task independent train/valid/test random shuffle')
parser.add_argument('--sessionsTrain',     type=int,    default=[],       choices=range(2,10),        help='Training session indices',   nargs='*')
parser.add_argument('--sessionsValid',     type=int,    default=[],       choices=range(2,10),        help='Validation session indices', nargs='*')
parser.add_argument('--sessionsTest',      type=int,    default=[],       choices=range(2,10),        help='Test session indices',       nargs='*')
parser.add_argument('--tasksTrain',        type=int,    default=[],       choices=range(1,10),        help='Training task indices',      nargs='*')
parser.add_argument('--tasksValid',        type=int,    default=[],       choices=range(1,10),        help='Validation task indices',    nargs='*')
parser.add_argument('--tasksTest',         type=int,    default=[],       choices=range(1,10),        help='Test task indices',          nargs='*')
parser.add_argument('--signersTrain',      type=int,    default=[],       choices=range(0,16),        help='Training signer indices',    nargs='*')
parser.add_argument('--signersValid',      type=int,    default=[],       choices=range(0,16),        help='Validation signer indices',  nargs='*')
parser.add_argument('--signersTest',       type=int,    default=[],       choices=range(0,16),        help='Test signer indices',        nargs='*')
parser.add_argument('--idxTrainBypass',    type=int,    default=[],       choices=range(0,94),        help='If you really want to set video indices directly', nargs='*')
parser.add_argument('--idxValidBypass',    type=int,    default=[],       choices=range(0,94),        help='If you really want to set video indices directly', nargs='*')
parser.add_argument('--idxTestBypass',     type=int,    default=[],       choices=range(0,94),        help='If you really want to set video indices directly', nargs='*')
parser.add_argument('--randSeed',          type=int,    default=17,                                   help='Random seed (numpy)')
parser.add_argument('--weightCorrection',  type=float,  default=0,                                    help='Correction for data imbalance (from 0 (no correction) to 1)')


# Fine parameters
parser.add_argument('--seqLength',        type=int,    default=100,       help='Length of sequences')
parser.add_argument('--batchSize',        type=int,    default=200,       help='Batch size')
parser.add_argument('--epochs',           type=int,    default=100,       help='Number of epochs')
parser.add_argument('--separation',       type=int,    default=0,         help='Separation between videos')
parser.add_argument('--dropout',          type=float,  default=0.5,       help='Dropout (0 to 1)')
parser.add_argument('--rnnNumber',        type=int,    default=1,         help='Number of RNN layers')
parser.add_argument('--rnnHiddenUnits',   type=int,    default=50,        help='Number of hidden units in RNN')
parser.add_argument('--mlpLayersNumber',  type=int,    default=0,         help='Number MLP layers after RNN')
parser.add_argument('--convolution',      type=int,    default=1,         help='Whether to use a conv. layer', choices=[0, 1])
parser.add_argument('--convFilt',         type=int,    default=200,       help='Number of convolution kernels')
parser.add_argument('--convFiltSize',     type=int,    default=3,         help='Size of convolution kernels')
parser.add_argument('--learningRate',     type=float,  default=0.001,     help='Learning rate')
parser.add_argument('--optimizer',        type=str,    default='rms',     help='Training optimizer',           choices=['rms', 'ada', 'sgd'])
parser.add_argument('--earlyStopping',    type=int,    default=0,         help='Early stopping',               choices=[0, 1])
parser.add_argument('--redLrOnPlat',      type=int,    default=0,         help='Reduce l_rate on plateau',     choices=[0, 1])
parser.add_argument('--redLrMonitor',     type=str,    default='val_f1K', help='Metric for l_rate reduction')
parser.add_argument('--redLrMonitorMode', type=str,    default='max',     help='Mode for l_rate reduction',    choices=['min', 'max'])
parser.add_argument('--redLrPatience',    type=int,    default=10,        help='Patience before l_rate reduc')
parser.add_argument('--redLrFactor',      type=float,  default=0.5,       help='Factor for each l_rate reduc')

# save data and monitor best
parser.add_argument('--saveModel',       type=str,    default='best',    help='Whether to save only best model, or all, or none', choices=['no', 'best', 'all'])
parser.add_argument('--saveBestMonitor', type=str,    default='val_f1K', help='What metric to decide best model')
parser.add_argument('--saveBestMonMode', type=str,    default='max',     help='Mode to define best',                              choices=['min', 'max'])

# Metrics
parser.add_argument('--stepWolf',        type=float,  default=0.1,       help='Step between Wolf metric eval points',             choices=['rms', 'ada', 'sgd'])

args = parser.parse_args()

# Random initilialization
np.random.seed(args.randSeed)

## PARAMETERS
corpus = 'DictaSign'

# Output type
outputName = args.outputName#'PT'
flsBinary  = bool(args.flsBinary)#True
flsKeep    = args.flsKeep#[]

# Training global setting
videoSplitMode    = args.videoSplitMode
fractionValid     = args.fractionValid
fractionTest      = args.fractionTest
signerIndependent = bool(args.signerIndependent)#False
taskIndependent   = bool(args.taskIndependent)
sessionsTrain     = args.sessionsTrain#[2,3,4,5,6,7,8]
sessionsValid     = args.sessionsValid#[9]
sessionsTest      = args.sessionsTest#[7] # session 7
tasksTrain        = args.tasksTrain#[2,3,4,5,6,7,8]
tasksValid        = args.tasksValid#[9]
tasksTest         = args.tasksTest#[7] # session 7
signersTrain      = args.signersTrain#[2,3,4,5,6,7,8]
signersValid      = args.signersValid#[9]
signersTest       = args.signersTest#[7] # session 7
idxTrainBypass    = args.idxTrainBypass
idxValidBypass    = args.idxValidBypass
idxTestBypass     = args.idxTestBypass
weightCorrection  = args.weightCorrection

# Fine parameters
seq_length          = args.seqLength
batch_size          = args.batchSize
epochs              = args.epochs
separation          = args.separation
dropout             = args.dropout
rnn_number          = args.rnnNumber
rnn_hidden_units    = args.rnnHiddenUnits
mlp_layers_number   = args.mlpLayersNumber
convolution         = bool(args.convolution)
convFilt            = args.convFilt
convFiltSize        = args.convFiltSize
learning_rate       = args.learningRate
optimizer           = args.optimizer
earlyStopping       = bool(args.earlyStopping)
reduceLrOnPlateau   = bool(args.redLrOnPlat)#False
reduceLrMonitor     = args.redLrMonitor
reduceLrMonitorMode = args.redLrMonitorMode
reduceLrPatience    = args.redLrPatience
reduceLrFactor      = args.redLrFactor

# save data and monitor best
save            = args.saveModel
saveMonitor     = args.saveBestMonitor
saveMonitorMode = args.saveBestMonMode

# Metrics
stepWolf     = args.stepWolf#0.1
metrics      = ['acc',  f1K,   precisionK,   recallK]
metricsNames = ['acc', 'f1K', 'precisionK', 'recallK']


saveBestName='recognitionUniqueDictaSign'+outputName


## GET VIDEO INDICES
if len(idxTrainBypass) + len(idxValidBypass) + len(idxTestBypass) > 0:
    idxTrain = np.array(idxTrainBypass)
    idxValid = np.array(idxValidBypass)
    idxTest  = np.array(idxTestBypass)
else:
    idxTrain, idxValid, idxTest = getVideoIndicesSplitDictaSign(sessionsTrain,
                                                                sessionsValid,
                                                                sessionsTest,
                                                                tasksTrain,
                                                                tasksValid,
                                                                tasksTest,
                                                                signersTrain,
                                                                signersValid,
                                                                signersTest,
                                                                signerIndependent,
                                                                taskIndependent,
                                                                videoSplitMode,
                                                                fractionValid,
                                                                fractionTest,
                                                                checkSplits=True,
                                                                checkSets=True)


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
features_test, annot_test   = get_data_concatenated(corpus=corpus,
                                                    output_form=output_form,
                                                    output_names_final=[outputName],
                                                    output_categories_or_names_original=output_categories_or_names_original,
                                                    video_indices=idxTest,
                                                    separation=separation)

nClasses = annot_train.shape[2]

classWeightsCorrected, _ = weightVectorImbalancedDataOneHot(annot_train[0, :, :])
classWeightsNotCorrected = np.ones(nClasses)
classWeightFinal         = weightCorrection*classWeightsCorrected + (1-weightCorrection)*classWeightsNotCorrected



model = get_model([outputName],[nClasses],[1],
                    dropout=dropout,
                    rnn_number=rnn_number,
                    rnn_hidden_units=rnn_hidden_units,
                    mlp_layers_number=mlp_layers_number,
                    conv=convolution,
                    conv_filt=convFilt,
                    conv_ker=convFiltSize,
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

print('On valid: ')
nRound=annot_valid.shape[1]//seq_length
timestepsRound = nRound*seq_length
predict_valid = model.predict(features_valid[:,:timestepsRound,:].reshape(-1, seq_length, features_valid.shape[2])).reshape(1, timestepsRound, nClasses)
predict_valid = predict_valid[0]
#    predict_valid[i*seq_length:(i+1)*seq_length,:]=model.predict(features_valid[:,i*seq_length:(i+1)*seq_length,:])[0]

acc = framewiseAccuracy(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True)
#accYanovich, accYanovichPerClass = framewiseAccuracyYanovich(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True)
pStarTp, pStarTr, rStarTp, rStarTr, fStarTp, fStarTr = prfStar(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,step=stepWolf)
Ip, Ir, Ipr = integralValues(fStarTp, fStarTr,step=stepWolf)

print('Accuracy : ' + str(acc))
#print('Accuracy Yanovich : ' + str(accYanovich))
#print('Accuracy Yanovich per class :')
#print(accYanovichPerClass)
print('Ip, Ir, Ipr (star) = ' + str(Ip) + ', ' + str(Ir) + ', ' + str(Ipr))
np.savez('reports/corpora/'+corpus+'/recognitionUnique/'+outputName+'_prf1.npz',pStarTp=pStarTp, pStarTr=pStarTr, rStarTp=rStarTp, rStarTr=rStarTr, fStarTp=fStarTp, fStarTr=fStarTr)

print('P*(0,0) = ' + str(pStarTp[0]))
print('R*(0,0) = ' + str(rStarTp[0]))
print('F1*(0,0) = ' + str(fStarTp[0]))

margin = 0
print('margin = ' + str(margin))
oldP, oldR, oldF1 = oldPRF1(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
oldPadapted, oldRadapted, oldF1adapted = oldPRF1adapted(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
print('P R F1')
print('Old: ' + str(oldP) + ' ' + str(oldR) + ' ' + str(oldF1))
print('Old-adapted: ' + str(oldPadapted) + ' ' + str(oldRadapted) + ' ' + str(oldF1adapted))
print('margin unit: ' + str(marginUnitP) + ' ' + str(marginUnitR) + ' ' + str(marginUnitF1))

margin = 10
print('margin = ' + str(margin))
oldP, oldR, oldF1 = oldPRF1(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
oldPadapted, oldRadapted, oldF1adapted = oldPRF1adapted(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
print('P R F1')
print('Old: ' + str(oldP) + ' ' + str(oldR) + ' ' + str(oldF1))
print('Old-adapted: ' + str(oldPadapted) + ' ' + str(oldRadapted) + ' ' + str(oldF1adapted))
print('margin unit: ' + str(marginUnitP) + ' ' + str(marginUnitR) + ' ' + str(marginUnitF1))

margin = 30
print('margin = ' + str(margin))
oldP, oldR, oldF1 = oldPRF1(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
oldPadapted, oldRadapted, oldF1adapted = oldPRF1adapted(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
print('P R F1')
print('Old: ' + str(oldP) + ' ' + str(oldR) + ' ' + str(oldF1))
print('Old-adapted: ' + str(oldPadapted) + ' ' + str(oldRadapted) + ' ' + str(oldF1adapted))
print('margin unit: ' + str(marginUnitP) + ' ' + str(marginUnitR) + ' ' + str(marginUnitF1))

margin = 50
print('margin = ' + str(margin))
oldP, oldR, oldF1 = oldPRF1(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
oldPadapted, oldRadapted, oldF1adapted = oldPRF1adapted(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_valid[0,:nRound*seq_length,:],predict_valid[:nRound*seq_length,:],True,True,margin)
print('P R F1')
print('Old: ' + str(oldP) + ' ' + str(oldR) + ' ' + str(oldF1))
print('Old-adapted: ' + str(oldPadapted) + ' ' + str(oldRadapted) + ' ' + str(oldF1adapted))
print('margin unit: ' + str(marginUnitP) + ' ' + str(marginUnitR) + ' ' + str(marginUnitF1))



print('On test:')
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

print('P*(0,0) = ' + str(pStarTp[0]))
print('R*(0,0) = ' + str(rStarTp[0]))
print('F1*(0,0) = ' + str(fStarTp[0]))

margin = 0
print('margin = ' + str(margin))
oldP, oldR, oldF1 = oldPRF1(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
oldPadapted, oldRadapted, oldF1adapted = oldPRF1adapted(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
print('P R F1')
print('Old: ' + str(oldP) + ' ' + str(oldR) + ' ' + str(oldF1))
print('Old-adapted: ' + str(oldPadapted) + ' ' + str(oldRadapted) + ' ' + str(oldF1adapted))
print('margin unit: ' + str(marginUnitP) + ' ' + str(marginUnitR) + ' ' + str(marginUnitF1))

margin = 10
print('margin = ' + str(margin))
oldP, oldR, oldF1 = oldPRF1(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
oldPadapted, oldRadapted, oldF1adapted = oldPRF1adapted(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
print('P R F1')
print('Old: ' + str(oldP) + ' ' + str(oldR) + ' ' + str(oldF1))
print('Old-adapted: ' + str(oldPadapted) + ' ' + str(oldRadapted) + ' ' + str(oldF1adapted))
print('margin unit: ' + str(marginUnitP) + ' ' + str(marginUnitR) + ' ' + str(marginUnitF1))

margin = 30
print('margin = ' + str(margin))
oldP, oldR, oldF1 = oldPRF1(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
oldPadapted, oldRadapted, oldF1adapted = oldPRF1adapted(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
print('P R F1')
print('Old: ' + str(oldP) + ' ' + str(oldR) + ' ' + str(oldF1))
print('Old-adapted: ' + str(oldPadapted) + ' ' + str(oldRadapted) + ' ' + str(oldF1adapted))
print('margin unit: ' + str(marginUnitP) + ' ' + str(marginUnitR) + ' ' + str(marginUnitF1))

margin = 50
print('margin = ' + str(margin))
oldP, oldR, oldF1 = oldPRF1(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
oldPadapted, oldRadapted, oldF1adapted = oldPRF1adapted(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_test[0,:nRound*seq_length,:],predict_test[:nRound*seq_length,:],True,True,margin)
print('P R F1')
print('Old: ' + str(oldP) + ' ' + str(oldR) + ' ' + str(oldF1))
print('Old-adapted: ' + str(oldPadapted) + ' ' + str(oldRadapted) + ' ' + str(oldF1adapted))
print('margin unit: ' + str(marginUnitP) + ' ' + str(marginUnitR) + ' ' + str(marginUnitF1))


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
