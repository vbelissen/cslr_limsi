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
import pickle
import argparse
import time

import os.path
from os import path

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
parser.add_argument('--outputsList',
                    type=str,
                    default=['fls', 'DS', 'PT', 'FBUOY'],
                    help='The outputs that the model is trained to recognize',
                    nargs='+')
parser.add_argument('--outputsWeightList',
                    type=int,
                    default=[1, 1, 1, 1],
                    help='The output weights in the loss',
                    nargs='+')
#parser.add_argument('--outputName', type=str,    default='PT', help='The output type that the model is trained to recognize')
#parser.add_argument('--flsBinary',  type=int,    default=1,    help='If the output is FLS, if seen as binary',              choices=[0, 1])
#parser.add_argument('--flsKeep',    type=int,    default=[],   help='If the output is FLS, list of FLS indices to consider', nargs='*')
parser.add_argument('--comment',
                    type=str,
                    default='',
                    help='A comment to describe this run')

# Training global setting
parser.add_argument('--videoSplitMode',
                    type=str,
                    default='auto',
                    choices=['manual', 'auto'],
                    help='Split mode for videos (auto or manually specified)')
parser.add_argument('--fractionValid',
                    type=float,
                    default=0.10,
                    help='Fraction of valid data wrt total (if auto split mode)')
parser.add_argument('--fractionTest',
                    type=float,
                    default=0.10,
                    help='Fraction of test data wrt total (if auto split mode)')
parser.add_argument('--signerIndependent',
                    type=int,
                    default=0,
                    choices=[0, 1],
                    help='Signer independent train/valid/test random shuffle')
parser.add_argument('--taskIndependent',
                    type=int,
                    default=0,
                    choices=[0, 1],
                    help='Task independent train/valid/test random shuffle')
parser.add_argument('--excludeTask9',
                    type=int,
                    default=0,
                    choices=[0, 1],
                    help='Whether to exclude task 9')
parser.add_argument('--tasksTrain',
                    type=int,
                    default=[],
                    choices=range(1,10),
                    help='Training task indices',
                    nargs='*')
parser.add_argument('--tasksValid',
                    type=int,
                    default=[],
                    choices=range(1,10),
                    help='Validation task indices',
                    nargs='*')
parser.add_argument('--tasksTest',
                    type=int,
                    default=[],
                    choices=range(1,10),
                    help='Test task indices',
                    nargs='*')
parser.add_argument('--signersTrain',
                    type=int,
                    default=[],
                    choices=range(0,16),
                    help='Training signer indices',
                    nargs='*')
parser.add_argument('--signersValid',
                    type=int,
                    default=[],
                    choices=range(0,16),
                    help='Validation signer indices',
                    nargs='*')
parser.add_argument('--signersTest',
                    type=int,
                    default=[],
                    choices=range(0,16),
                    help='Test signer indices',
                    nargs='*')
parser.add_argument('--idxTrainBypass',
                    type=int,
                    default=[],
                    choices=range(0,94),
                    help='If you really want to set video indices directly',
                    nargs='*')
parser.add_argument('--idxValidBypass',
                    type=int,
                    default=[],
                    choices=range(0,94),
                    help='If you really want to set video indices directly',
                    nargs='*')
parser.add_argument('--idxTestBypass',
                    type=int,
                    default=[],
                    choices=range(0,94),
                    help='If you really want to set video indices directly',
                    nargs='*')
parser.add_argument('--randSeed',
                    type=int,
                    default=17,
                    help='Random seed (numpy)')
parser.add_argument('--weightCorrection',
                    type=float,
                    default=0,
                    help='Correction for data imbalance (from 0 (no correction) to 1)')


# Fine parameters
parser.add_argument('--seqLength',
                    type=int,
                    default=100,
                    help='Length of sequences')
parser.add_argument('--batchSize',
                    type=int,
                    default=200,
                    help='Batch size')
parser.add_argument('--epochs',
                    type=int,
                    default=100,
                    help='Number of epochs')
parser.add_argument('--separation',
                    type=int,
                    default=0,
                    help='Separation between videos')
parser.add_argument('--dropout',
                    type=float,
                    default=0.5,
                    help='Dropout (0 to 1)')
parser.add_argument('--rnnNumber',
                    type=int,
                    default=1,
                    help='Number of RNN layers')
parser.add_argument('--rnnHiddenUnits',
                    type=int,
                    default=50,
                    help='Number of hidden units in RNN')
parser.add_argument('--mlpLayersNumber',
                    type=int,
                    default=0,
                    help='Number MLP layers after RNN')
parser.add_argument('--convolution',
                    type=int,
                    default=1,
                    help='Whether to use a conv. layer',
                    choices=[0, 1])
parser.add_argument('--convFilt',
                    type=int,
                    default=200,
                    help='Number of convolution kernels')
parser.add_argument('--convFiltSize',
                    type=int,
                    default=3,
                    help='Size of convolution kernels')
parser.add_argument('--learningRate',
                    type=float,
                    default=0.001,
                    help='Learning rate')
parser.add_argument('--optimizer',
                    type=str,
                    default='rms',
                    help='Training optimizer',
                    choices=['rms', 'ada', 'sgd'])
parser.add_argument('--earlyStopping',
                    type=int,
                    default=0,
                    help='Early stopping',
                    choices=[0, 1])
parser.add_argument('--redLrOnPlat',
                    type=int,
                    default=0,
                    help='Reduce l_rate on plateau',
                    choices=[0, 1])
parser.add_argument('--redLrMonitor',
                    type=str,
                    default='val_f1K',
                    help='Metric for l_rate reduction')
parser.add_argument('--redLrMonitorMode',
                    type=str,
                    default='max',
                    help='Mode for l_rate reduction',
                    choices=['min', 'max'])
parser.add_argument('--redLrPatience',
                    type=int,
                    default=10,
                    help='Patience before l_rate reduc')
parser.add_argument('--redLrFactor',
                    type=float,
                    default=0.5,
                    help='Factor for each l_rate reduc')

# save data and monitor best
parser.add_argument('--saveModel',
                    type=str,
                    default='all',
                    help='Whether to save only best model, or all, or none',
                    choices=['no', 'best', 'all'])
parser.add_argument('--saveBestMonitor',
                    type=str,
                    default='val_f1K',
                    help='What metric to decide best model')
parser.add_argument('--saveBestMonMode',
                    type=str,
                    default='max',
                    help='Mode to define best',
                    choices=['min', 'max'])
parser.add_argument('--saveGlobalresults',
                    type=str,
                    default='reports/corpora/DictaSign/recognitionMulti/global/globalMulti.dat',
                    help='Where to save global results')
parser.add_argument('--savePredictions',
                    type=str,
                    default='reports/corpora/DictaSign/recognitionMulti/predictions/',
                    help='Where to save predictions')


# Metrics
parser.add_argument('--stepWolf',
                    type=float,
                    default=0.1,
                    help='Step between Wolf metric eval points',
                    choices=['rms', 'ada', 'sgd'])

args = parser.parse_args()

# Random initilialization
np.random.seed(args.randSeed)

## PARAMETERS
corpus = 'DictaSign'

# Output type
#outputName = args.outputName#'PT'
#flsBinary  = bool(args.flsBinary)#True
#flsKeep    = args.flsKeep#[]
outputsList       = args.outputsList
outputsWeightList = args.outputsWeightList
comment           = args.comment#[]

# Training global setting
videoSplitMode    = args.videoSplitMode
fractionValid     = args.fractionValid
fractionTest      = args.fractionTest
signerIndependent = bool(args.signerIndependent)#False
taskIndependent   = bool(args.taskIndependent)
excludeTask9      = bool(args.excludeTask9)
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
save              = args.saveModel
saveMonitor       = args.saveBestMonitor
saveMonitorMode   = args.saveBestMonMode
saveGlobalresults = args.saveGlobalresults
savePredictions   = args.savePredictions

# Metrics
stepWolf     = args.stepWolf#0.1
metrics      = ['acc',  f1K,   precisionK,   recallK]
metricsNames = ['acc', 'f1K', 'precisionK', 'recallK']

nOutputs = len(outputsList)
outputName = ''
outputNbList = []
outputCategories = []
outputAssembleList = []
for i in range(nOutputs-1):
    outputName += outputsList[i]
    outputName += '_'
    outputCategories.append([1])
    outputAssembleList.append([outputsList[i]])
    outputNbList.append(2)
outputName += outputsList[nOutputs-1]
outputCategories.append([1])
outputAssembleList.append([outputsList[nOutputs-1]])
outputNbList.append(2)

timeString = str(round(time.time()/10))
saveBestName='recognitionMultiDictaSign_'+outputName+'_'+timeString

if path.exists(saveGlobalresults):
    dataGlobal = pickle.load(open(saveGlobalresults, 'rb'))
else:
    dataGlobal = {}

if outputName not in dataGlobal:
    dataGlobal[outputName] = {}

dataGlobal[outputName][timeString] = {}

dataGlobal[outputName][timeString]['comment'] = comment

dataGlobal[outputName][timeString]['params'] = {}
#dataGlobal[outputName][timeString]['params']['flsBinary']           = flsBinary
#dataGlobal[outputName][timeString]['params']['flsKeep']             = flsKeep
dataGlobal[outputName][timeString]['params']['videoSplitMode']      = videoSplitMode
dataGlobal[outputName][timeString]['params']['fractionValid']       = fractionValid
dataGlobal[outputName][timeString]['params']['fractionTest']        = fractionTest
dataGlobal[outputName][timeString]['params']['signerIndependent']   = signerIndependent
dataGlobal[outputName][timeString]['params']['taskIndependent']     = taskIndependent
dataGlobal[outputName][timeString]['params']['excludeTask9']        = excludeTask9
dataGlobal[outputName][timeString]['params']['tasksTrain']          = tasksTrain
dataGlobal[outputName][timeString]['params']['tasksValid']          = tasksValid
dataGlobal[outputName][timeString]['params']['tasksTest']           = tasksTest
dataGlobal[outputName][timeString]['params']['signersTrain']        = signersTrain
dataGlobal[outputName][timeString]['params']['signersValid']        = signersValid
dataGlobal[outputName][timeString]['params']['signersTest']         = signersTest
dataGlobal[outputName][timeString]['params']['idxTrainBypass']      = idxTrainBypass
dataGlobal[outputName][timeString]['params']['idxValidBypass']      = idxValidBypass
dataGlobal[outputName][timeString]['params']['idxTestBypass']       = idxTestBypass
dataGlobal[outputName][timeString]['params']['weightCorrection']    = weightCorrection
dataGlobal[outputName][timeString]['params']['seq_length']          = seq_length
dataGlobal[outputName][timeString]['params']['batch_size']          = batch_size
dataGlobal[outputName][timeString]['params']['epochs']              = epochs
dataGlobal[outputName][timeString]['params']['separation']          = separation
dataGlobal[outputName][timeString]['params']['dropout']             = dropout
dataGlobal[outputName][timeString]['params']['rnn_number']          = rnn_number
dataGlobal[outputName][timeString]['params']['rnn_hidden_units']    = rnn_hidden_units
dataGlobal[outputName][timeString]['params']['mlp_layers_number']   = mlp_layers_number
dataGlobal[outputName][timeString]['params']['convolution']         = convolution
dataGlobal[outputName][timeString]['params']['convFilt']            = convFilt
dataGlobal[outputName][timeString]['params']['convFiltSize']        = convFiltSize
dataGlobal[outputName][timeString]['params']['learning_rate']       = learning_rate
dataGlobal[outputName][timeString]['params']['optimizer']           = optimizer
dataGlobal[outputName][timeString]['params']['earlyStopping']       = earlyStopping
dataGlobal[outputName][timeString]['params']['reduceLrOnPlateau']   = reduceLrOnPlateau
dataGlobal[outputName][timeString]['params']['reduceLrMonitor']     = reduceLrMonitor
dataGlobal[outputName][timeString]['params']['reduceLrMonitorMode'] = reduceLrMonitorMode
dataGlobal[outputName][timeString]['params']['reduceLrPatience']    = reduceLrPatience
dataGlobal[outputName][timeString]['params']['reduceLrFactor']      = reduceLrFactor
dataGlobal[outputName][timeString]['params']['save']                = save
dataGlobal[outputName][timeString]['params']['saveMonitor']         = saveMonitor
dataGlobal[outputName][timeString]['params']['saveMonitorMode']     = saveMonitorMode
dataGlobal[outputName][timeString]['params']['saveGlobalresults']   = saveGlobalresults
dataGlobal[outputName][timeString]['params']['savePredictions']     = savePredictions
dataGlobal[outputName][timeString]['params']['stepWolf']            = stepWolf



## GET VIDEO INDICES
if len(idxTrainBypass) + len(idxValidBypass) + len(idxTestBypass) > 0:
    idxTrain = np.array(idxTrainBypass)
    idxValid = np.array(idxValidBypass)
    idxTest  = np.array(idxTestBypass)
else:
    idxTrain, idxValid, idxTest = getVideoIndicesSplitDictaSign(tasksTrain,
                                                                tasksValid,
                                                                tasksTest,
                                                                signersTrain,
                                                                signersValid,
                                                                signersTest,
                                                                signerIndependent,
                                                                taskIndependent,
                                                                excludeTask9,
                                                                videoSplitMode,
                                                                fractionValid,
                                                                fractionTest,
                                                                checkSplits=True,
                                                                checkSets=True)


#get_annotations_videos_categories('DictaSign',['PT', 'DS', 'fls'], [[1], [1], [1]], output_assemble=[['PT'], [ 'DS'], ['fls']], video_indices=np.arange(2),from_notebook=True)


features_train, annot_train = get_data_concatenated(corpus=corpus,
                                                    output_form='mixed',
                                                    output_names_final=outputsList,
                                                    output_categories_or_names_original=outputCategories,
                                                    output_assemble=outputAssembleList,
                                                    video_indices=idxTrain,
                                                    separation=separation)
features_valid, annot_valid = get_data_concatenated(corpus=corpus,
                                                    output_form='mixed',
                                                    output_names_final=outputsList,
                                                    output_categories_or_names_original=outputCategories,
                                                    output_assemble=outputAssembleList,
                                                    video_indices=idxValid,
                                                    separation=separation)
features_test, annot_test   = get_data_concatenated(corpus=corpus,
                                                    output_form='mixed',
                                                    output_names_final=outputsList,
                                                    output_categories_or_names_original=outputCategories,
                                                    output_assemble=outputAssembleList,
                                                    video_indices=idxTest,
                                                    separation=separation)

classWeightFinal = []
#for i in range(nOutputs):
#    nClasses = outputNbList[i]#annot_train.shape[2]
#    classWeightsCorrected, _ = weightVectorImbalancedDataOneHot(annot_train[i][0, :, :])
#    classWeightsNotCorrected = np.ones(nClasses)
#    classWeightFinal.append(weightCorrection*classWeightsCorrected + (1-weightCorrection)*classWeightsNotCorrected)


model = get_model(outputsList,outputNbList,outputsWeightList,
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

history = train_model(model,
            features_train,
            annot_train,
            features_valid,
            annot_valid,
            output_class_weights=classWeightFinal,
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


time_distributed_1_f1K = np.array(history['time_distributed_1_f1K'])
time_distributed_2_f1K = np.array(history['time_distributed_2_f1K'])
time_distributed_3_f1K = np.array(history['time_distributed_3_f1K'])
time_distributed_4_f1K = np.array(history['time_distributed_4_f1K'])


time_distributed_sum_f1K = time_distributed_1_f1K + time_distributed_2_f1K + time_distributed_3_f1K + time_distributed_4_f1K
print(time_distributed_sum_f1K.shape)

bestf1K_idx = np.argmax(time_distributed_sum_f1K)
bestf1K_str = str(bestf1K_idx+1).zfill(3)

# Results
print('Results')
model.load_weights(saveBestName+'.'+bestf1K_str+'.hdf5')
dataGlobal[outputName][timeString]['results'] = {}
dataGlobal[outputName][timeString]['results']['metrics'] = {}

nOutputs

# Valid results
for metricName in history.keys():
    dataGlobal[outputName][timeString]['results']['metrics'][metricName] = history[metricName]
for config in ['valid', 'test']:
    dataGlobal[outputName][timeString]['results'][config] = {}
    for iOut in range(nOutputs):
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]] = {}
    acc = []
    frameP = []
    frameR = []
    frameF1 = []
    pStarTp = []
    pStarTr = []
    rStarTp = []
    rStarTr = []
    fStarTp = []
    fStarTr = []
    if config == 'valid':
        print('Validation set')
        nRound_valid=annot_valid[0].shape[1]//seq_length
        timestepsRound_valid = nRound_valid *seq_length
        predict_valid = model.predict(features_valid[:,:timestepsRound_valid,:].reshape(-1, seq_length, features_valid.shape[2]))
        for iOut in range(nOutputs):
            predict_valid[iOut] = predict_valid[iOut].reshape(1, timestepsRound_valid, nClasses)
        #predict_valid = predict_valid[0]
        for iOut in range(nOutputs):
            acc.append(framewiseAccuracy(annot_valid[iOut][0,:nRound_valid *seq_length,:],predict_valid[iOut][0, :nRound_valid *seq_length,:],True,True))
            a, b, c = framewisePRF1(annot_valid[iOut][0,:nRound_valid *seq_length,:], predict_valid[iOut][0, :nRound_valid *seq_length,:], True, True)
            frameP.append(a)
            frameR.append(b)
            frameF1.append(c)
            a, b, c, d, e, f = prfStar(annot_valid[iOut][0,:nRound_valid*seq_length,:], predict_valid[iOut][0, :nRound_valid *seq_length,:], True, True, step=stepWolf)
            pStarTp.append(a)
            pStarTr.append(b)
            rStarTp.append(c)
            rStarTr.append(d)
            fStarTp.append(e)
            fStarTr.append(f)
        nameHistoryAppend = 'val_'
    else:
        print('Test set')
        nRound_test=annot_test[0].shape[1]//seq_length
        timestepsRound_test = nRound_test *seq_length
        predict_test = model.predict(features_test[:,:timestepsRound_test,:].reshape(-1, seq_length, features_test.shape[2]))
        for iOut in range(nOutputs):
            predict_test[iOut] = predict_test[iOut].reshape(1, timestepsRound_test, nClasses)
        #predict_test = predict_test[0]
        for iOut in range(nOutputs):
            acc.append(framewiseAccuracy(annot_test[iOut][0,:nRound_test *seq_length,:],predict_test[iOut][0, :nRound_test *seq_length,:],True,True))
            a, b, c = framewisePRF1(annot_test[iOut][0,:nRound_test *seq_length,:], predict_test[iOut][0, :nRound_test *seq_length,:], True, True)
            frameP.append(a)
            frameR.append(b)
            frameF1.append(c)
            a, b, c, d, e, f = prfStar(annot_test[iOut][0,:nRound_test*seq_length,:], predict_test[iOut][0, :nRound_test *seq_length,:], True, True, step=stepWolf)
            pStarTp.append(a)
            pStarTr.append(b)
            rStarTp.append(c)
            rStarTr.append(d)
            fStarTp.append(e)
            fStarTr.append(f)
        nameHistoryAppend = ''
    for iOut in range(nOutputs):
        print(outputsList[iOut])
        print('Framewise accuracy: ' + str(acc[iOut]))
        print('Framewise P, R, F1: ' + str(frameP[iOut]) + ', ' + str(frameR[iOut]) + ', ' + str(frameF1[iOut]))
        print('P*(0,0), R*(0,0), F1*(0,0):' + str(pStarTp[iOut][0]) + ', ' + str(rStarTp[iOut][0]) + ', ' + str(fStarTp[iOut][0]))
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['frameAcc'] = acc[iOut]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['frameP']  = frameP[iOut]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['frameR']  = frameR[iOut]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['frameF1'] = frameF1[iOut]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['pStarTp'] = pStarTp[iOut]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['pStarTr'] = pStarTr[iOut]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['rStarTp'] = rStarTp[iOut]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['rStarTr'] = rStarTr[iOut]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['fStarTp'] = fStarTp[iOut]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['fStarTr'] = fStarTr[iOut]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['pStarZeroZero'] = pStarTp[iOut][0]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['rStarZeroZero'] = rStarTp[iOut][0]
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['fStarZeroZero'] = fStarTp[iOut][0]
        Ip, Ir, Ipr = integralValues(fStarTp[iOut], fStarTr[iOut], step=stepWolf)
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['Ip']  = Ip
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['Ir']  = Ir
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['Ipr'] = Ipr
        print('Ip, Ir, Ipr (star): ' + str(Ip) + ', ' + str(Ir) + ', ' + str(Ipr))
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['middleUnitP']  = {}
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['middleUnitR']  = {}
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['middleUnitF1'] = {}
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['marginUnitP']  = {}
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['marginUnitR']  = {}
        dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['marginUnitF1'] = {}
        for margin in [0, 12, 25, 50]:
            print('margin = ' + str(margin))
            if config == 'valid':
                middleUnitP, middleUnitR, middleUnitF1 = middleUnitPRF1(annot_valid[iOut][0, :nRound_valid*seq_length,:], predict_valid[iOut][0, :nRound_valid*seq_length,:],True,True,margin)
                marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_valid[iOut][0, :nRound_valid*seq_length,:], predict_valid[iOut][0, :nRound_valid*seq_length,:],True,True,margin)
            else:
                middleUnitP, middleUnitR, middleUnitF1 = middleUnitPRF1(annot_test[iOut][0, :nRound_test*seq_length,:], predict_test[iOut][0, :nRound_test*seq_length,:],True,True,margin)
                marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_test[iOut][0, :nRound_test*seq_length,:], predict_test[iOut][0, :nRound_test*seq_length,:],True,True,margin)
            print('P, R, F1 (middleUnit): ' + str(middleUnitP) + ', ' + str(middleUnitR) + ', ' + str(middleUnitF1))
            print('P, R, F1 (marginUnit): ' + str(marginUnitP) + ', ' + str(marginUnitR) + ', ' + str(marginUnitF1))
            dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['middleUnitP'][margin]  = middleUnitP
            dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['middleUnitR'][margin]  = middleUnitR
            dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['middleUnitF1'][margin] = middleUnitF1
            dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['marginUnitP'][margin]  = marginUnitP
            dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['marginUnitR'][margin]  = marginUnitR
            dataGlobal[outputName][timeString]['results'][config][outputsList[iOut]]['marginUnitF1'][margin] = marginUnitF1

pickle.dump(dataGlobal, open(saveGlobalresults,'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#np.savez(savePredictions+saveBestName, true=annot_test[0,:timestepsRound_test,:], pred=predict_test, idxTest=idxTest, separation=separation)
