

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
parser.add_argument('--outputName',
                    type=str,
                    default='PT',
                    help='The output type that the model is trained to recognize')
parser.add_argument('--flsBinary',
                    type=int,
                    default=1,
                    help='If the output is FLS, if seen as binary',
                    choices=[0, 1])
parser.add_argument('--flsKeep',
                    type=int,
                    default=[],
                    help='If the output is FLS, list of FLS indices to consider',
                    nargs='*')
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
parser.add_argument('--inputType',
                    type=str,
                    default='3Dfeatures_HS',
                    choices=['2Draw',
                              '2Draw_HS',
                              '2Draw_HS_noOP',
                              '2Draw_noHands',
                              '2Dfeatures',
                              '2Dfeatures_HS',
                              '2Dfeatures_HS_noOP',
                              '2Dfeatures_noHands',
                              '3Draw',
                              '3Draw_HS',
                              '3Draw_HS_noOP',
                              '3Draw_noHands',
                              '3Dfeatures',
                              '3Dfeatures_HS',
                              '3Dfeatures_HS_noOP',
                              '3Dfeatures_noHands',
                              'none'],
                    help='Type of features')
parser.add_argument('--inputNormed',
                    type=int,
                    default=1,
                    choices=[0, 1],
                    help='If features are normed')
parser.add_argument('--inputFeaturesFrames',
                    type=str,
                    default='features',
                    choices=['features', 'frames', 'both'],
                    help='Features type')
parser.add_argument('--imgWidth',
                    type=int,
                    default=224,
                    choices=range(0,1000),
                    help='CNN width')
parser.add_argument('--imgHeight',
                    type=int,
                    default=224,
                    choices=range(0,1000),
                    help='CNN height')
parser.add_argument('--cnnType',
                    type=str,
                    default='resnet',
                    choices=['resnet', 'vgg', 'mobilenet'],
                    help='CNN type')
parser.add_argument('--cnnFirstTrainedLayer',
                    type=int,
                    default=165,
                    help='Index of first trained layer in CNN')
parser.add_argument('--cnnReduceDim',
                    type=int,
                    default=0,
                    help='If greater than 0, the reduced dimension of CNN output vector')

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
                    default='best',
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
                    default='reports/corpora/DictaSign/recognitionUnique/global/globalUnique.dat',
                    help='Where to save global results')
parser.add_argument('--savePredictions',
                    type=str,
                    default='reports/corpora/DictaSign/recognitionUnique/predictions/',
                    help='Where to save predictions')
parser.add_argument('--saveModels',
                    type=str,
                    default='models/corpora/DictaSign/recognitionUnique/',
                    help='Where to save predictions')
parser.add_argument('--fromNotebook',
                    type=int,
                    default=0,
                    help='When the script is run from a jupyter notebook',
                    choices=[0, 1])

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
outputName = args.outputName#'PT'
flsBinary  = bool(args.flsBinary)#True
flsKeep    = args.flsKeep#[]
comment    = args.comment#[]

# Training global setting
videoSplitMode       = args.videoSplitMode
fractionValid        = args.fractionValid
fractionTest         = args.fractionTest
signerIndependent    = bool(args.signerIndependent)#False
taskIndependent      = bool(args.taskIndependent)
excludeTask9         = bool(args.excludeTask9)
tasksTrain           = args.tasksTrain#[2,3,4,5,6,7,8]
tasksValid           = args.tasksValid#[9]
tasksTest            = args.tasksTest#[7] # session 7
signersTrain         = args.signersTrain#[2,3,4,5,6,7,8]
signersValid         = args.signersValid#[9]
signersTest          = args.signersTest#[7] # session 7
idxTrainBypass       = args.idxTrainBypass
idxValidBypass       = args.idxValidBypass
idxTestBypass        = args.idxTestBypass
weightCorrection     = args.weightCorrection
inputType            = args.inputType
inputNormed          = bool(args.inputNormed)
inputFeaturesFrames  = args.inputFeaturesFrames
imgWidth             = args.imgWidth
imgHeight            = args.imgHeight
cnnType              = args.cnnType
cnnFirstTrainedLayer = args.cnnFirstTrainedLayer
cnnReduceDim         = args.cnnReduceDim


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
saveModels        = args.saveModels
fromNotebook       = bool(args.fromNotebook)

# Metrics
stepWolf     = args.stepWolf#0.1
metrics      = ['acc',  f1K,   precisionK,   recallK]
metricsNames = ['acc', 'f1K', 'precisionK', 'recallK']

timeString = str(round(time.time()/10))
saveBestName = 'recognitionUniqueDictaSign_'+outputName+'_'+timeString

features_dict, features_number = getFeaturesDict(inputType=inputType,
                                                 inputNormed=inputNormed)

if path.exists(saveGlobalresults):
    dataGlobal = pickle.load(open(saveGlobalresults, 'rb'))
else:
    dataGlobal = {}

if outputName not in dataGlobal:
    dataGlobal[outputName] = {}

dataGlobal[outputName][timeString] = {}

dataGlobal[outputName][timeString]['comment'] = comment

dataGlobal[outputName][timeString]['params'] = {}
dataGlobal[outputName][timeString]['params']['flsBinary']            = flsBinary
dataGlobal[outputName][timeString]['params']['flsKeep']              = flsKeep
dataGlobal[outputName][timeString]['params']['videoSplitMode']       = videoSplitMode
dataGlobal[outputName][timeString]['params']['fractionValid']        = fractionValid
dataGlobal[outputName][timeString]['params']['fractionTest']         = fractionTest
dataGlobal[outputName][timeString]['params']['signerIndependent']    = signerIndependent
dataGlobal[outputName][timeString]['params']['taskIndependent']      = taskIndependent
dataGlobal[outputName][timeString]['params']['excludeTask9']         = excludeTask9
dataGlobal[outputName][timeString]['params']['tasksTrain']           = tasksTrain
dataGlobal[outputName][timeString]['params']['tasksValid']           = tasksValid
dataGlobal[outputName][timeString]['params']['tasksTest']            = tasksTest
dataGlobal[outputName][timeString]['params']['signersTrain']         = signersTrain
dataGlobal[outputName][timeString]['params']['signersValid']         = signersValid
dataGlobal[outputName][timeString]['params']['signersTest']          = signersTest
dataGlobal[outputName][timeString]['params']['idxTrainBypass']       = idxTrainBypass
dataGlobal[outputName][timeString]['params']['idxValidBypass']       = idxValidBypass
dataGlobal[outputName][timeString]['params']['idxTestBypass']        = idxTestBypass
dataGlobal[outputName][timeString]['params']['weightCorrection']     = weightCorrection
dataGlobal[outputName][timeString]['params']['inputType']            = inputType
dataGlobal[outputName][timeString]['params']['inputNormed']          = inputNormed
dataGlobal[outputName][timeString]['params']['inputFeaturesFrames']  = inputFeaturesFrames
dataGlobal[outputName][timeString]['params']['imgWidth']             = imgWidth
dataGlobal[outputName][timeString]['params']['imgHeight']            = imgHeight
dataGlobal[outputName][timeString]['params']['cnnType']              = cnnType
dataGlobal[outputName][timeString]['params']['cnnFirstTrainedLayer'] = cnnFirstTrainedLayer
dataGlobal[outputName][timeString]['params']['cnnReduceDim']         = cnnReduceDim
dataGlobal[outputName][timeString]['params']['seq_length']           = seq_length
dataGlobal[outputName][timeString]['params']['batch_size']           = batch_size
dataGlobal[outputName][timeString]['params']['epochs']               = epochs
dataGlobal[outputName][timeString]['params']['separation']           = separation
dataGlobal[outputName][timeString]['params']['dropout']              = dropout
dataGlobal[outputName][timeString]['params']['rnn_number']           = rnn_number
dataGlobal[outputName][timeString]['params']['rnn_hidden_units']     = rnn_hidden_units
dataGlobal[outputName][timeString]['params']['mlp_layers_number']    = mlp_layers_number
dataGlobal[outputName][timeString]['params']['convolution']          = convolution
dataGlobal[outputName][timeString]['params']['convFilt']             = convFilt
dataGlobal[outputName][timeString]['params']['convFiltSize']         = convFiltSize
dataGlobal[outputName][timeString]['params']['learning_rate']        = learning_rate
dataGlobal[outputName][timeString]['params']['optimizer']            = optimizer
dataGlobal[outputName][timeString]['params']['earlyStopping']        = earlyStopping
dataGlobal[outputName][timeString]['params']['reduceLrOnPlateau']    = reduceLrOnPlateau
dataGlobal[outputName][timeString]['params']['reduceLrMonitor']      = reduceLrMonitor
dataGlobal[outputName][timeString]['params']['reduceLrMonitorMode']  = reduceLrMonitorMode
dataGlobal[outputName][timeString]['params']['reduceLrPatience']     = reduceLrPatience
dataGlobal[outputName][timeString]['params']['reduceLrFactor']       = reduceLrFactor
dataGlobal[outputName][timeString]['params']['save']                 = save
dataGlobal[outputName][timeString]['params']['saveMonitor']          = saveMonitor
dataGlobal[outputName][timeString]['params']['saveMonitorMode']      = saveMonitorMode
dataGlobal[outputName][timeString]['params']['saveGlobalresults']    = saveGlobalresults
dataGlobal[outputName][timeString]['params']['savePredictions']      = savePredictions
dataGlobal[outputName][timeString]['params']['saveModels']           = saveModels
dataGlobal[outputName][timeString]['params']['fromNotebook']         = fromNotebook
dataGlobal[outputName][timeString]['params']['stepWolf']             = stepWolf



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
                                                                checkSets=True,
                                                                from_notebook=fromNotebook)


if outputName == 'fls' :
    nKept = len(flsKeep)
    if nKept == 0 and not flsBinary:
        sys.exit('Can not get categorical output if non-zero categories are not listed')
    if flsBinary:
        selected_outputs = [['fls']]
        nonZeros = [flsKeep]
    else:
        selected_outputs = []
        nonZeros = []
        for iKept in range(nKept):
            selected_outputs.append(['fls'])
            nonZeros.append([flsKeep[iKept]])
else:
    selected_outputs = [[outputName]]
    nonZeros = [[]]

features_train, annot_train = get_data_concatenated(corpus=corpus,
                                                    output_form='sign_types',
                                                    types=selected_outputs,
                                                    nonZero=nonZeros,
                                                    binary=[],
                                                    video_indices=idxTrain,
                                                    features_dict=features_dict,
                                                    features_type=inputFeaturesFrames,
                                                    from_notebook=fromNotebook)
features_valid, annot_valid = get_data_concatenated(corpus=corpus,
                                                    output_form='sign_types',
                                                    types=selected_outputs,
                                                    nonZero=nonZeros,
                                                    binary=[],
                                                    video_indices=idxValid,
                                                    features_dict=features_dict,
                                                    features_type=inputFeaturesFrames,
                                                    from_notebook=fromNotebook)
features_test, annot_test   = get_data_concatenated(corpus=corpus,
                                                    output_form='sign_types',
                                                    types=selected_outputs,
                                                    nonZero=nonZeros,
                                                    binary=[],
                                                    video_indices=idxTest,
                                                    features_dict=features_dict,
                                                    features_type=inputFeaturesFrames,
                                                    from_notebook=fromNotebook)


nClasses = annot_train.shape[2]

classWeightsCorrected, _ = weightVectorImbalancedDataOneHot(annot_train[0, :, :])
classWeightsNotCorrected = np.ones(nClasses)
classWeightFinal         = weightCorrection*classWeightsCorrected + (1-weightCorrection)*classWeightsNotCorrected


model = get_model(output_names=[outputName],
                  output_classes=[nClasses],
                  output_weights=[1],
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
                  metrics=metrics,
                  features_number=features_number,
                  features_type=inputFeaturesFrames,
                  img_width=imgWidth,
                  img_height=imgHeight,
                  cnnType=cnnType,
                  cnnFirstTrainedLayer=cnnFirstTrainedLayer,
                  cnnReduceDim=cnnReduceDim)

history = train_model(model=model,
                      features_train=features_train,
                      annot_train=annot_train,
                      features_valid=features_valid,
                      annot_valid=annot_valid,
                      output_class_weights=[classWeightFinal],
                      batch_size=batch_size,
                      epochs=epochs,
                      seq_length=seq_length,
                      save=save,
                      saveMonitor=saveMonitor,
                      saveMonitorMode=saveMonitorMode,
                      saveBestName=saveModels+saveBestName,
                      reduceLrOnPlateau=reduceLrOnPlateau,
                      reduceLrMonitor=reduceLrMonitor,
                      reduceLrMonitorMode=reduceLrMonitorMode,
                      reduceLrPatience=reduceLrPatience,
                      reduceLrFactor=reduceLrFactor,
                      features_type=inputFeaturesFrames,
                      img_width=imgWidth,
                      img_height=imgHeight,
                      cnnType=cnnType)


# Results
print('Results')
model.load_weights(saveModels+saveBestName+'-best.hdf5')
dataGlobal[outputName][timeString]['results'] = {}
dataGlobal[outputName][timeString]['results']['metrics'] = {}

# Valid results
for metricName in history.keys():
    dataGlobal[outputName][timeString]['results']['metrics'][metricName] = history[metricName]
for config in ['valid', 'test']:
    dataGlobal[outputName][timeString]['results'][config] = {}
    if config == 'valid':
        print('Validation set')
        nRound_valid = annot_valid.shape[1]//seq_length
        timestepsRound_valid = nRound_valid*seq_length
        predict_valid = model_predictions(model=model,
                                          features=[features_valid[0][:,:timestepsRound_valid,:], features_valid[1][:timestepsRound_valid]],
                                          features_type=inputFeaturesFrames,
                                          seq_length=seq_length,
                                          categories_per_output=[nClasses],
                                          img_width=imgWidth,
                                          img_height=imgHeight,
                                          cnnType=cnnType,
                                          batch_size=0)
        predict_valid = predict_valid.reshape(1, timestepsRound_valid, nClasses)
        #predict_valid = predict_valid[0]
        acc = framewiseAccuracy(annot_valid[0,:nRound_valid*seq_length,:],
                                predict_valid[0,:nRound_valid*seq_length,:],
                                True,
                                True)
        frameP, frameR, frameF1 = framewisePRF1(annot_valid[0,:nRound_valid*seq_length,:],
                                                predict_valid[0,:nRound_valid*seq_length,:],
                                                True,
                                                True)
        pStarTp, pStarTr, rStarTp, rStarTr, fStarTp, fStarTr = prfStar(annot_valid[0,:nRound_valid*seq_length,:],
                                                                       predict_valid[0,:nRound_valid*seq_length,:],
                                                                       True,
                                                                       True,
                                                                       step=stepWolf)
        nameHistoryAppend = 'val_'
    else:
        print('Test set')
        nRound_test = annot_test.shape[1]//seq_length
        timestepsRound_test = nRound_test*seq_length
        predict_test = model_predictions(model=model,
                                         features=[features_test[0][:,:timestepsRound_test,:], features_test[1][:timestepsRound_test]],
                                         features_type=inputFeaturesFrames,
                                         seq_length=seq_length,
                                         categories_per_output=[nClasses],
                                         img_width=imgWidth,
                                         img_height=imgHeight,
                                         cnnType=cnnType,
                                         batch_size=batch_size)
        predict_test = predict_test.reshape(1, timestepsRound_test, nClasses)
        #predict_test = predict_test[0]
        acc = framewiseAccuracy(annot_test[0,:nRound_test*seq_length,:],
                                predict_test[0,:nRound_test*seq_length,:],
                                True,
                                True)
        frameP, frameR, frameF1 = framewisePRF1(annot_test[0,:nRound_test*seq_length,:],
                                                predict_test[0,:nRound_test*seq_length,:],
                                                True,
                                                True)
        pStarTp, pStarTr, rStarTp, rStarTr, fStarTp, fStarTr = prfStar(annot_test[0,:nRound_test*seq_length,:],
                                                                       predict_test[0,:nRound_test*seq_length,:],
                                                                       True,
                                                                       True,
                                                                       step=stepWolf)
        nameHistoryAppend =  ''

    print('Framewise accuracy: ' + str(acc))
    print('Framewise P, R, F1: ' + str(frameP) + ', ' + str(frameR) + ', ' + str(frameF1))
    print('P*(0,0), R*(0,0), F1*(0,0):' + str(pStarTp[0]) + ', ' + str(rStarTp[0]) + ', ' + str(fStarTp[0]))
    dataGlobal[outputName][timeString]['results'][config]['frameAcc'] = acc
    dataGlobal[outputName][timeString]['results'][config]['frameP']  = frameP
    dataGlobal[outputName][timeString]['results'][config]['frameR']  = frameR
    dataGlobal[outputName][timeString]['results'][config]['frameF1'] = frameF1
    dataGlobal[outputName][timeString]['results'][config]['pStarTp'] = pStarTp
    dataGlobal[outputName][timeString]['results'][config]['pStarTr'] = pStarTr
    dataGlobal[outputName][timeString]['results'][config]['rStarTp'] = rStarTp
    dataGlobal[outputName][timeString]['results'][config]['rStarTr'] = rStarTr
    dataGlobal[outputName][timeString]['results'][config]['fStarTp'] = fStarTp
    dataGlobal[outputName][timeString]['results'][config]['fStarTr'] = fStarTr
    dataGlobal[outputName][timeString]['results'][config]['pStarZeroZero'] = pStarTp[0]
    dataGlobal[outputName][timeString]['results'][config]['rStarZeroZero'] = rStarTp[0]
    dataGlobal[outputName][timeString]['results'][config]['fStarZeroZero'] = fStarTp[0]
    Ip, Ir, Ipr = integralValues(fStarTp, fStarTr,step=stepWolf)
    dataGlobal[outputName][timeString]['results'][config]['Ip']  = Ip
    dataGlobal[outputName][timeString]['results'][config]['Ir']  = Ir
    dataGlobal[outputName][timeString]['results'][config]['Ipr'] = Ipr
    print('Ip, Ir, Ipr (star): ' + str(Ip) + ', ' + str(Ir) + ', ' + str(Ipr))
    dataGlobal[outputName][timeString]['results'][config]['middleUnitP']  = {}
    dataGlobal[outputName][timeString]['results'][config]['middleUnitR']  = {}
    dataGlobal[outputName][timeString]['results'][config]['middleUnitF1'] = {}
    dataGlobal[outputName][timeString]['results'][config]['marginUnitP']  = {}
    dataGlobal[outputName][timeString]['results'][config]['marginUnitR']  = {}
    dataGlobal[outputName][timeString]['results'][config]['marginUnitF1'] = {}
    for margin in [0, 12, 25, 50]:
        print('margin = ' + str(margin))
        if config == 'valid':
            middleUnitP, middleUnitR, middleUnitF1 = middleUnitPRF1(annot_valid[0,:nRound_valid*seq_length,:],
                                                                    predict_valid[0,:nRound_valid*seq_length,:],
                                                                    True,
                                                                    True,
                                                                    margin)
            marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_valid[0,:nRound_valid*seq_length,:],
                                                                    predict_valid[0,:nRound_valid*seq_length,:],
                                                                    True,
                                                                    True,
                                                                    margin)
        else:
            middleUnitP, middleUnitR, middleUnitF1 = middleUnitPRF1(annot_test[0,:nRound_test*seq_length,:],
                                                                    predict_test[0,:nRound_test*seq_length,:],
                                                                    True,
                                                                    True,
                                                                    margin)
            marginUnitP, marginUnitR, marginUnitF1 = marginUnitPRF1(annot_test[0,:nRound_test*seq_length,:],
                                                                    predict_test[0,:nRound_test*seq_length,:],
                                                                    True,
                                                                    True,
                                                                    margin)
        print('P, R, F1 (middleUnit): ' + str(middleUnitP) + ', ' + str(middleUnitR) + ', ' + str(middleUnitF1))
        print('P, R, F1 (marginUnit): ' + str(marginUnitP) + ', ' + str(marginUnitR) + ', ' + str(marginUnitF1))
        dataGlobal[outputName][timeString]['results'][config]['middleUnitP'][margin]  = middleUnitP
        dataGlobal[outputName][timeString]['results'][config]['middleUnitR'][margin]  = middleUnitR
        dataGlobal[outputName][timeString]['results'][config]['middleUnitF1'][margin] = middleUnitF1
        dataGlobal[outputName][timeString]['results'][config]['marginUnitP'][margin]  = marginUnitP
        dataGlobal[outputName][timeString]['results'][config]['marginUnitR'][margin]  = marginUnitR
        dataGlobal[outputName][timeString]['results'][config]['marginUnitF1'][margin] = marginUnitF1

pickle.dump(dataGlobal,
            open(saveGlobalresults,'wb'),
            protocol=pickle.HIGHEST_PROTOCOL)

np.savez(savePredictions+saveBestName,
         true=annot_test[0,:timestepsRound_test,:],
         pred=predict_test,
         idxTest=idxTest,
         separation=separation)
