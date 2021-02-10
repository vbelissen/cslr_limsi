
#from src.models.data_utils import *
from src.models.model_utils import *
from src.models.train_model import *
from src.models.perf_utils import *
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
from matplotlib import pyplot as plt
import numpy as np
import pickle

plt.switch_backend('TKAgg')

listeOutputs = ['fls','DS','PT','FBUOY']
colors = ['navy', 'darkred', 'black', 'darkgreen']
linestyles = ['-', '--', '-.', ':']
outputsLatex = {'fls':'\\acrshort{fls}', 'DS':'\\acrshort{ds}', 'PT':'\\acrshort{pts}', 'FBUOY':'\\acrshort{fbuoy}'}
listeFeatures = ['2Dfeatures_HS','2Dfeatures_HS','2Dfeatures_HS','2Dfeatures_HS']
nOuts = 4

idxTest = np.load('reports/corpora/DictaSign/recognitionUnique/predictions/recognitionUniqueDictaSign_DS_159336941.npz')['idxTest']
idxTestClasse = np.sort(idxTest)
print(idxTestClasse)
nbVid = idxTestClasse.size

nbFrames = np.zeros(nbVid)
annotation_raw = np.load('data/processed/DictaSign/annotations.npz', encoding='latin1', allow_pickle=True)['dataBrut_DS']

idxDebFin = {'debut': np.zeros(nbVid), 'fin':np.zeros(nbVid)}

namesTest = []
namesAll = np.load('data/processed/DictaSign/list_videos.npy')
for i in range(nbVid):
    namesTest.append(namesAll[idxTestClasse[i]])
    nbFrames[i] = annotation_raw[idxTestClasse[i]].shape[0]
print(namesTest)

results_videos = {}
for i in range(nbVid):
    results_videos[namesTest[i]] = {}
    for iOut in range(nOuts):
        out = listeOutputs[iOut]
        results_videos[namesTest[i]][out] = {}
        results_videos[namesTest[i]][out]['true'] = np.zeros(int(nbFrames[i]))
        results_videos[namesTest[i]][out]['pred'] = np.zeros(int(nbFrames[i]))



savePredictions='reports/corpora/DictaSign/recognitionUnique/predictions/'
saveGlobalResults='reports/corpora/DictaSign/recognitionUnique/global/globalUnique.dat'
dataGlobal=pickle.load(open(saveGlobalResults,'rb'))

for iOut in range(nOuts):
    out = listeOutputs[iOut]
    print(out)
    dataGlobal[out].keys()
    for k in dataGlobal[out].keys():
        if dataGlobal[out][k]['comment'] == 'plot thesis v3':
            if dataGlobal[out][k]['params']['inputType'] == listeFeatures[iOut]:
                print(k)
                print(dataGlobal[out][k]['params']['inputType'])
                K = k
                saveBestName='recognitionUniqueDictaSign_'+out+'_'+str(K)

                predict = np.load(savePredictions+saveBestName+'.npz')
                true=predict['true']
                pred=predict['pred']
                idxTest=predict['idxTest']

                T = true.shape[0]

                idxDebFin = {'debut': np.zeros(nbVid), 'fin':np.zeros(nbVid)}

                idxDebTmp = 0
                for i in range(nbVid):
                    nbFramesVid = annotation_raw[idxTest[i]].shape[0]
                    finTmp = np.min([T, idxDebTmp+nbFramesVid])
                    results_videos[namesAll[idxTest[i]]][out]['true'][0:finTmp-idxDebTmp] = true[idxDebTmp:finTmp,1]
                    results_videos[namesAll[idxTest[i]]][out]['pred'][0:finTmp-idxDebTmp] = pred[idxDebTmp:finTmp,1]
                    idxDebTmp += nbFramesVid


                #plt.figure()
                #plt.plot(np.arange(T), true[:,1])
                #plt.plot(np.arange(T), pred[:,1])
                #plt.show()


import tikzplotlib
from os.path import expanduser
home = expanduser("~")
plotList = range(nbVid)#[8]
for j in plotList:
    for iOut in range(nOuts):
        out = listeOutputs[iOut]
        plt.figure()
        plt.title(namesAll[idxTestClasse[j]])
        plt.yticks([0,0.5,1])
        plt.xlabel("$t$")
        plt.plot(results_videos[namesAll[idxTestClasse[j]]][out]['true'], color=colors[iOut])
        tikzplotlib.save(home+'/thesis/testSequences/'+namesAll[idxTestClasse[j]]+'_'+out+'_true.tex')
        plt.close()
        plt.figure()
        plt.title(namesAll[idxTestClasse[j]])
        plt.yticks([0,0.5,1])
        plt.xlabel("$t$")
        plt.plot(results_videos[namesAll[idxTestClasse[j]]][out]['pred'], color=colors[iOut], linestyle=linestyles[iOut])
        tikzplotlib.save(home+'/thesis/testSequences/'+namesAll[idxTestClasse[j]]+'_'+out+'_pred_v3.tex')
        plt.close()
    
    


for j in plotList:
    plt.figure()
    plt.title(namesAll[idxTestClasse[j]])
    for iOut in range(nOuts):
        out = listeOutputs[iOut]
        plt.plot(results_videos[namesAll[idxTestClasse[j]]][out]['true'], color=colors[iOut])
        plt.plot(results_videos[namesAll[idxTestClasse[j]]][out]['pred'], color=colors[iOut], linestyle=linestyles[iOut])
    plt.yticks([0,0.5,1])
    plt.xlabel("$t$")
    tikzplotlib.save(home+'/thesis/testSequences/'+namesAll[idxTestClasse[j]]+'_v3.tex')
    plt.show()





outType='PT'
idxClasse = 0
plt.figure()
t = np.arange(7340, 7375)
plt.plot(t, results_videos[namesAll[idxTestClasse[idxClasse]]][outType]['true'][t], label='Annotation')
plt.plot(t, results_videos[namesAll[idxTestClasse[idxClasse]]][outType]['pred'][t], '--', label='Prediction')
plt.xlim((t[0],t[-1]+1))
plt.ylim((0,1))
plt.yticks([0,0.5,1])
plt.xlabel("$t$")
plt.legend()
#plt.title(outputsLatex[outType]+' '+namesTest[idxClasse]+' '+str(t[0])+'-'+str(t[-1]+1))

#
#tikzplotlib.save(home+'/thesis/testSequences/'+outType+'_'+namesTest[idxClasse]+'_'+str(t[0])+'_'+str(t[-1]+1)+'.tex')
#a = tikzplotlib.get_tikz_code()
#print(a)
plt.show()


#model=get_model(['fls'],[2],[1],dropout=0.5,rnn_number=1,rnn_hidden_units=50,mlp_layers_number=0,conv=True,conv_filt=200,conv_ker=3,time_steps=100,learning_rate=0.001,metrics=['acc',  f1K,   precisionK,   recallK],features_number=298)

#smodel.load_weights('models/'+saveBestName+'-best.hdf5')
