from models.data_utils import *
from models.model_utils import *
from models.train_model import *
from models.perf_utils import *
import math

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

# Data split
fractionValid = 0.10
fractionTest = 0.05
videosToDelete = ['dorm_prank_1053_small_0_1.mov', 'DSP_DeadDog.mov', 'DSP_Immigrants.mov', 'DSP_Trip.mov']
lengthCriterion = 300


## GET VIDEO INDICES
tmpAnnot = np.load('data/processed/NCSLGR/annotations.npz', encoding='latin1', allow_pickle=True)
namesVideos = np.load('data/processed/NCSLGR/list_videos.npy')
nVideos = namesVideos.shape[0]
idxKeep = np.ones(nVideos)
for v in videosToDelete:
    idxV = np.where(namesVideos==v)[0][0]
    idxKeep[idxV] = 0
idxKeepLong = np.zeros(nVideos)
idxKeepShort = np.zeros(nVideos)
for idxV in range(nVideos):
    if idxKeep[idxV]:
        tmpLength = tmpAnnot['lexical_with_ns_not_fs'][idxV].shape[0]
        if tmpLength > lengthCriterion:
            idxKeepLong[idxV] = 1
        else:
            idxKeepShort[idxV] = 1

def getVideoIndices():
    #Long
    nbLong = np.sum(idxKeepLong)
    startTestLong = 0
    endTestLong = startTestLong + math.ceil(fractionTest*nbLong)
    startValidLong = endTestLong
    endValidLong = startValidLong + math.ceil(fractionValid*nbLong)
    startTrainLong = endValidLong
    endTrainLong = nbLong
    #Short
    nbShort = np.sum(idxKeepShort)
    startTestShort = 0
    endTestShort = startTestShort + math.ceil(fractionTest*nbShort)
    startValidShort = endTestShort
    endValidShort = startValidShort + math.ceil(fractionValid*nbShort)
    startTrainShort = endValidShort
    endTrainShort = nbShort

    idxLong = np.where(idxKeepLong)[0]
    idxShort = np.where(idxKeepShort)[0]
    np.random.shuffle(idxLong)
    np.random.shuffle(idxShort)

    idxTrain = np.hstack([idxShort[startTrainShort:endTrainShort], idxLong[startTrainLong:endTrainLong]])
    idxValid = np.hstack([idxShort[startValidShort:endValidShort], idxLong[startValidLong:endTValidLong]])
    idxTest =  np.hstack([idxShort[startTestShort:endTestShort],   idxLong[startTestLong:endTestLong]])
    np.random.shuffle(idxTrain)
    np.random.shuffle(idxValid)
    np.random.shuffle(idxTest)

    return idxTrain, idxValid, idxTest


idxTrain, idxValid, idxTest = getVideoIndices()
print(idxTrain)
print(idxValid)
print(idxTest)



# A model with 1 output matrix:
# [other, Pointing, Depicting, Lexical]
model_2 = get_model(outputNames,[4],[1])
features_2_train, annot_2_train = get_data_concatenated(corpus,
                                                        'sign_types',
                                                        catNames,
                                                        catDetails,
                                                        video_indices=np.arange(0,10))
features_2_valid, annot_2_valid = get_data_concatenated(corpus,
                                                        'sign_types',
                                                        catNames,
                                                        catDetails,
                                                        video_indices=np.arange(10,20))
train_model(model_2, features_2_train, annot_2_train, features_2_valid, annot_2_valid, 1000, 10, 100)
