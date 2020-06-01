from models.data_utils import *
from models.model_utils import *
from models.train_model import *
from models.perf_utils import *

import math
import numpy as np

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

# Data split
fractionValid = 0.10
fractionTest = 0.05
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
