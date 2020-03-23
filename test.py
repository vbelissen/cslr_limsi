from src.models.data_utils import *
from src.models.model_utils import *
from src.models.train_model import *

# A model with 3 outputs:
# Pointing signs (binary, weight = 1)
# Depicting signs (binary, weight = 1)
# Lexical signs (categorical, 4 different lexical signs (plus one NULL sign), weight = 1)
model_1 = get_model(['PT', 'DS', 'fls'],[2,2,5],[1,1,1])
features_1_train, annot_1_train = get_data_concatenated('DictaSign',
                                                        'mixed',
                                                        ['PT', 'DS', 'fls'],
                                                        [[1], [1], [41891,43413,43422,42992]],
                                                        video_indices=np.arange(0,70))
features_1_valid, annot_1_valid = get_data_concatenated('DictaSign',
                                                        'mixed',
                                                        ['PT', 'DS', 'fls'],
                                                        [[1], [1], [41891,43413,43422,42992]],
                                                        video_indices=np.arange(70,94))
train_model(model_1, features_1_train, annot_1_train, features_1_valid, annot_1_valid, 2000, 5, 100)

# A model with 1 output matrix:
# [other, Pointing, Depicting, Lexical]
model_2 = get_model(['PT-DS-fls'],[4],[1])
features_2_train, annot_2_train = get_data_concatenated('NCSLGR',
                                                        'sign_types',
                                                        ['PT', 'DS', 'fls'],
                                                        [['IX_1p', 'IX_2p', 'IX_3p'],
                                                         [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'],
                                                         ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']],
                                                        video_indices=np.arange(0,10))
features_2_valid, annot_2_valid = get_data_concatenated('NCSLGR',
                                                        'sign_types',
                                                        ['PT', 'DS', 'fls'],
                                                        [['IX_1p', 'IX_2p', 'IX_3p'],
                                                         [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'],
                                                         ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']],
                                                        video_indices=np.arange(10,20))
train_model(model_2, features_2_train, annot_2_train, features_2_valid, annot_2_valid, 1000, 10, 100)

model_3 = get_model(['PT', 'DS', 'fls'],[2,2,2],[1,1,1])
features_3_train, annot_3_train = get_data_concatenated('NCSLGR',
                                                        'mixed',
                                                        ['PT', 'DS', 'fls'],
                                                        [[1], [1], [1]],
                                                        output_assemble=[['IX_1p', 'IX_2p', 'IX_3p'],
                                                                         [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'],
                                                                         ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']],
                                                        video_indices=np.arange(0,10))
features_3_valid, annot_3_valid = get_data_concatenated('NCSLGR',
                                                        'mixed',
                                                        ['PT', 'DS', 'fls'],
                                                        [[1], [1], [1]],
                                                        output_assemble=[['IX_1p', 'IX_2p', 'IX_3p'],
                                                                         [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'],
                                                                         ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']],
                                                        video_indices=np.arange(10,20))
train_model(model_3, features_3_train, annot_3_train, features_3_valid, annot_3_valid, 1000, 10, 100)

model_4 = get_model(['PT'],[2],[1])
features_4_train, annot_4_train = get_data_concatenated('DictaSign',
                                                        'mixed',
                                                        ['PT'],
                                                        [[1]],
                                                        video_indices=np.arange(0,70))
features_4_valid, annot_4_valid = get_data_concatenated('DictaSign',
                                                        'mixed',
                                                        ['PT'],
                                                        [[1]],
                                                        video_indices=np.arange(70,94))
train_model(model_4, features_4_train, annot_4_train, features_4_valid, annot_4_valid, 2000, 5, 100)

model_5 = get_model(['PT'],[2],[1])
features_5_train, annot_5_train = get_data_concatenated('DictaSign',
                                                        'sign_types',
                                                        ['PT'],
                                                        [['PT']],
                                                        video_indices=np.arange(0,70))
features_5_valid, annot_5_valid = get_data_concatenated('DictaSign',
                                                        'sign_types',
                                                        ['PT'],
                                                        [['PT']],
                                                        video_indices=np.arange(70,94))
train_model(model_5, features_5_train, annot_5_train, features_5_valid, annot_5_valid, 2000, 10, 100)
