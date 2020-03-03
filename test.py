from src.models.data_utils import *
from src.models.model_utils import *
from src.models.train_model import *

model_1 = get_model(['PT', 'DS'],[2,2],[1,1])
features_1_train, annot_1_train = get_data_concatenated('DictaSign','mixed',['PT', 'DS', 'fls'], [[1], [1], [41891,43413,43422,42992]], video_indices=np.arange(0,10))
features_1_valid, annot_1_valid = get_data_concatenated('DictaSign','mixed',['PT', 'DS', 'fls'], [[1], [1], [41891,43413,43422,42992]], video_indices=np.arange(10,20))

features_2_train, annot_2_train = get_data_concatenated('NCSLGR','sign_types',['PT', 'DS', 'fls'], [['IX_1p', 'IX_2p', 'IX_3p'], [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'], ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']], video_indices=np.arange(0,10))
features_2_valid, annot_2_valid = get_data_concatenated('NCSLGR','sign_types',['PT', 'DS', 'fls'], [['IX_1p', 'IX_2p', 'IX_3p'], [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'], ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']], video_indices=np.arange(10,20))

features_3_train, annot_3_train = get_data_concatenated('NCSLGR','mixed',['PT', 'DS', 'fls'], [[1], [1], [1]], output_assemble=[['IX_1p', 'IX_2p', 'IX_3p'], [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'], ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']], video_indices=np.arange(0,10))
features_3_valid, annot_3_valid = get_data_concatenated('NCSLGR','mixed',['PT', 'DS', 'fls'], [[1], [1], [1]], output_assemble=[['IX_1p', 'IX_2p', 'IX_3p'], [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'], ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']], video_indices=np.arange(10,20))