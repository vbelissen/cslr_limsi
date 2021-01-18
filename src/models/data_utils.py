import numpy as np
import sys
import math
from sklearn.utils.class_weight import compute_class_weight


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
v0 = tf.__version__[0]
if v0 == '2':
    from tensorflow.keras.utils import to_categorical # For tensorflow 2, keras is included in tf
elif v0 == '1':
    from keras.utils import to_categorical # For tensorflow 1.2.0
else:
    sys.exit('Tensorflow version should be 1.X or 2.X')

# 16 signers:
signerRefsDictaSign = np.array(['A11','B15','A2','B0','A1','B14','A9','B17','A6','B13','A10','B16','A7','B4','A3','B5'])

def get_raw_annotation_from_file(corpus, from_notebook=False):
    """
        Gets raw annotation from data file

        Inputs:
            corpus: 'DictaSign' or 'NCSLGR'
            from_notebook: True if used in Jupyter notebook

        Outputs:
            Annotation data
    """
    if from_notebook:
        parent = '../'
    else:
        parent = ''

    annotation_raw = np.load(parent + 'data/processed/' + corpus + '/annotations.npz', encoding='latin1', allow_pickle=True)

    return annotation_raw

def get_raw_annotation_type_video(corpus, type, video_index, provided_annotation=None, from_notebook=False):
    """
        Gets raw annotation from data file or from provided data,
        for a given annotation type and a given video

        Inputs:
            corpus: 'DictaSign' or 'NCSLGR'
            type:
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
            video_index: integer
            provided_annotation: raw annotation data (not needed)
            from_notebook: True if used in Jupyter notebook

        Outputs:
            Annotation data
    """
    if from_notebook:
        parent = '../'
    else:
        parent = ''

    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)

    if corpus == 'DictaSign':
        string_prefix = 'dataBrut_'
    else:
        string_prefix = ''

    return provided_annotation[string_prefix+type][video_index]

def concatenate_annotations(corpus, type, video_indices, separation=0, provided_annotation=None, from_notebook=False):
    """
        Concatenates annotations of one type for several videos

        Inputs:
            corpus: 'DictaSign' or 'NCSLGR'
            type:
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
            video_indices: list/array of integers
            separation: (integer) frames to separate videos
            provided_annotation: raw annotation data (not needed)
            from_notebook: True if used in Jupyter notebook

        Outputs:
            Concatenated data, shape (1, time_steps, 1)
    """

    video_indices = list(video_indices)
    N_videos = len(video_indices)
    if N_videos == 0:
        sys.exit('At least one video index is required')

    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)

    total_time_steps = 0
    for i_v in video_indices:
        temp = get_raw_annotation_type_video(corpus, type, i_v, provided_annotation, from_notebook)
        total_time_steps += temp.shape[0]
        total_time_steps += separation

    output = np.zeros((1,total_time_steps, 1))
    temp_index = 0
    for i_v in video_indices:
        temp = get_raw_annotation_type_video(corpus, type, i_v, provided_annotation, from_notebook)
        temp_length = temp.shape[0]
        output[0,temp_index:temp_index+temp_length,0] = temp[:,0]
        temp_index += temp_length
        temp_index += separation

    return output

def concatenate_fuse_annotations(corpus, types, video_indices, separation=0, provided_annotation=None, from_notebook=False):
    """
        Concatenates and fuse annotations of several type for several videos
        #Types are assumed to be binary

        Inputs:
            corpus: 'DictaSign' or 'NCSLGR'
            types:
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
            video_indices: list/array of integers
            separation: (integer) frames to separate videos
            provided_annotation: raw annotation data (not needed)
            from_notebook: True if used in Jupyter notebook

        Outputs:
            Concatenated fused data, shape (1, time_steps, 1)
    """

    video_indices = list(video_indices)
    N_videos = len(video_indices)
    if N_videos == 0:
        sys.exit('At least one video index is required')

    types = list(types)
    N_types = len(types)
    if N_types == 0:
        sys.exit('At least one annotation type is required')

    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)

    total_time_steps = concatenate_annotations(corpus, types[0], video_indices, separation, provided_annotation, from_notebook).shape[1]

    output = np.zeros((1, total_time_steps, N_types))
    for i_t in range(N_types):
        t = types[i_t]
        temp = concatenate_annotations(corpus, t, video_indices, separation, provided_annotation, from_notebook)
        output[:,:,i_t] = temp[:,:,0]

    return (np.sum(output,axis=2).reshape(1, total_time_steps, 1)>0).astype(float)

def concatenate_binarize_annotations(corpus, type, nonZero, video_indices, separation=0, provided_annotation=None, from_notebook=False):
    """
        Concatenates and binarize annotations of one type for several videos

        Inputs:
            corpus: 'DictaSign' or 'NCSLGR'
            type:
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
            nonZero: 'all' (if anything other than 0 should be counted as 1)
                      or a list of non-zero categories to count positively
            video_indices: list/array of integers
            separation: (integer) frames to separate videos
            provided_annotation: raw annotation data (not needed)
            from_notebook: True if used in Jupyter notebook

        Outputs:
            Concatenated binarized data, shape (1, time_steps, 1)
    """

    video_indices = list(video_indices)
    N_videos = len(video_indices)
    if N_videos == 0:
        sys.exit('At least one video index is required')

    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)

    output = concatenate_annotations(corpus, type, video_indices, separation, provided_annotation, from_notebook)

    if nonZero!='all':
        N_nonZero = len(nonZero)
        mask_class_garbage = np.ones(output.shape).astype(bool)
        for i_c in range(N_nonZero):
            mask_class_garbage *= (output != nonZero[i_c])
        output[mask_class_garbage] = 0

    return (output>0).astype(float)

def concatenate_categorize_annotations(corpus, type, nonZero, video_indices, separation=0, provided_annotation=None, from_notebook=False):
    """
        Concatenates and make categorical annotations of one type for several videos

        Inputs:
            corpus: 'DictaSign' or 'NCSLGR'
            type:
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
            nonZero: a list of non-zero categories to count positively
            video_indices: list/array of integers
            separation: (integer) frames to separate videos
            provided_annotation: raw annotation data (not needed)
            from_notebook: True if used in Jupyter notebook

        Outputs:
            Concatenated categorical data, shape (1, time_steps, C+1) where C is the number of nonZero categories
    """

    video_indices = list(video_indices)
    N_videos = len(video_indices)
    if N_videos == 0:
        sys.exit('At least one video index is required')

    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)

    output_raw = concatenate_annotations(corpus, type, video_indices, separation, provided_annotation, from_notebook)
    output_binary = concatenate_binarize_annotations(corpus, type, nonZero, video_indices, separation, provided_annotation, from_notebook)

    C = len(nonZero)
    if C==0:
        sys.exit('At least one non-zero value is required')

    output_raw[output_binary==0]=0

    for i_C in range(1,C+1):
        output_raw[output_raw==nonZero[i_C-1]] = i_C

    return to_categorical(output_raw, C+1)

def get_concatenated_sign_types(corpus, types, nonZero, video_indices, separation=0, provided_annotation=None, from_notebook=False):
    """
        Concatenates and returns a matrix of sign types

        Inputs:
            corpus: 'DictaSign' or 'NCSLGR'
            types: a list of lists of types to regroup
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
            nonZero: a list lists of non-zero categories to count positively
                     if anything other than 0 is counted positive, choose []
            video_indices: list/array of integers
            separation: (integer) frames to separate videos
            provided_annotation: raw annotation data (not needed)
            from_notebook: True if used in Jupyter notebook

        Outputs:
            Concatenated categorical data, shape (1, time_steps, C+1) where C is the number of major types

            e.g.1 : types = [['DS'], ['PT']], nonZeros = [[], []]
            e.g.2 : types = [['PT_PRO1', 'PT_PRO2', 'PT_PRO3'], ['fls']], nonZeros = [[], [41891,43413,42495,42093]]
            e.g.3 : types = [['PT_PRO1', 'PT_PRO2', 'PT_PRO3'], ['fls']], nonZeros = [[], []]

    """

    video_indices = list(video_indices)
    N_videos = len(video_indices)
    if N_videos == 0:
        sys.exit('At least one video index is required')

    types = list(types)
    N_types = len(types)
    if N_types == 0:
        sys.exit('At least one annotation type is required')

    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)

    tmp = concatenate_annotations(corpus, types[0][0], video_indices, separation, provided_annotation, from_notebook)

    output = np.zeros((1, tmp.shape[1], N_types+1))

    for i_t in range(N_types):
        if len(types[i_t])==0:
            sys.exit('There should be at least one annotation category per type')
        elif len(types[i_t])>1:
            if len(nonZero[i_t])>0:
                sys.exit('Grouping several annotation types with non-binary annotation is ambiguous')
            else: # len(nonZero[i_t])==0
                output[0,:,i_t+1] = concatenate_fuse_annotations(corpus, types[i_t], video_indices, separation, provided_annotation, from_notebook)[0,:,0]
        else: # len(types[i_t])==1
            if len(nonZero[i_t])>0:
                output[0,:,i_t+1] = concatenate_binarize_annotations(corpus, types[i_t][0], nonZero[i_t], video_indices, separation, provided_annotation, from_notebook)[0,:,0]
            else: # len(nonZero[i_t])==0:
                output[0,:,i_t+1] = concatenate_binarize_annotations(corpus, types[i_t][0], 'all', video_indices, separation, provided_annotation, from_notebook)[0,:,0]

    output[0,:,0] = (1-(np.sum(output,axis=2)>0)).astype(bool)

    return output

def get_concatenated_mixed(corpus, types, nonZero, binary, video_indices, separation=0, provided_annotation=None, from_notebook=False):
    """
        Concatenates and returns a list of outputs, binary or categorical

        Inputs:
            corpus: 'DictaSign' or 'NCSLGR'
            types: a list of lists of types to regroup
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
            nonZero: a list lists of non-zero categories to count positively
                     if anything other than 0 is counted positive, choose []
            binary: list of True/False
            video_indices: list/array of integers
            separation: (integer) frames to separate videos
            provided_annotation: raw annotation data (not needed)
            from_notebook: True if used in Jupyter notebook

        Outputs:
            Concatenated categorical data, shape (1, time_steps, C+1) where C is the number of major types

            e.g.1 : types = [['DS'], ['PT']], nonZeros = [[], []], binary=[True, True]
            e.g.2 : types = [['PT_PRO1', 'PT_PRO2', 'PT_PRO3'], ['fls']], nonZeros = [[], [41891,43413,42495,42093]], binary=[True, True]
            e.g.2b : types = [['PT_PRO1', 'PT_PRO2', 'PT_PRO3'], ['fls']], nonZeros = [[], [41891,43413,42495,42093]], binary=[True, False]
            e.g.3 : types = [['PT_PRO1', 'PT_PRO2', 'PT_PRO3'], ['fls']], nonZeros = [[], []], binary=[True, True]

    """

    video_indices = list(video_indices)
    N_videos = len(video_indices)
    if N_videos == 0:
        sys.exit('At least one video index is required')

    types = list(types)
    N_types = len(types)
    if N_types == 0:
        sys.exit('At least one annotation type is required')

    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)

    tmp = concatenate_annotations(corpus, types[0][0], video_indices, separation, provided_annotation, from_notebook)

    output_list = []
    for i_t in range(N_types):
        if len(types[i_t])==0:
            sys.exit('There should be at least one annotation category per type')
        elif len(types[i_t])>1:
            if len(nonZero[i_t])>0 or not binary[i_t]:
                sys.exit('Grouping several annotation types with non-binary annotation is ambiguous')
            else: # len(nonZero[i_t])==0 and binary[i_t]:
                output_list.append(to_categorical(concatenate_fuse_annotations(corpus, types[i_t], video_indices, separation, provided_annotation, from_notebook),2))
        else: # len(types[i_t])==1
            if len(nonZero[i_t])>0:
                if binary[i_t]:
                    output_list.append(to_categorical(concatenate_binarize_annotations(corpus, types[i_t][0], nonZero[i_t], video_indices, separation, provided_annotation, from_notebook),2))
                else: # not binary[i_t]:
                    output_list.append(concatenate_categorize_annotations(corpus, types[i_t][0], nonZero[i_t], video_indices, separation, provided_annotation, from_notebook))
            else: # len(nonZero[i_t])==0
                if binary[i_t]:
                    output_list.append(to_categorical(concatenate_binarize_annotations(corpus, types[i_t][0], 'all', video_indices, separation, provided_annotation, from_notebook),2))
                else: # not binary[i_t]:
                    sys.exit('Non-binary categorical output requires at least one nonZero value')
    return output_list

def get_features_videos(corpus,
                        features_dict={'features_HS':np.arange(0, 420),
                                       'features_HS_norm':np.array([]),
                                       'raw':np.array([]),
                                       'raw_norm':np.array([]),
                                       '2Dfeatures':np.array([]),
                                       '2Dfeatures_norm':np.array([])},
                        video_indices=np.arange(94),
                        from_notebook=False):
    """
        Gets all wanted features.

        Inputs:
            corpus (string)
            features_dict: a dictionary indication which features to keep
                e.g.: {'features_HS':np.arange(0, 420), 'features_HS_norm':np.array([]), 'raw':np.array([]), 'raw_norm':np.array([])}
            video_indices: list or numpy array of wanted videos
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            features (list  of numpy arrays [1, time_steps, features_number])
    """

    features = []

    features_number = 0
    for key in features_dict:
        features_number += features_dict[key].size

    if from_notebook:
        parent = '../'
    else:
        parent = ''

    if corpus == 'DictaSign':
        annotation_raw = np.load(parent + 'data/processed/DictaSign/annotations.npz', encoding='latin1', allow_pickle=True)['dataBrut_DS'] # for counting nb of images
    elif corpus == 'NCSLGR':
        annotation_raw = np.load(parent + 'data/processed/NCSLGR/annotations.npz', encoding='latin1', allow_pickle=True)['lexical_with_ns_not_fs'] # for counting nb of images
    else:
        sys.exit('Invalid corpus name')

    for vid_idx in video_indices:
        time_steps = annotation_raw[vid_idx].shape[0]
        features.append(np.zeros((1, time_steps, features_number)))

    features_number_idx = 0
    for key in features_dict:
        key_features_idx = features_dict[key]
        key_features_number = key_features_idx.size
        if key_features_number > 0:
            key_features = np.load(parent + 'data/processed/' + corpus + '/' + key + '.npy', encoding='latin1', allow_pickle=True)
            index_vid_tmp = 0
            for vid_idx in video_indices:
                features[index_vid_tmp][0, :, features_number_idx:features_number_idx+key_features_number] = key_features[vid_idx][:, key_features_idx]
                index_vid_tmp += 1
            features_number_idx += key_features_number

    return features


def get_sequence_features(corpus,
                           vid_idx=0,
                           img_start_idx=0,
                           features_dict={'features_HS':np.arange(0, 420),
                                          'features_HS_norm':np.array([]),
                                          'raw':np.array([]),
                                          'raw_norm':np.array([]),
                                          '2Dfeatures':np.array([]),
                                          '2Dfeatures_norm':np.array([])},
                           time_steps=100,
                           preloaded_features=None,
                           from_notebook=False):
    """
        Function returning features for a sequence.

        Inputs:
            corpus (string)
            vid_idx (int): which video
            img_start_idx (int): which start image
            features_dict: a dictionary indicating which features to keep
                e.g.: {'features_HS':np.arange(0, 420),
                       'features_HS_norm':np.array([]),
                       'raw':np.array([]),
                       'raw_norm':np.array([]),
                       '2Dfeatures':np.array([]),
                       '2Dfeatures_norm':np.array([])}}
            time_steps: length of sequence (int)
            preloaded_features: if features are already loaded, in the format of a list (features for each video)
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            X: a numpy array [1, time_steps, features_number] for features
    """

    if from_notebook:
        parent = '../'
    else:
        parent = ''

    if preloaded_features is None:
        features_number = 0
        for key in features_dict:
            features_number += features_dict[key].size
    else:
        features_number = preloaded_features[vid_idx].shape[1]

    X = np.zeros((1, time_steps, features_number))

    if preloaded_features is None:
        features_number_idx = 0
        for key in features_dict:
            key_features_idx = features_dict[key]
            key_features_number = key_features_idx.size
            if key_features_number > 0:
                key_features = np.load(parent + 'data/processed/' + corpus + '/' + key + '.npy', encoding='latin1', allow_pickle=True)[vid_idx]
                X[0, :, features_number_idx:features_number_idx+key_features_number] = key_features[img_start_idx:img_start_idx + time_steps, key_features_idx]
                features_number_idx += key_features_number
    else:
        X[0, :, :] = preloaded_features[vid_idx][0, img_start_idx:img_start_idx+time_steps, :]

    return X

def get_sequence_annotations_mixed(corpus,
                                   types,
                                   nonZero,
                                   binary,
                                   video_index,
                                   img_start_idx=0,
                                   time_steps=100,
                                   provided_annotation=None,
                                   from_notebook=False):
    """
        For returning annotations for a sequence, in the form of a list of different categories.
            e.g.: get_sequence_annotations_categories('DictaSign',
                                         ['fls', 'DS'],
                                         [[41891,43413,43422,42992],[1]],
                                         vid_idx=17,
                                         img_start_idx=258,
                                         time_steps=100)

        Inputs:
            output_names: list of outputs (strings)
            output_categories: list of lists of meaningful annotation categories for each output
            vid_idx (int): which video
            img_start_idx (int): which start image
            time_steps: length of sequences (int)
            provided_annotation: raw annotation data (not needed)
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            Y: a list, comprising annotations
    """

    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)


    Y = get_concatenated_mixed(corpus, types, nonZero, binary, video_indices=[video_index], separation=0, provided_annotation=provided_annotation, from_notebook=from_notebook)

    N_types = len(types)

    for i_t in range(N_types):
        Y[i_t] = Y[i_t][:,img_start_idx:img_start_idx+time_steps,:]

    return Y


def get_sequence_annotations_sign_types(corpus,
                                        types,
                                        nonZero,
                                        video_index,
                                        img_start_idx=0,
                                        time_steps=100,
                                        provided_annotation=None,
                                        from_notebook=False):
    """
        For returning annotations for a sequence, in the form of a list of different categories, each of which is a list of video annotations.
            e.g.: get_sequence_annotations('DictaSign',
                                         ['fls', 'DS'],
                                         [[41891,43413,43422,42992],[1]],
                                         vid_idx=17,
                                         img_start_idx=258,
                                         time_steps=100)

        Inputs:
            corpus (string)
            output_names_final: list of outputs (strings) corresponding to the desired output_categories
            output_names_original: original names that are used to compose final outputs
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
            vid_idx (int): which video
            img_start_idx (int): which start image
            time_steps: length of sequences (int)
            provided_annotation: raw annotation data (not needed)
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            Y: an annotation array
    """
    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)

    Y = get_concatenated_sign_types(corpus, types, nonZero, [video_index], 0, provided_annotation, from_notebook)

    return Y[:, img_start_idx:img_start_idx+time_steps, :]


def get_sequence(corpus,
                 output_form,
                 types,
                 nonZero,
                 binary,
                 video_index,
                 img_start_idx,
                 features_dict={'features_HS':np.arange(0, 420),
                                'features_HS_norm':np.array([]),
                                'raw':np.array([]),
                                'raw_norm':np.array([]),
                                '2Dfeatures':np.array([]),
                                '2Dfeatures_norm':np.array([])},
                 time_steps=100,
                 preloaded_features=None,
                 provided_annotation=None,
                 features_type='features',
                 frames_path_before_video='/localHD/DictaSign/convert/img/DictaSign_lsf_',
                 empty_image_path='/localHD/DictaSign/convert/img/white.jpg',
                 from_notebook=False):
    """
        For returning features and annotations for a sequence.

        Inputs:
            corpus (string): DictaSign or NCSLGR
            output_form: 'mixed' if different and separated Outputs
                         'sign_types' if annotation is only a binary matrix of sign types
            output_names_final: list of outputs (strings)
            output_categories_or_names_original:
                if output_form: 'mixed': list of lists of meaningful annotation categories for each output
                if output_form: 'sign_types': list of lists of original names that are used to compose final outputs
            vid_idx (int): which video
            img_start_idx (int): which start image
            features_dict: a dictionary indication which features to keep
                e.g.: {'features_HS':np.arange(0, 420), 'features_HS_norm':np.array([]), 'raw':np.array([]), 'raw_norm':np.array([])}
            time_steps: length of sequences (int)
            preloaded_features: if features are already loaded, in the format of a list (features for each video)
            provided_annotation: raw annotation data (not needed)
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            X: a numpy array [1, time_steps, features_number] for features
            Y: array or list, comprising annotations
    """

    if from_notebook:
        parent = '../'
    else:
        parent = ''

    list_videos = np.load(parent + 'data/processed/' + corpus + '/list_videos.npy')


    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)

    if features_type == 'features' or features_type == 'both':
        X_features = get_sequence_features(corpus=corpus,
                              video_index=video_index,
                              img_start_idx=img_start_idx,
                              features_dict=features_dict,
                              time_steps=time_steps,
                              preloaded_features=preloaded_features,
                              from_notebook=from_notebook)
    else:
        X_features = np.array([])

    X_frames = np.repeat('',time_steps).astype('<U100')

    if features_type == 'frames' or features_type == 'both':
        tmp_vid       = np.repeat(frames_path_before_video + list_videos[video_index] + '_front/', time_steps)
        tmp_frames    = np.char.zfill((np.arange(time_steps)+1+img_start_idx).astype('<U5'),5)
        tmp_extension = np.repeat('.jpg', time_steps)
        X_frames = np.core.defchararray.add(np.core.defchararray.add(tmp_vid, tmp_frames), tmp_extension)

    if output_form == 'mixed':
        Y = get_sequence_annotations_mixed(corpus=corpus,
                                           types=types,
                                           nonZero=nonZero,
                                           binary=binary,
                                           video_index=video_index,
                                           img_start_idx=img_start_idx,
                                           time_steps=time_steps,
                                           provided_annotation=provided_annotation,
                                           from_notebook=from_notebook)
    elif output_form == 'sign_types':
        Y = get_sequence_annotations_sign_types(corpus=corpus,
                                                types=types,
                                                nonZero=nonZero,
                                                video_index=video_index,
                                                img_start_idx=img_start_idx,
                                                time_steps=time_steps,
                                                provided_annotation=provided_annotation,
                                                from_notebook=from_notebook)
    else:
        sys.exit('Invalid output form')

    return [X_features, X_frames], Y


def get_data_concatenated(corpus,
                          output_form,
                          types,
                          nonZero,
                          binary=[],
                          video_indices=np.arange(10),
                          features_dict={'features_HS':np.arange(0, 420),
                                         'features_HS_norm':np.array([]),
                                         'raw':np.array([]),
                                         'raw_norm':np.array([]),
                                         '2Dfeatures':np.array([]),
                                         '2Dfeatures_norm':np.array([])},
                          preloaded_features=None,
                          provided_annotation=None,
                          separation=100,
                          from_notebook=False,
                          return_idx_trueData=False,
                          features_type='features',
                          frames_path_before_video='/localHD/DictaSign/convert/img/DictaSign_lsf_',
                          empty_image_path='/localHD/DictaSign/convert/img/white.jpg'):
    """
        For returning concatenated features and annotations for a set of videos (e.g. train set...)
            e.g. features_2_train, annot_2_train = get_data_concatenated('NCSLGR',
                                                                         'sign_types',
                                                                         [['IX_1p', 'IX_2p', 'IX_3p'],
                                                                          [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'],
                                                                          ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']],
                                                                         )
                 features_1_train, annot_1_train = get_data_concatenated('DictaSign',
                                                                         'mixed',
                                                                         [['PT'], ['DS'], ['fls']],
                                                                         [[1], [1], [41891,43413,43422,42992]],
                                                                         [True, True, False]
                                                                         )
                 features_3_train, annot_3_train = get_data_concatenated('NCSLGR',
                                                                         'mixed',
                                                                         [['IX_1p', 'IX_2p', 'IX_3p'],
                                                                          [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'],
                                                                          ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']]
                                                                         )

        Inputs:
            corpus (string)
            output_form: 'mixed' if different and separated Outputs
                         'sign_types' if annotation is only a binary matrix of sign types
            types: a list of lists of original names that are used to compose final outputs
            nonZero: a list of lists of nonzero values to consider
                     if 4 outputs with all nonZero values should be considered, nonZero=[[],[],[],[]]
            binary: only considered when output_form=mixed
                           It's a list (True/False) indicating whether the values should be categorical or binary
            features_dict: a dictionary indication which features to keep
                e.g.: {'features_HS':np.arange(0, 420), 'features_HS_norm':np.array([]), 'raw':np.array([]), 'raw_norm':np.array([])}
            preloaded_features: if features are already loaded, in the format of a list (features for each video)
            provided_annotation: raw annotation data (not needed)
            video_indices: numpy array for a list of videos
            separation: in order to separate consecutive videos
            from_notebook: if notebook script, data is in parent folder
            return_idx_trueData: if True, returns a binary vector with 0 where separations are
            features_type: 'features', 'frames', 'both'
            frames_path_before_video: video frames are supposed to be in folders
                                      like '/localHD/DictaSign/convert/img/DictaSign_lsf_S7_T2_A10',
                                      then frames_path_before_video='/localHD/DictaSign/convert/img/DictaSign_lsf_'
            empty_image_path: path of a white frame

        Outputs:
            X: [a numpy array [1, total_time_steps, features_number] for features,
                a list of frame paths]
            Y: array or list, comprising annotations
    """

    if provided_annotation is None:
        provided_annotation = get_raw_annotation_from_file(corpus, from_notebook)

    if output_form == 'mixed':
        Y = get_concatenated_mixed(corpus=corpus,
                                   types=types,
                                   nonZero=nonZero,
                                   binary=binary,
                                   video_indices=video_indices,
                                   separation=separation,
                                   provided_annotation=provided_annotation,
                                   from_notebook=from_notebook)

    elif output_form == 'sign_types':
        Y = get_concatenated_sign_types(corpus=corpus,
                                        types=types,
                                        nonZero=nonZero,
                                        video_indices=video_indices,
                                        separation=separation,
                                        provided_annotation=provided_annotation,
                                        from_notebook=from_notebook)
    else:
        sys.exit('Invalid output form')

    if from_notebook:
        parent = '../'
    else:
        parent = ''

    list_videos = np.load(parent + 'data/processed/' + corpus + '/list_videos.npy')

    # Getting video lengths:
    video_number = video_indices.size
    video_lengths = np.zeros(video_number, dtype=int)
    total_length = 0
    for i_vid in range(video_number):
        tmp = get_raw_annotation_type_video(corpus, types[0][0], video_indices[i_vid], provided_annotation, from_notebook)
        video_lengths[i_vid] = tmp.shape[0]
        total_length += video_lengths[i_vid]
        total_length += separation

    if preloaded_features is None and features_type != 'frames':
        preloaded_features = get_features_videos(corpus, features_dict, video_indices, from_notebook)

    if features_type == 'features' or features_type == 'both':
        features_number = preloaded_features[0].shape[2]
        X_features = np.zeros((1, total_length, features_number))
    else:
        X_features = np.array([])

    X_frames = np.repeat('',total_length).astype('<U100')

    idx_trueData = np.zeros(total_length)

    img_start_idx = 0
    for i_vid in range(video_number):
        vid_idx = video_indices[i_vid]
        if features_type == 'features' or features_type == 'both':
            X_features[0, img_start_idx:img_start_idx+video_lengths[i_vid], :] = preloaded_features[i_vid][0, :, :]
        if features_type == 'frames' or features_type == 'both':
            tmp_vid       = np.repeat(frames_path_before_video + list_videos[vid_idx] + '_front/', video_lengths[i_vid])
            tmp_frames    = np.char.zfill((np.arange(video_lengths[i_vid])+1).astype('<U5'),5)
            tmp_extension = np.repeat('.jpg', video_lengths[i_vid])
            X_frames[img_start_idx:img_start_idx + video_lengths[i_vid]] = np.core.defchararray.add(np.core.defchararray.add(tmp_vid, tmp_frames), tmp_extension)
            X_frames[img_start_idx + video_lengths[i_vid]:img_start_idx + video_lengths[i_vid]+separation] = empty_image_path
        idx_trueData[img_start_idx:img_start_idx+video_lengths[i_vid]] = 1
        img_start_idx += video_lengths[i_vid]
        img_start_idx += separation

    if return_idx_trueData:
        return [X_features, X_frames], Y, idx_trueData
    else:
        return [X_features, X_frames], Y

def getVideoIndicesSplitNCSLGR(fractionValid=0.10,
                               fractionTest=0.05,
                               videosToDelete=['dorm_prank_1053_small_0_1.mov',
                                               'DSP_DeadDog.mov',
                                               'DSP_Immigrants.mov',
                                               'DSP_Trip.mov'],
                               lengthCriterion=300,
                               includeLong=True,
                               includeShort=True,
                               from_notebook=False):
    """
        Train/valid/test split for NCSLGR
        it uses a length criterion so that data is balanced between train, valid and test

        Inputs:
            fractionValid
            fractionTest
            videosToDelete: list of videos to ignore (bad quality...)
            lengthCriterion
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            idxTrain, idxValid, idxTest: numpy arrays
    """

    if from_notebook:
        parent = '../'
    else:
        parent = ''

    tmpAnnot = np.load(parent+'data/processed/NCSLGR/annotations.npz', encoding='latin1', allow_pickle=True)
    namesVideos = np.load(parent+'data/processed/NCSLGR/list_videos.npy')
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
    #Long
    nbLong = np.sum(idxKeepLong)
    startTestLong = 0
    endTestLong = int(startTestLong + math.ceil(fractionTest*nbLong))
    startValidLong = endTestLong
    endValidLong = int(startValidLong + math.ceil(fractionValid*nbLong))
    startTrainLong = endValidLong
    endTrainLong = int(nbLong)
    #Short
    nbShort = np.sum(idxKeepShort)
    startTestShort = 0
    endTestShort = int(startTestShort + math.ceil(fractionTest*nbShort))
    startValidShort = endTestShort
    endValidShort = int(startValidShort + math.ceil(fractionValid*nbShort))
    startTrainShort = endValidShort
    endTrainShort = int(nbShort)

    idxLong = np.where(idxKeepLong)[0]
    idxShort = np.where(idxKeepShort)[0]
    np.random.shuffle(idxLong)
    np.random.shuffle(idxShort)

    if includeLong and includeShort:
        idxTrain = np.hstack([idxShort[startTrainShort:endTrainShort], idxLong[startTrainLong:endTrainLong]])
        idxValid = np.hstack([idxShort[startValidShort:endValidShort], idxLong[startValidLong:endValidLong]])
        idxTest =  np.hstack([idxShort[startTestShort:endTestShort],   idxLong[startTestLong:endTestLong]])
    elif includeLong and not includeShort:
        idxTrain = idxLong[startTrainLong:endTrainLong]
        idxValid = idxLong[startValidLong:endValidLong]
        idxTest =  idxLong[startTestLong:endTestLong]
    elif includeShort and not includeLong:
        idxTrain = idxShort[startTrainShort:endTrainShort]
        idxValid = idxShort[startValidShort:endValidShort]
        idxTest =  idxShort[startTestShort:endTestShort]
    else:
        sys.exit('Long or Short videos should be included')

    np.random.shuffle(idxTrain)
    np.random.shuffle(idxValid)
    np.random.shuffle(idxTest)

    return idxTrain, idxValid, idxTest

def signerRefToSignerIdxDictaSign(signerRef):
    return np.where(signerRefsDictaSign==signerRef)[0][0]

def signerIdxToSignerRefDictaSign(signerIdx):
    return signerRefsDictaSign[signerIdx]

def getVideoIndicesSplitDictaSign(tasksTrain=[],
                                  tasksValid=[],
                                  tasksTest=[],
                                  signersTrain=[],
                                  signersValid=[],
                                  signersTest=[],
                                  signerIndependent=False,
                                  taskIndependent=False,
                                  excludeTask9=False,
                                  videoSplitMode='auto',
                                  fractionValid=0.2,
                                  fractionTest=0.2,
                                  checkSplits=False,
                                  checkSets=False,
                                  from_notebook=False):
    if videoSplitMode == 'manual':
        idxTrain, idxValid, idxTest = getVideoIndicesSplitDictaSignManual([tasksTrain,tasksValid,tasksTest],
                                                                          [signersTrain,signersValid,signersTest],
                                                                          excludeTask9,
                                                                          from_notebook)
    elif videoSplitMode == 'auto':
        idxTrain, idxValid, idxTest = getVideoIndicesSplitDictaSignAuto(signerIndependent,
                                                                        taskIndependent,
                                                                        excludeTask9,
                                                                        fractionValid,
                                                                        fractionTest,
                                                                        from_notebook)
    else:
        sys.exit('videoSplitMode should be either manual or auto')

    if checkSplits:
        verifSplitSettingDictaSign(videoSplitMode,
                                   tasksTrain,    tasksValid,    tasksTest,
                                   signersTrain,  signersValid,  signersTest)
    if checkSets:
        verifSets(idxTrain, idxValid, idxTest)

    return idxTrain, idxValid, idxTest

def getVideoIndicesSplitDictaSignManual(tasksSplit,
                                        signersSplit,
                                        excludeTask9,
                                        from_notebook=False):#sessionsSplit,
    """
        Train/valid/test split for DictaSign
        Returns intersection of tasks and signers

        Inputs:
            ###sessionsSplit: list of 3 lists of session indices (2 to 9) for train, valid, test
            tasksSplit: list of 3 lists of task indices (1 to 9) for train, valid, test
            signersSplit: list of 3 lists of signer indices (0 to 15) for train, valid, test
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            idxTrain, idxValid, idxTest: numpy arrays
    """
    if from_notebook:
        parent = '../'
    else:
        parent = ''

    l = np.load(parent+'data/processed/DictaSign/list_videos.npy')
    nVideos = len(l)
    #sess = []
    task = []
    signer = []
    for iV in range(nVideos):
        tmp = l[iV].replace('S','').replace('T','').split('_')
        #sess.append(int(tmp[0]))
        task.append(int(tmp[1]))
        signer.append(signerRefToSignerIdxDictaSign(tmp[2]))

    #sess = np.array(sess)
    task = np.array(task)
    signer = np.array(signer)

    #idxTrainSession = np.ones(nVideos)
    idxTrainTask = np.ones(nVideos)
    idxTrainSigner = np.ones(nVideos)
    #idxValidSession = np.ones(nVideos)
    idxValidTask = np.ones(nVideos)
    idxValidSigner = np.ones(nVideos)
    #idxTestSession = np.ones(nVideos)
    idxTestTask = np.ones(nVideos)
    idxTestSigner = np.ones(nVideos)

    for i in range(3):
        #if len(sessionsSplit[i])==0:
        #    sessionsSplit[i] = [j for j in range(2,10)]
        if len(tasksSplit[i])==0:
            if excludeTask9:
                tasksSplit[i] = [j for j in range(1,9)]
            else:
                tasksSplit[i] = [j for j in range(1,10)]
        if len(signersSplit[i])==0:
            signersSplit[i] = [j for j in range(0,16)]

    # for sessionTrain in sessionsSplit[0]:
    #     idxTrainSession[sess == sessionTrain] = 0
    for taskTrain in tasksSplit[0]:
        idxTrainTask[task == taskTrain] = 0
    for signerTrain in signersSplit[0]:
        idxTrainSigner[signer == signerTrain] = 0
    #idxTrainSession = 1-idxTrainSession
    idxTrainTask = 1-idxTrainTask
    idxTrainSigner = 1-idxTrainSigner
    idxTrain = idxTrainTask*idxTrainSigner

    #for sessionValid in sessionsSplit[1]:
    #    idxValidSession[sess == sessionValid] = 0
    for taskValid in tasksSplit[1]:
        idxValidTask[task == taskValid] = 0
    for signerValid in signersSplit[1]:
        idxValidSigner[signer == signerValid] = 0
    #idxValidSession = 1-idxValidSession
    idxValidTask = 1-idxValidTask
    idxValidSigner = 1-idxValidSigner
    idxValid = idxValidTask*idxValidSigner

    #for sessionTest in sessionsSplit[2]:
    #    idxTestSession[sess == sessionTest] = 0
    for taskTest in tasksSplit[2]:
        idxTestTask[task == taskTest] = 0
    for signerTest in signersSplit[2]:
        idxTestSigner[signer == signerTest] = 0
    #idxTestSession = 1-idxTestSession
    idxTestTask = 1-idxTestTask
    idxTestSigner = 1-idxTestSigner
    idxTest = idxTestTask*idxTestSigner

    idxTrain = np.where(idxTrain)[0]
    idxValid = np.where(idxValid)[0]
    idxTest = np.where(idxTest)[0]
    np.random.shuffle(idxTrain)
    np.random.shuffle(idxValid)
    np.random.shuffle(idxTest)

    return idxTrain.astype(int), idxValid.astype(int), idxTest.astype(int)

def getVideoIndicesSplitDictaSignAuto(signerIndependent,
                                      taskIndependent,
                                      excludeTask9,
                                      fractionValid,
                                      fractionTest,
                                      from_notebook=False):

    if signerIndependent and taskIndependent:
        print('Attention, not all videos can be used if SI and TI setting are simultaneous')

    if excludeTask9:
        effectiveNbTasks = 8
    else:
        effectiveNbTasks = 9

    signersIdx = np.arange(16)
    tasksIdx = np.arange(1,effectiveNbTasks+1)
    np.random.shuffle(signersIdx)
    np.random.shuffle(tasksIdx)

    if from_notebook:
        parent = '../'
    else:
        parent = ''

    l = np.load(parent+'data/processed/DictaSign/list_videos.npy')
    nVideos = len(l)
    sess   = []
    task   = []
    signer = []
    for iV in range(nVideos):
        tmp = l[iV].replace('S','').replace('T','').split('_')
        sess.append(int(tmp[0]))
        task.append(int(tmp[1]))
        signer.append(signerRefToSignerIdxDictaSign(tmp[2]))

    sess   = np.array(sess)
    task   = np.array(task)
    signer = np.array(signer)

    idxVideos = np.arange(nVideos)
    np.random.shuffle(idxVideos)

    annotation_raw = np.load(parent + 'data/processed/DictaSign/annotations.npz', encoding='latin1', allow_pickle=True)['dataBrut_DS'] # for counting nb of images

    frames = np.zeros(nVideos)
    for iV in range(nVideos):
        frames[iV] = annotation_raw[iV].shape[0]
    totalFrames = np.sum(frames)

    minFramesTest  = fractionTest * totalFrames
    minFramesValid = fractionValid * totalFrames

    idxTrain = []
    idxValid = []
    idxTest = []

    if not signerIndependent and not taskIndependent:
        currentFill = 'test'
        framesCumulated = 0
        for iV in range(nVideos):
            idxVid = idxVideos[iV]
            framesVid = frames[idxVid]
            if currentFill == 'test':
                idxTest.append(idxVid)
            elif currentFill == 'valid':
                idxValid.append(idxVid)
            else:
                idxTrain.append(idxVid)
            framesCumulated += framesVid
            if currentFill == 'test' and framesCumulated > minFramesTest:
                currentFill = 'valid'
                framesCumulated = 0
            if currentFill == 'valid' and framesCumulated > minFramesValid:
                currentFill = 'train'
    elif signerIndependent and not taskIndependent:
        signersTestNumber  = int(np.max([1, round(fractionTest*16)])) # 16 signers in DictaSign
        signersValidNumber = int(np.max([1, round(fractionValid*16)])) # 16 signers in DictaSign
        signersTest  = signersIdx[:signersTestNumber]
        signersValid = signersIdx[signersTestNumber:signersTestNumber+signersValidNumber]
        for iV in range(nVideos):
            idxVid = idxVideos[iV]
            signerVid = signer[idxVid]
            if signerVid in signersTest:
                idxTest.append(idxVid)
            elif signerVid in signersValid:
                idxValid.append(idxVid)
            else:
                idxTrain.append(idxVid)
    elif taskIndependent and not signerIndependent:
        tasksTestNumber  = int(np.max([1, round(fractionTest*effectiveNbTasks)])) # 9 tasks in DictaSign
        tasksValidNumber = int(np.max([1, round(fractionValid*effectiveNbTasks)])) # 9 tasks in DictaSign
        tasksTest  = tasksIdx[:tasksTestNumber]
        tasksValid = tasksIdx[tasksTestNumber:tasksTestNumber+tasksValidNumber]
        for iV in range(nVideos):
            idxVid = idxVideos[iV]
            taskVid = task[idxVid]
            if taskVid in tasksTest:
                idxTest.append(idxVid)
            elif taskVid in tasksValid:
                idxValid.append(idxVid)
            else:
                idxTrain.append(idxVid)
    else: # signerIndependent and taskIndependent
        apparentFractionTest  = np.power(fractionTest,2/3)#np.sqrt(fractionTest)
        apparentFractionValid = np.power(fractionValid,2/3)#np.sqrt(fractionValid)
        signersTestNumber  = int(np.max([1, round(apparentFractionTest*16)])) # 16 signers in DictaSign
        signersValidNumber = int(np.max([1, round(apparentFractionValid*16)])) # 16 signers in DictaSign
        signersTest  = signersIdx[:signersTestNumber]
        signersValid = signersIdx[signersTestNumber:signersTestNumber+signersValidNumber]
        signersTrain = signersIdx[signersTestNumber+signersValidNumber:]
        tasksTestNumber  = int(np.max([1, round(apparentFractionTest*effectiveNbTasks)])) # 9 tasks in DictaSign
        tasksValidNumber = int(np.max([1, round(apparentFractionValid*effectiveNbTasks)])) # 9 tasks in DictaSign
        tasksTest  = tasksIdx[:tasksTestNumber]
        tasksValid = tasksIdx[tasksTestNumber:tasksTestNumber+tasksValidNumber]
        tasksTrain = tasksIdx[tasksTestNumber+tasksValidNumber:]
        for iV in range(nVideos):
            idxVid = idxVideos[iV]
            signerVid = signer[idxVid]
            taskVid   = task[idxVid]
            if signerVid in signersTest and taskVid in tasksTest:
                idxTest.append(idxVid)
            elif signerVid in signersValid and taskVid in tasksValid:
                idxValid.append(idxVid)
            elif signerVid in signersTrain and taskVid in tasksTrain:
                idxTrain.append(idxVid)

    return np.array(idxTrain).astype(int), np.array(idxValid).astype(int), np.array(idxTest).astype(int)

def weightVectorImbalancedDataOneHot(data):
    # [samples, classes]
    # returns vector and dictionary
    dataIntegers = np.argmax(data, axis=1)
    class_weights = compute_class_weight('balanced', np.unique(dataIntegers), dataIntegers)
    return class_weights, dict(enumerate(class_weights))

def verifSets(idxTrain, idxValid, idxTest):
    interTrainValid = np.intersect1d(idxTrain, idxValid)
    interTrainTest  = np.intersect1d(idxTrain, idxTest)
    interValidTest  = np.intersect1d(idxValid, idxTest)
    if np.size(idxTrain) == 0:
        sys.exit('Train set is empty!')
    if np.size(idxValid) == 0:
        sys.exit('Valid set is empty!')
    if np.size(idxTest) == 0:
        sys.exit('Test set is empty!')
    if np.size(interTrainValid) > 0:
        print('Train and valid sets have common videos:')
        for i in interTrainValid:
            print('Video ' + str(i))
        #sys.exit()
    if np.size(interTrainTest) > 0:
        print('Train and test sets have common videos:')
        for i in interTrainTest:
            print('Video ' + str(i))
        #sys.exit()
    if np.size(interValidTest) > 0:
        print('Valid and test sets have common videos:')
        for i in interValidTest:
            print('Video ' + str(i))
        #sys.exit()
    print('Number of videos:')
    print('Train: ' + str(idxTrain.size))
    print('Valid: ' + str(idxValid.size))
    print('Test: '  + str(idxTest.size))
    print('Total: ' + str(idxTrain.size + idxValid.size + idxTest.size))

def verifSplitSettingDictaSign(videoSplitMode, tasksTrain, tasksValid, tasksTest, signersTrain, signersValid,  signersTest):
    totalSize = len(tasksTrain) + len(tasksValid) + len(tasksTest) + len(signersTrain) + len(signersValid) + len(signersTest)
    if videoSplitMode == 'auto' and totalSize > 0:
        sys.exit('Video indices can not be both automatically determined and manually specified')
    if videoSplitMode == 'manual':
        if totalSize == 0:
            sys.exit('Video indices are supposed to be manually specified')
        #if len(tasksTrain) == 0 or len(tasksValid) == 0 or len(tasksTest) == 0 or len(signersTrain) == 0 or len(signersValid) == 0 or len(signersTest) == 0:
        #    sys.exit('Empty set')

def getFeaturesDict(inputType, inputNormed):

    features_dict={'features_HS':np.array([]),
                   'features_HS_norm':np.array([]),
                   'raw':np.array([]),
                   'raw_norm':np.array([]),
                   '2Dfeatures':np.array([]),
                   '2Dfeatures_norm':np.array([])}

    if inputNormed:
        suffix='_norm'
    else:
        suffix=''
    if inputType=='2Draw':
        features_dict['raw'+suffix]         = np.sort(np.hstack([np.arange(0,14),np.arange(28,42),np.arange(42,42+68),np.arange(42+2*68,42+3*68)]))
        features_dict['features_HS'+suffix] = np.arange(122, 244)
    elif inputType=='2Draw_HS':
        features_dict['raw'+suffix]         = np.sort(np.hstack([np.arange(0,14),np.arange(28,42),np.arange(42,42+68),np.arange(42+2*68,42+3*68)]))
        features_dict['features_HS'+suffix] = np.arange(0, 244)
    elif inputType=='2Draw_HS_noOP':
        features_dict['raw'+suffix]         = np.sort(np.hstack([np.arange(0,14),np.arange(28,42),np.arange(42,42+68),np.arange(42+2*68,42+3*68)]))
        features_dict['features_HS'+suffix] = np.arange(0, 122)
    elif inputType=='2Draw_noHands':
        features_dict['raw'+suffix]         = np.sort(np.hstack([np.arange(0,14),np.arange(28,42),np.arange(42,42+68),np.arange(42+2*68,42+3*68)]))
    elif inputType=='2Dfeatures':
        features_dict['2Dfeatures'+suffix]  = np.arange(0, 96)
        features_dict['features_HS'+suffix] = np.arange(122, 244)
    elif inputType=='2Dfeatures_HS':
        features_dict['2Dfeatures'+suffix]  = np.arange(0, 96)
        features_dict['features_HS'+suffix] = np.arange(0, 244)
    elif inputType=='2Dfeatures_HS_noOP':
        features_dict['2Dfeatures'+suffix]  = np.arange(0, 96)
        features_dict['features_HS'+suffix] = np.arange(0, 122)
    elif inputType=='2Dfeatures_noHands':
        features_dict['2Dfeatures'+suffix]  = np.arange(0, 96)
    elif inputType=='3Draw':
        features_dict['raw'+suffix]         = np.arange(0, 246)
        features_dict['features_HS'+suffix] = np.arange(122, 244)
    elif inputType=='3Draw_HS':
        features_dict['raw'+suffix]         = np.arange(0, 246)
        features_dict['features_HS'+suffix] = np.arange(0, 244)
    elif inputType=='3Draw_HS_noOP':
        features_dict['raw'+suffix]         = np.arange(0, 246)
        features_dict['features_HS'+suffix] = np.arange(0, 122)
    elif inputType=='3Draw_noHands':
        features_dict['raw'+suffix]         = np.arange(0, 246)
    elif inputType=='3Dfeatures':
        features_dict['features_HS'+suffix] = np.arange(122, 420)
    elif inputType=='3Dfeatures_HS':
        features_dict['features_HS'+suffix] = np.arange(0, 420)
    elif inputType=='3Dfeatures_HS_noOP':
        features_dict['features_HS'+suffix] = np.sort(np.hstack([np.arange(0,122),np.arange(244,420)]))
    elif inputType=='3Dfeatures_noHands':
        features_dict['features_HS'+suffix] = np.arange(244, 420)


    features_number = features_dict['features_HS'].size + features_dict['features_HS_norm'].size + features_dict['raw'].size + features_dict['raw_norm'].size + features_dict['2Dfeatures'].size + features_dict['2Dfeatures_norm'].size

    return features_dict, features_number
