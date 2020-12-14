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

def categorical_conversion_seq(data, nonZeroCategories=[1]):
    """
        Converting annotations of one type to categorical values (for a sequence or a video).

        Inputs:
            data: a numpy array of annotations [time_steps, 1]
            nonZeroCategories: list of meaningful annotation categories

        Outputs:
            converted_data: categorical values of annotations [1, time_steps, categories]
    """
    nonZeroCategories_number = len(nonZeroCategories)
    if nonZeroCategories_number > 1:
        mask_class_garbage = np.ones(data.shape).astype(bool)
        for i_c in range(1, nonZeroCategories_number):
            mask_class_garbage *= (data != nonZeroCategories[i_c - 1])
        data[mask_class_garbage] = 0
        for i_c in range(1, nonZeroCategories_number):
            data[data == nonZeroCategories[i_c - 1]] = i_c
    converted_data = np.zeros((1, data.shape[0], nonZeroCategories_number+1))
    converted_data[0, :, :] = to_categorical(data, nonZeroCategories_number+1)
    return converted_data


def binary_conversion_seq(data):
    """
        Converting annotations of one type, with different values, to binary values (for a sequence or a video).

        Inputs:
            data: a numpy array of annotations [time_steps, 1] or [time_steps]

        Outputs:
            converted_data: categorical values of annotations [1, time_steps, 1]
    """
    time_steps = data.shape[0]
    converted_data = np.zeros((1, time_steps, 1))
    converted_data[0, :, 0] = (data > 0).reshape(time_steps)
    return converted_data.astype(float)


def categorical_conversion_videos(data, nonZeroCategories=[1], video_indices=None):
    """
        Converting annotations of one type to categorical values (for all videos).

        Inputs:
            data: a list of numpy array of annotations (for each video)
            nonZeroCategories: list of meaningful annotation categories
            video_indices: list or numpy array of wanted videos

        Outputs:
            data_copy: list of categorical values of annotations
    """
    data_copy = []
    video_number = len(data)
    if video_indices is None:
        range_idx = range(video_number)
    else:
        range_idx = video_indices
    for i_video in range_idx:
        data_copy.append(categorical_conversion_seq(data[i_video], nonZeroCategories))
    return data_copy


def get_annotations_videos_sign_types_binary(corpus, output_names_final, output_names_original, video_indices=np.arange(94), from_notebook=False):
    """
        Gets all wanted annotations, in the form of a list of video annotations of sign types.
            e.g.: get_annotations_videos_sign_types_binary('NCSLGR',
            ['Pointing', 'Classifiers'],
            [['IX_1p', 'IX_2p', 'IX_3p', 'IX_loc'], ['DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']],
            np.arange(10))

        Inputs:
            corpus (string): 'DictaSign' or 'NCSLGR'
            output_names_final: list of outputs (strings) corresponding to the desired output_categories
            output_names_original: original names that are used to compose final outputs
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
            video_indices: list or numpy array of wanted videos
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            annotations (lists (videos) of numpy arrays)
    """

    if from_notebook:
        parent = '../'
    else:
        parent = ''

    final_number = len(output_names_final)
    video_annotations = []
    video_number = len(video_indices)
    annotation_raw = np.load(parent + 'data/processed/' + corpus + '/annotations.npz', encoding='latin1', allow_pickle=True)
    if corpus == 'DictaSign':
        string_prefix = 'dataBrut_'
    else:
        string_prefix = ''
    for i_vid in range(video_number):
        vid_idx = video_indices[i_vid]
        video_length = annotation_raw[string_prefix+output_names_original[0][0]][vid_idx].shape[0]
        current_video_annotations = np.zeros((1, video_length, final_number+1))
        for i_final_output in range(final_number):
            original_number = len(output_names_original[i_final_output])
            for i_original_output in range(original_number):
                current_video_annotations[0, :, i_final_output+1] += binary_conversion_seq(annotation_raw[string_prefix+output_names_original[i_final_output][i_original_output]][vid_idx]).reshape(video_length)
        current_video_annotations = (current_video_annotations > 0)
        current_video_annotations[0, :, 0] = 1 - (np.sum(current_video_annotations[0, :, 1:], axis=1)>0)
        video_annotations.append(current_video_annotations.astype(float))
    return video_annotations

def get_annotations_videos_categories(corpus, output_names, output_categories, output_assemble=[], video_indices=np.arange(94), from_notebook=False):
    """
        Gets all wanted annotations, in the form of a list of different categories, each of which is a list of video annotations.
            e.g.: get_annotations_videos_categories('DictaSign', ['fls', 'DS'], [[41891,43413,43422,42992],[1]], video_indices=np.arange(10))
                  get_annotations_videos_categories('NCSLGR',['PT', 'DS', 'fls'], [[1], [1], [1]], output_assemble=[['IX_1p', 'IX_2p', 'IX_3p'], [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'], ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']], video_indices=np.arange(10))

        Inputs:
            corpus (string): 'DictaSign' or 'NCSLGR'
            output_names: list of outputs (strings)
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
                if output_assemble =! [] and binary category, names are not really used
            output_categories: list of lists of meaningful annotation categories for each output
            output_assemble: used only if output_form: 'mixed'. List of lists of original names that can be assembled to compose final outputs, only considered if binary annotation category
            video_indices: list or numpy array of wanted videos
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            annotations (list (categories) of lists (videos) of numpy arrays)
    """

    if from_notebook:
        parent = '../'
    else:
        parent = ''

    annotations = []
    output_number = len(output_names)
    annotation_raw = np.load(parent + 'data/processed/' + corpus + '/annotations.npz', encoding='latin1', allow_pickle=True)
    if corpus == 'DictaSign':
        string_prefix = 'dataBrut_'
    else:
        string_prefix = ''
    for i_output in range(output_number):
        if output_assemble != [] and len(output_categories[i_output]) == 1:
            annotations.append(get_annotations_videos_sign_types_binary(corpus, [output_names[i_output]], [output_assemble[i_output]], video_indices, from_notebook))
        else:
            annotations.append(categorical_conversion_videos(annotation_raw[string_prefix+output_names[i_output]], output_categories[i_output], video_indices))
    return annotations



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
            key_features = np.load(parent + 'data/processed/' + corpus + '/' + key + '.npy', encoding='latin1')
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
        For returning features for a sequence.

        Inputs:
            corpus (string)
            output_weights: list of weights for each_output
            vid_idx (int): which video
            img_start_idx (int): which start image
            features_dict: a dictionary indication which features to keep
                e.g.: {'features_HS':np.arange(0, 420), 'features_HS_norm':np.array([]), 'raw':np.array([]), 'raw_norm':np.array([])}
            time_steps: length of sequences (int)
            preloaded_features: if features are already loaded, in the format of a list (features for each video)
            preloaded_annotations: if annotations are already loaded, in the format of a list of lists (for each output type, each video), categorical values
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
                key_features = np.load(parent + 'data/processed/' + corpus + '/' + key + '.npy', encoding='latin1')[vid_idx]
                X[0, img_start_idx:img_start_idx + time_steps, features_number_idx:features_number_idx+key_features_number] = key_features[img_start_idx:img_start_idx + time_steps, key_features_idx]
                features_number_idx += key_features_number
    else:
        X[0, img_start_idx:img_start_idx+time_steps, :] = preloaded_features[vid_idx][0, img_start_idx:img_start_idx+time_steps, :]

    return X

def get_sequence_annotations_categories(corpus,
                                       output_names,
                                       output_categories,
                                       vid_idx=0,
                                       img_start_idx=0,
                                       time_steps=100,
                                       strides=1,
                                       preloaded_annotations=None,
                                       from_notebook=False):
    """
        For returning annotations for a sequence, in the form of a list of different categories.
            e.g.: get_sequence_annotations_categories('DictaSign',
                                         ['fls', 'DS'],
                                         [[41891,43413,43422,42992],[1]],
                                         vid_idx=17,
                                         img_start_idx=258,
                                         time_steps=100,
                                         strides=1)

        Inputs:
            output_names: list of outputs (strings)
            output_categories: list of lists of meaningful annotation categories for each output
            vid_idx (int): which video
            img_start_idx (int): which start image
            time_steps: length of sequences (int)
            strides: size of convolution strides
            preloaded_annotations: if annotations are already loaded, in the format of a list of lists (for each output type, each video), categorical values
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            Y: a list, comprising annotations
    """

    if from_notebook:
        parent = '../'
    else:
        parent = ''

    if corpus == 'DictaSign':
        string_prefix = 'dataBrut_'
    else:
        string_prefix = ''

    Y = []

    output_number = len(output_names)
    if preloaded_annotations is None:
        annotation_output = np.load(parent + 'data/processed/' + corpus + '/annotations.npz', encoding='latin1', allow_pickle=True)
    for i_output in range(output_number):
        output_classes = len(output_categories[i_output])+1
        if preloaded_annotations is None:
            Y.append(categorical_conversion_seq(annotation_output[string_prefix+output_names[i_output]][vid_idx][img_start_idx:img_start_idx + time_steps, :], nonZeroCategories=output_categories[i_output]).reshape(1, time_steps, output_classes))
        else:
            Y.append(preloaded_annotations[i_output][vid_idx][0, img_start_idx:img_start_idx+time_steps, :].reshape(1, time_steps, output_classes))
        if strides > 1:
            Y[-1] = np.roll(Y[-1], -int(strides / 2), axis=1)[:, ::strides, :]

    return Y


def get_sequence_annotations_sign_types_binary(corpus,
                                               output_names_final,
                                               output_names_original,
                                               vid_idx=0,
                                               img_start_idx=0,
                                               time_steps=100,
                                               strides=1,
                                               preloaded_annotations=None,
                                               from_notebook=False):
    """
        For returning annotations for a sequence, in the form of a list of different categories, each of which is a list of video annotations.
            e.g.: get_sequence_annotations('DictaSign',
                                         ['fls', 'DS'],
                                         [[41891,43413,43422,42992],[1]],
                                         vid_idx=17,
                                         img_start_idx=258,
                                         time_steps=100,
                                         strides=1)

        Inputs:
            corpus (string)
            output_names_final: list of outputs (strings) corresponding to the desired output_categories
            output_names_original: original names that are used to compose final outputs
                DictaSign: subset of ['fls' (with different categories), 'PT', 'PT_PRO1', 'PT_PRO2', 'PT_PRO3', 'PT_LOC', 'PT_DET', 'PT_LBUOY', 'PT_BUOY', 'DS', 'DSA', 'DSG', 'DSL', 'DSM', 'DSS', 'DST', 'DSX', 'FBUOY', 'N', 'FS']
                NCSLGR: subset of ['other', 'lexical_with_ns_not_fs' (only 0/1), 'fingerspelling', 'fingerspelled_loan_signs', 'IX_1p', 'IX_2p', 'IX_3p', 'IX_loc', 'POSS', 'SELF', 'gesture', 'part_indef', 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL']
            vid_idx (int): which video
            img_start_idx (int): which start image
            time_steps: length of sequences (int)
            strides: size of convolution strides
            preloaded_annotations: if annotations are already loaded, in the format of a list (for each video)
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            Y: an annotation array
    """

    if corpus == 'DictaSign':
        string_prefix = 'dataBrut_'
    else:
        string_prefix = ''

    if preloaded_annotations is None:
        Y = get_annotations_videos_sign_types_binary(corpus, output_names_final, output_names_original, [vid_idx], from_notebook)[0]
    else:
        Y = preloaded_annotations[vid_idx]

    output_classes = Y.shape[2]

    return Y[0, img_start_idx:img_start_idx+time_steps, :].reshape(1, time_steps, output_classes)


def get_sequence(corpus,
                   output_form,
                   output_names_final,
                   output_categories_or_names_original,
                   vid_idx=0,
                   img_start_idx=0,
                   features_dict={'features_HS':np.arange(0, 420),
                                  'features_HS_norm':np.array([]),
                                  'raw':np.array([]),
                                  'raw_norm':np.array([]),
                                  '2Dfeatures':np.array([]),
                                  '2Dfeatures_norm':np.array([])},
                   time_steps=100,
                   strides=1,
                   preloaded_features=None,
                   preloaded_annotations=None,
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
            strides: size of convolution strides
            preloaded_features: if features are already loaded, in the format of a list (features for each video)
            preloaded_annotations: if annotations are already loaded, in the format of a list of lists (for each output type, each video), categorical values
            from_notebook: if notebook script, data is in parent folder

        Outputs:
            X: a numpy array [1, time_steps, features_number] for features
            Y: array or list, comprising annotations
    """

    X = get_sequence_features(corpus, vid_idx, img_start_idx, features_dict, time_steps, strides, preloaded_features, from_notebook)
    if output_form == 'mixed':
        Y = get_sequence_annotations_categories(corpus,
                                                output_names_final,
                                                output_categories_or_names_original,
                                                vid_idx,
                                                img_start_idx,
                                                time_steps,
                                                strides,
                                                preloaded_annotations,
                                                rom_notebook)
        #if len(Y) == 1:
        #    Y = Y[0]
    elif output_form == 'sign_types':
        Y = get_sequence_annotations_sign_types_binary(corpus,
                                                       output_names_final,
                                                       output_categories_or_names_original,
                                                       vid_idx,
                                                       img_start_idx,
                                                       time_steps,
                                                       strides,
                                                       preloaded_annotations,
                                                       from_notebook)
    else:
        sys.exit('Invalid output form')

    return X, Y

def get_data_concatenated(corpus,
                          output_form,
                          output_names_final,
                          output_categories_or_names_original,
                          output_assemble=[],
                          features_dict={'features_HS':np.arange(0, 420),
                                         'features_HS_norm':np.array([]),
                                         'raw':np.array([]),
                                         'raw_norm':np.array([]),
                                         '2Dfeatures':np.array([]),
                                         '2Dfeatures_norm':np.array([])},
                          preloaded_features=None,
                          preloaded_annotations=None,
                          video_indices=np.arange(10),
                          separation=100,
                          from_notebook=False,
                          return_idx_trueData=False,
                          features_type='features',
                          frames_path_before_video='/localHD/DictaSign/convert/img/DictaSign_lsf_',
                          empty_image_path='/localHD/DictaSign/convert/img/white.jpg'):
    """
        For returning concatenated features and annotations for a set of videos (e.g. train set...).
            e.g. features_2_train, annot_2_train = get_data_concatenated('NCSLGR',
                                                                         'sign_types',
                                                                         ['PT', 'DS', 'fls'],
                                                                         [['IX_1p', 'IX_2p', 'IX_3p'],
                                                                          [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'],
                                                                          ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']])
                 features_1_train, annot_1_train = get_data_concatenated('DictaSign',
                                                                         'mixed',
                                                                         ['PT', 'DS', 'fls'],
                                                                         [[1], [1], [41891,43413,43422,42992]])
                 features_3_train, annot_3_train = get_data_concatenated('NCSLGR',
                                                                         'mixed',
                                                                         ['PT', 'DS', 'fls'], [[1], [1], [1]],
                                                                         output_assemble=[['IX_1p', 'IX_2p', 'IX_3p'],
                                                                                          [ 'DCL', 'LCL', 'SCL', 'BCL', 'ICL', 'BPCL', 'PCL'], ['lexical_with_ns_not_fs', 'fingerspelling', 'fingerspelled_loan_signs']])

        Inputs:
            corpus (string)
            output_form: 'mixed' if different and separated Outputs
                         'sign_types' if annotation is only a binary matrix of sign types
            output_names_final: list of outputs (strings)
            output_categories_or_names_original:
                if output_form: 'mixed': list of lists of meaningful annotation categories for each output
                if output_form: 'sign_types': list of lists of original names that are used to compose final outputs
            output_assemble: used only if output_form: 'mixed'. List of lists of original names that can be assembled to compose final outputs, only considered if binary annotation category
            features_dict: a dictionary indication which features to keep
                e.g.: {'features_HS':np.arange(0, 420), 'features_HS_norm':np.array([]), 'raw':np.array([]), 'raw_norm':np.array([])}
            preloaded_features: if features are already loaded, in the format of a list (features for each video)
            preloaded_annotations: if annotations are already loaded, in the format of a list of lists (for each output type, each video), categorical values
            video_indices: numpy array for a list of videos
            separation: in order to separate consecutive videos
            from_notebook: if notebook script, data is in parent folder
            return_idx_trueData: if True, returns a binary vector with 0 where separations are
            features_type: 'features', 'frames', 'both'

        Outputs:
            X: [a numpy array [1, total_time_steps, features_number] for features,
                a list of frame paths]
            Y: array or list, comprising annotations
    """

    if from_notebook:
        parent = '../'
    else:
        parent = ''

    list_videos = np.load(parent+'data/processed/DictaSign/list_videos.npy')

    video_number = video_indices.size
    video_lengths = np.zeros(video_number, dtype=int)
    total_length = 0

    if preloaded_features is None and features_type != 'frames':
        preloaded_features = get_features_videos(corpus, features_dict, video_indices, from_notebook)
    if preloaded_annotations is None:
        if output_form == 'mixed':
            preloaded_annotations = get_annotations_videos_categories(corpus,
                                                                      output_names_final,
                                                                      output_categories_or_names_original,
                                                                      output_assemble,
                                                                      video_indices,
                                                                      from_notebook)
            for i_vid in range(video_number):
                vid_idx = video_indices[i_vid]
                video_lengths[i_vid] = preloaded_annotations[0][i_vid].shape[1]
                total_length += video_lengths[i_vid]
                total_length += separation
        elif output_form == 'sign_types':
            preloaded_annotations = get_annotations_videos_sign_types_binary(corpus,
                                                                             output_names_final,
                                                                             output_categories_or_names_original,
                                                                             video_indices,
                                                                             from_notebook)
            for i_vid in range(video_number):
                vid_idx = video_indices[i_vid]
                video_lengths[i_vid] = preloaded_annotations[i_vid].shape[1]
                total_length += video_lengths[i_vid]
                total_length += separation
        else: sys.exit('Invalid output form')


    if features_type == 'features' or features_type == 'both':
        features_number = preloaded_features[0].shape[2]
        X_features = np.zeros((1, total_length, features_number))
    else:
        X_features = np.array([])

    X_frames = np.repeat('',total_length).astype('<U100')

    idx_trueData = np.zeros(total_length)

    output_number = len(output_names_final)
    if output_form == 'mixed':
        Y = []
        for i_output in range(output_number):
            output_classes = len(output_categories_or_names_original[i_output]) + 1
            Y.append(np.zeros((1, total_length, output_classes)))
    else:
        Y = np.zeros((1, total_length, output_number + 1))

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
        if output_form == 'mixed':
            for i_output in range(output_number):
                Y[i_output][0, img_start_idx:img_start_idx + video_lengths[i_vid], :] = preloaded_annotations[i_output][i_vid][0, :, :]
                Y[i_output][0, img_start_idx + video_lengths[i_vid]:img_start_idx + video_lengths[i_vid]+separation, 0] = 1
        else:
            Y[0, img_start_idx:img_start_idx + video_lengths[i_vid], :] = preloaded_annotations[i_vid][0, :, :]
            Y[0, img_start_idx + video_lengths[i_vid]:img_start_idx + video_lengths[i_vid]+separation, 0] = 1
        img_start_idx += video_lengths[i_vid]
        img_start_idx += separation

    if return_idx_trueData:
        return [X_features, X_frames], Y, idx_trueData
    else:
        return [X_features, X_frames], Y


def getVideoIndicesSplitNCSLGR(fractionValid=0.10,
                               fractionTest=0.05,
                               videosToDelete=['dorm_prank_1053_small_0_1.mov', 'DSP_DeadDog.mov', 'DSP_Immigrants.mov', 'DSP_Trip.mov'],
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

def getVideoIndicesSplitDictaSign(tasksTrain, tasksValid, tasksTest, signersTrain, signersValid, signersTest, signerIndependent, taskIndependent, excludeTask9, videoSplitMode, fractionValid, fractionTest, checkSplits=False, checkSets=False):
    if videoSplitMode == 'manual':
        idxTrain, idxValid, idxTest = getVideoIndicesSplitDictaSignManual([tasksTrain,tasksValid,tasksTest],
                                                                          [signersTrain,signersValid,signersTest],
                                                                          excludeTask9)
    elif videoSplitMode == 'auto':
        idxTrain, idxValid, idxTest = getVideoIndicesSplitDictaSignAuto(signerIndependent, taskIndependent, excludeTask9, fractionValid, fractionTest)
    else:
        sys.exit('videoSplitMode should be either manual or auto')

    if checkSplits:
        verifSplitSettingDictaSign(videoSplitMode,
                                   tasksTrain,    tasksValid,    tasksTest,
                                   signersTrain,  signersValid,  signersTest)
    if checkSets:
        verifSets(idxTrain, idxValid, idxTest)

    return idxTrain, idxValid, idxTest

def getVideoIndicesSplitDictaSignManual(tasksSplit, signersSplit, excludeTask9, from_notebook=False):#sessionsSplit,
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

def getVideoIndicesSplitDictaSignAuto(signerIndependent, taskIndependent, excludeTask9, fractionValid, fractionTest, from_notebook=False):

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
