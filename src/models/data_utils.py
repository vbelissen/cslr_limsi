import numpy as np
import sys

import tensorflow as tf
v0 = tf.__version__[0]
if v0 == '2':
    from tensorflow.keras.utils import to_categorical # For tensorflow 2, keras is included in tf
elif v0 == '1':
    from keras.utils import to_categorical # For tensorflow 1.2.0
else:
    sys.exit('Tensorflow version should be 1.X or 2.X')

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


def get_annotations_videos_sign_types_binary(corpus, output_names_final, output_names_original, video_indices=np.arange(94)):
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

        Outputs:
            annotations (lists (videos) of numpy arrays)
    """
    final_number = len(output_names_final)
    video_annotations = []
    video_number = len(video_indices)
    annotation_raw = np.load('data/processed/' + corpus + '/annotations.npz', encoding='latin1', allow_pickle=True)
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

def get_annotations_videos_categories(corpus, output_names, output_categories, output_assemble=[], video_indices=np.arange(94)):
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

        Outputs:
            annotations (list (categories) of lists (videos) of numpy arrays)
    """
    annotations = []
    output_number = len(output_names)
    annotation_raw = np.load('data/processed/' + corpus + '/annotations.npz', encoding='latin1', allow_pickle=True)
    if corpus == 'DictaSign':
        string_prefix = 'dataBrut_'
    else:
        string_prefix = ''
    for i_output in range(output_number):
        if output_assemble != [] and len(output_categories[i_output]) == 1:
            annotations.append(get_annotations_videos_sign_types_binary(corpus, [output_names[i_output]], [output_assemble[i_output]], video_indices))
        else:
            annotations.append(categorical_conversion_videos(annotation_raw[string_prefix+output_names[i_output]], output_categories[i_output], video_indices))
    return annotations



def get_features_videos(corpus, features_dict={'features_HS':np.arange(0, 420), 'features_HS_norm':np.array([]), 'raw':np.array([]), 'raw_norm':np.array([])}, video_indices=np.arange(94)):
    """
        Gets all wanted features.

        Inputs:
            corpus (string)
            features_dict: a dictionary indication which features to keep
                e.g.: {'features_HS':np.arange(0, 420), 'features_HS_norm':np.array([]), 'raw':np.array([]), 'raw_norm':np.array([])}
            video_indices: list or numpy array of wanted videos

        Outputs:
            features (list  of numpy arrays [1, time_steps, features_number])
    """

    features = []

    features_number = 0
    for key in features_dict:
        features_number += features_dict[key].size

    if corpus == 'DictaSign':
        annotation_raw = np.load('data/processed/DictaSign/annotations.npz', encoding='latin1', allow_pickle=True)['dataBrut_DS'] # for counting nb of images
    elif corpus == 'NCSLGR':
        annotation_raw = np.load('data/processed/NCSLGR/annotations.npz', encoding='latin1', allow_pickle=True)['lexical_with_ns_not_fs'] # for counting nb of images
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
            key_features = np.load('data/processed/' + corpus + '/' + key + '.npy', encoding='latin1')
            index_vid_tmp = 0
            for vid_idx in video_indices:
                features[index_vid_tmp][0, :, features_number_idx:features_number_idx+key_features_number] = key_features[vid_idx][:, key_features_idx]
                index_vid_tmp += 1
            features_number_idx += key_features_number

    return features


def get_sequence_features(corpus,
                           vid_idx=0,
                           img_start_idx=0,
                           features_dict={'features_HS':np.arange(0, 420), 'features_HS_norm':np.array([]), 'raw':np.array([]), 'raw_norm':np.array([])},
                           time_steps=100,
                           preloaded_features=None):
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

        Outputs:
            X: a numpy array [1, time_steps, features_number] for features
    """
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
                key_features = np.load('data/processed/' + corpus + '/' + key + '.npy', encoding='latin1')[vid_idx]
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
                                       preloaded_annotations=None):
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

        Outputs:
            Y: a list, comprising annotations
    """

    if corpus == 'DictaSign':
        string_prefix = 'dataBrut_'
    else:
        string_prefix = ''

    Y = []

    output_number = len(output_names)
    if preloaded_annotations is None:
        annotation_output = np.load('data/processed/' + corpus + '/annotations.npz', encoding='latin1', allow_pickle=True)
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
                                               preloaded_annotations=None):
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

        Outputs:
            Y: an annotation array
    """

    if corpus == 'DictaSign':
        string_prefix = 'dataBrut_'
    else:
        string_prefix = ''

    if preloaded_annotations is None:
        Y = get_annotations_videos_sign_types_binary(corpus, output_names_final, output_names_original, [vid_idx])[0]
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
                   features_dict={'features_HS':np.arange(0, 420), 'features_HS_norm':np.array([]), 'raw':np.array([]), 'raw_norm':np.array([])},
                   time_steps=100,
                   strides=1,
                   preloaded_features=None,
                   preloaded_annotations=None):
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

        Outputs:
            X: a numpy array [1, time_steps, features_number] for features
            Y: array or list, comprising annotations
    """

    X = get_sequence_features(corpus, vid_idx, img_start_idx, features_dict, time_steps, strides, preloaded_features)
    if output_form == 'mixed':
        Y = get_sequence_annotations_categories(corpus, output_names_final, output_categories_or_names_original, vid_idx, img_start_idx, time_steps, strides, preloaded_annotations)
        #if len(Y) == 1:
        #    Y = Y[0]
    elif output_form == 'sign_types':
        Y = get_sequence_annotations_sign_types_binary(corpus, output_names_final, output_categories_or_names_original, vid_idx, img_start_idx, time_steps, strides, preloaded_annotations)
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
                                         'raw_norm':np.array([])},
                          preloaded_features=None,
                          preloaded_annotations=None,
                          video_indices=np.arange(10),
                          separation=100):
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

        Outputs:
            X: a numpy array [1, total_time_steps, features_number] for features
            Y: array or list, comprising annotations
    """

    if preloaded_features is None:
        preloaded_features = get_features_videos(corpus, features_dict, video_indices)
    if preloaded_annotations is None:
        if output_form == 'mixed':
            preloaded_annotations = get_annotations_videos_categories(corpus, output_names_final, output_categories_or_names_original, output_assemble, video_indices)
        elif output_form == 'sign_types':
            preloaded_annotations = get_annotations_videos_sign_types_binary(corpus, output_names_final, output_categories_or_names_original, video_indices)
        else: sys.exit('Invalid output form')

    video_number = video_indices.size
    video_lengths = np.zeros(video_number, dtype=int)
    total_length = 0
    for i_vid in range(video_number):
        vid_idx = video_indices[i_vid]
        video_lengths[i_vid] = preloaded_features[i_vid].shape[1]
        total_length += video_lengths[i_vid]
        total_length += separation

    features_number = preloaded_features[0].shape[2]

    X = np.zeros((1, total_length, features_number))

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
        X[0, img_start_idx:img_start_idx+video_lengths[i_vid], :] = preloaded_features[i_vid][0, :, :]
        if output_form == 'mixed':
            for i_output in range(output_number):
                Y[i_output][0, img_start_idx:img_start_idx + video_lengths[i_vid], :] = preloaded_annotations[i_output][i_vid][0, :, :]
                Y[i_output][0, img_start_idx + video_lengths[i_vid]:img_start_idx + video_lengths[i_vid]+separation, 0] = 1
        else:
            Y[0, img_start_idx:img_start_idx + video_lengths[i_vid], :] = preloaded_annotations[i_vid][0, :, :]
            Y[0, img_start_idx + video_lengths[i_vid]:img_start_idx + video_lengths[i_vid]+separation, 0] = 1
        img_start_idx += video_lengths[i_vid]
        img_start_idx += separation

    return X, Y
