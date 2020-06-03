import numpy as np
import sys
import itertools as it

import tensorflow as tf
v0 = tf.__version__[0]
if v0 == '2':
    # For tensorflow 2, keras is included in tf
    from tensorflow.keras import backend
elif v0 == '1':
    #For tensorflow 1.2.0
    from keras import backend

def framewiseAccuracy(dataTrue, dataPred, trueIsCat, predIsCatOrProb, idxNotSeparation=np.array([])):
    """
        Computes accuracy of predictions wrt annotations.

        Inputs:
            dataTrue: a numpy array of annotations, shape [timeSteps] (values are classes)
                or [timeSteps, nbClasses] (categorical data)
            dataPred: a numpy array of predictions, shape [timeSteps] (values are classes),
                or [timeSteps, nbClasses] (probabilities or categorical)
            trueIsCat, predIsCatOrProb: bool
            idxNotSeparation: binary vector indicating where separations are (0)

        Outputs:
            a single accuracy value
    """

    dataIsSeparated = (idxNotSeparation.size > 0)

    if not trueIsCat:
        if len(dataTrue.shape) > 1:
            if dataTrue.shape[1] > 1:
                sys.exit('True data should be a vector (not categorical) because trueIsCat=False')
    if not predIsCatOrProb:
        if len(dataPred.shape) > 1:
            if dataPred.shape[1] > 1:
                sys.exit('Pred data should be a vector (not categorical or probabilities) because predIsCatOrProb=False')

    if dataIsSeparated:
        dataTrue = dataTrue[idxNotSeparation]
        dataPred = dataPred[idxNotSeparation]

    trueLength = dataTrue.shape[0]
    predLength = dataPred.shape[0]

    if trueLength != predLength:
        sys.exit('Annotation and prediction data should have the same length')
    if trueIsCat:
        dataTrue = np.argmax(dataTrue,axis=1)
    if predIsCatOrProb:
        dataPred = np.argmax(dataPred,axis=1)

    return np.sum(dataTrue == dataPred)/trueLength

def framewiseAccuracyYanovich(dataTrue, dataPred, trueIsCat):
    """
        Computes accuracy of predictions wrt annotations, as defined in Yanovich et al.

        Inputs:
            dataTrue: a numpy array of annotations, shape [timeSteps] (values are classes)
                or [timeSteps, nbClasses] (categorical data)
            dataPred: a numpy array of predictions, shape [timeSteps, nbClasses] (probabilities or categorical)
            trueIsCat: bool
            idxNotSeparation: binary vector indicating where separations are (0)

        Outputs:
            a single accuracy value, and the accuracy per class (array)
    """

    if not trueIsCat:
        if len(dataTrue.shape) > 1:
            if dataTrue.shape[1] > 1:
                sys.exit('True data should be a vector (not categorical) because trueIsCat=False')


    trueLength = dataTrue.shape[0]
    predLength = dataPred.shape[0]

    nClasses = dataPred.shape[1]

    if trueLength != predLength:
        sys.exit('Annotation and prediction data should have the same length')
    if trueIsCat:
        dataTrue = np.argmax(dataTrue[:,1:],axis=1)
    else:
        dataTrue = dataTrue - 1
    dataPred = np.argmax(dataPred[:,1:],axis=1)

    timestepsPerClass = np.zeros(nClasses-1)
    accPerClass = np.zeros(nClasses-1)
    for iC in range(nClasses-1):
        trueC = (dataTrue == iC)
        predC = (dataPred == iC)
        trueAndPredC = np.sum(trueC * predC)
        timestepsPerClass[iC] = np.sum(trueC)
        accPerClass[iC] = trueAndPredC/timestepsPerClass[iC]

    timestepsTotalNotZero = np.sum(timestepsPerClass)

    return np.sum(timestepsPerClass*accPerClass)/timestepsTotalNotZero, accPerClass



def ignore_acc(y_true, y_pred):
    y_true_class = backend.argmax(y_true, axis=-1)
    y_pred_class = backend.argmax(y_pred, axis=-1)

    ignore_mask = backend.cast(backend.not_equal(y_pred_class, class_to_ignore), 'int32')
    matches = backend.cast(backend.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
    accuracy = backend.sum(matches) / backend.maximum(backend.sum(ignore_mask), 1)
    return accuracy


def framewisePRF1(dataTrue, dataPred, trueIsCat, predIsCatOrProb, idxNotSeparation=np.array([])):
    """
        Computes precision, recall and f1-score of predictions wrt annotations.
        Data must be binary.

        Inputs:
            dataTrue: a numpy array of annotations, shape [timeSteps] (values are classes)
                or [timeSteps, 2] (categorical data)
            dataPred: a numpy array of predictions, shape [timeSteps] (values are classes),
                or [timeSteps, 2] (probabilities or categorical)
            trueIsCat, predIsCatOrProb: bool (if annotations are categorical,
                if predictions are categorical/probability values for each category)
            idxNotSeparation: binary vector indicating where separations are (0)
        Outputs:
            a single accuracy value
    """

    trueLength = dataTrue.shape[0]
    predLength = dataPred.shape[0]

    if not trueIsCat:
        if len(dataTrue.shape) > 1:
            if dataTrue.shape[1] > 1:
                sys.exit('True data should be a vector (not categorical) because trueIsCat=False')
    if not predIsCatOrProb:
        if len(dataPred.shape) > 1:
            if dataPred.shape[1] > 1:
                sys.exit('Pred data should be a vector (not categorical or probabilities) because predIsCatOrProb=False')

    if trueLength != predLength:
        sys.exit('Annotation and prediction data should have the same length')
    if np.max(dataTrue) > 1 or np.max(dataPred) > 1:
        sys.exit('Binary data required')
    if trueIsCat:
        if dataTrue.shape[1] > 2:
            sys.exit('Binary data required (2 classes)')
    if predIsCatOrProb:
        if dataPred.shape[1] > 2:
            sys.exit('Binary data required (2 classes)')
    if trueIsCat:
        dataTrue = np.argmax(dataTrue,axis=1)
    if predIsCatOrProb:
        dataPred = np.argmax(dataPred,axis=1)

    if dataIsSeparated:
        dataTrue = dataTrue[idxNotSeparation]
        dataPred = dataPred[idxNotSeparation]

    TP = np.sum(dataTrue*dataPred)
    FP = np.sum((1-dataTrue)*dataPred)
    TN = np.sum((1-dataTrue)*(1-dataPred))
    FN = np.sum(dataTrue*(1-dataPred))

    if TP+FP > 0:
        P = TP/(TP+FP)
    else:
        P = 0

    if TP+FN > 0:
        R = TP/(TP+FN)
    else:
        R = 0

    if P+R > 0:
        F1 = 2*P*R/(P+R)
    else:
        F1 = 0

    return P, R, F1

def valuesConsecutive(data, isCatOrProb):
    """
        Returns list of consecutive values
        (value, start, end (+1), nb of values) (excluding zero values)

        Inputs:
            data: a numpy array of annotations/predictions, shape [timeSteps] (values are classes)
                or [timeSteps, nbClasses] (probabilities or categorical)
            isCatOrProb: bool

        Outputs:
            a list of lists with 4 elements
    """

    if not isCatOrProb:
        if len(data.shape) > 1:
            if data.shape[1] > 1:
                sys.exit('data should be a vector (not categorical or probabilities) because isCatOrProb=False')

    if isCatOrProb:
        data = np.argmax(data,axis=1)
    #
    g = it.groupby(enumerate(data), lambda x:x[1])
    l = [(x[0], list(x[1])) for x in g if x[0] != 0]
    #[(x[0], len(x[1]), x[1][0][0]) for x in l]
    return [(x[0], x[1][0][0], x[1][0][0]+len(x[1]), len(x[1])) for x in l]

def windowUnitsPredForTrue(iTrue, nbUnitsTrue, nbUnitsPred, fractionTotal):
    """
        Returns a window of pred units indices to look for

        Inputs:


        Outputs:

    """
    iPredApprox = iTrue*nbUnitsPred/nbUnitsTrue
    windowSizeApprox = fractionTotal*nbUnitsPred
    min = np.max([0, round(iPredApprox - windowSizeApprox/2)])
    max = np.min([nbUnitsPred, round(iPredApprox + windowSizeApprox/2)])
    return min, max

def matrixMatch(consecTrue, consecPred, seqLength, fractionTotal):
    """
        Returns matrix of match score to calculate best matches
        (value, start, end (+1), nb of values) (excluding zero values)

        Inputs:
            consecTrue: list of consecutive values
            (value, start, end (+1), nb of values) (excluding zero values)
            consecPred: list of consecutive values
            (value, start, end (+1), nb of values) (excluding zero values)
            seqLength: original length of sequence
            fractionTotal: used to define a window, to not calculate all matrix values

        Outputs:
            a matrix of match scores (Wolf measure - normalized intersection between units)
    """
    #l = dataTrue.size
    l = seqLength
    tempVectorTrue = np.ones(l)
    tempVectorPred = np.ones(l)
    tempVector = np.ones(l)
    #consecTrue = valuesConsecutive(dataTrue)
    #consecPred = valuesConsecutive(dataPred)
    nbUnitsTrue = len(consecTrue)
    nbUnitsPred = len(consecPred)
    matrixM = np.zeros((nbUnitsTrue,nbUnitsPred))
    for iTrue in range(nbUnitsTrue):
        valuesUnitTrue = consecTrue[iTrue]
        tempVectorTrue[:valuesUnitTrue[1]] = 0
        tempVectorTrue[valuesUnitTrue[2]:] = 0
        min, max = windowUnitsPredForTrue(iTrue, nbUnitsTrue, nbUnitsPred, fractionTotal)
        for iPred in range(min, max):#range(nbUnitsPred):
            valuesUnitPred = consecPred[iPred]
            tempVectorPred[:valuesUnitPred[1]] = 0
            tempVectorPred[valuesUnitPred[2]:] = 0
            tempVector = tempVectorTrue*tempVectorPred
            matrixM[iTrue,iPred] = 2 * (valuesUnitTrue[0] == valuesUnitPred[0]) * np.sum(tempVector)/(valuesUnitTrue[3] + valuesUnitPred[3])
            tempVectorPred[:] = 1
        tempVectorTrue[:] = 1
    return matrixM

def idxBestMatches(dataTrue, dataPred, matMatch, trueIsCat, predIsCatOrProb):
    """
        Returns best matches for each true unit, and for each detected unit

        Inputs:
            dataTrue: a numpy array of annotations, shape [timeSteps] (values are classes)
                or [timeSteps, 2] (categorical data)
            dataPred: a numpy array of predictions, shape [timeSteps] (values are classes),
                or [timeSteps, 2] (probabilities or categorical)
            matMatch: matrix of match scores
            trueIsCat, predIsCatOrProb: bool (if annotations are categorical,
                if predictions are categorical/probability values for each category)

        Outputs:
            two numpy arrays
    """

    if not trueIsCat:
        if len(dataTrue.shape) > 1:
            if dataTrue.shape[1] > 1:
                sys.exit('True data should be a vector (not categorical) because trueIsCat=False')
    if not predIsCatOrProb:
        if len(dataPred.shape) > 1:
            if dataPred.shape[1] > 1:
                sys.exit('Pred data should be a vector (not categorical or probabilities) because predIsCatOrProb=False')

    #matMatch = matrixMatch(valuesConsecutive(dataTrue, trueIsCat), valuesConsecutive(dataPred, predIsCatOrProb), dataTrue.shape[0])
    return np.argmax(matMatch,axis=0), np.argmax(matMatch,axis=1)

def isMatched(idxTrue, idxPred, tp, tr, consecTrue, consecPred, seqLength):
    """
        Returns 1 if two units match, with thresholds tp and tr (Wolf measure)

        Inputs:
            idxTrue: index of true unit
            idxPred: index of pred unit
            tp: threshold (between 0 and 1)
            tr: threshold (between 0 and 1)
            consecTrue: list of consecutive values
            (value, start, end (+1), nb of values) (excluding zero values)
            consecPred: list of consecutive values
            (value, start, end (+1), nb of values) (excluding zero values)
            seqLength: original length of sequence

        Outputs:
            a matrix of match scores (Wolf measure - normalized intersection between units)
    """
    l = seqLength
    tempVectorTrue = np.ones(l)
    tempVectorPred = np.ones(l)
    valuesUnitTrue = consecTrue[idxTrue]
    valuesUnitPred = consecPred[idxPred]
    tempVectorTrue[:valuesUnitTrue[1]] = 0
    tempVectorTrue[valuesUnitTrue[2]:] = 0
    tempVectorPred[:valuesUnitPred[1]] = 0
    tempVectorPred[valuesUnitPred[2]:] = 0
    tempVector = tempVectorTrue*tempVectorPred
    intersect = np.sum(tempVectorTrue * tempVectorPred)
    if intersect/valuesUnitPred[3] > tp and intersect/valuesUnitTrue[3] > tr and valuesUnitTrue[0] == valuesUnitPred[0]:
        return 1
    else:
        return 0

def prfStar(dataTrue, dataPred, trueIsCat, predIsCatOrProb, step=0.01, fractionTotal=0.1):
    """
        Returns P, R, F1 for thresholds (tp, 0) and (0, tr)

        Inputs:
            dataTrue: a numpy array of annotations, shape [timeSteps] (values are classes)
                or [timeSteps, 2] (categorical data)
            dataPred: a numpy array of predictions, shape [timeSteps] (values are classes),
                or [timeSteps, 2] (probabilities or categorical)
            trueIsCat, predIsCatOrProb: bool (if annotations are categorical,
                if predictions are categorical/probability values for each category)
            step: between tp and tr values

        Outputs:
            P*(tp,0), P*(0,tr), R*(tp,0), R*(0,tr), F1*(tp,0), F1*(0,tr)
    """
    # Returns

    if not trueIsCat:
        if len(dataTrue.shape) > 1:
            if dataTrue.shape[1] > 1:
                sys.exit('True data should be a vector (not categorical) because trueIsCat=False')
    if not predIsCatOrProb:
        if len(dataPred.shape) > 1:
            if dataPred.shape[1] > 1:
                sys.exit('Pred data should be a vector (not categorical or probabilities) because predIsCatOrProb=False')()

    seqLength = dataTrue.shape[0]
    tpVector = np.arange(0,1+step,step)
    trVector = np.arange(0,1+step,step)
    nbValues = tpVector.size

    pStarTp = np.zeros(nbValues)
    pStarTr = np.zeros(nbValues)
    rStarTp = np.zeros(nbValues)
    rStarTr = np.zeros(nbValues)
    fStarTp = np.zeros(nbValues)
    fStarTr = np.zeros(nbValues)

    consecTrue = valuesConsecutive(dataTrue, trueIsCat)
    consecPred = valuesConsecutive(dataPred, predIsCatOrProb)

    nbUnitsTrue = len(consecTrue)
    nbUnitsPred = len(consecPred)

    M = matrixMatch(consecTrue, consecPred, seqLength, fractionTotal)

    idxBestMatchesTrue, idxBestMatchesPred = idxBestMatches(dataTrue, dataPred, M, trueIsCat, predIsCatOrProb)

    for iPred in range(nbUnitsPred):
        idxBestMatchTrue = idxBestMatchesTrue[iPred]
        for iTp in range(nbValues):
            pStarTp[iTp] += isMatched(idxBestMatchTrue, iPred, tpVector[iTp], 0, consecTrue, consecPred, seqLength)
        for iTr in range(nbValues):
            pStarTr[iTr] += isMatched(idxBestMatchTrue, iPred, 0, trVector[iTr], consecTrue, consecPred, seqLength)
    pStarTp /= nbUnitsPred
    pStarTr /= nbUnitsPred

    for iTrue in range(nbUnitsTrue):
        idxBestMatchPred = idxBestMatchesPred[iTrue]
        for iTp in range(nbValues):
            rStarTp[iTp] += isMatched(iTrue, idxBestMatchPred, tpVector[iTp], 0, consecTrue, consecPred, seqLength)
        for iTr in range(nbValues):
            rStarTr[iTr] += isMatched(iTrue, idxBestMatchPred, 0, trVector[iTr], consecTrue, consecPred, seqLength)
    rStarTp /= nbUnitsTrue
    rStarTr /= nbUnitsTrue

    fStarTp = 2 * 1. / (1. / pStarTp + 1. / rStarTp)
    fStarTr = 2 * 1. / (1. / pStarTr + 1. / rStarTr)

    #fStarTp = 2 * 1. / (1. / pStarTp + 1. / rStarTp)
    #fStarTr = 2 * 1. / (1. / pStarTr + 1. / rStarTr)

    return pStarTp, pStarTr, rStarTp, rStarTr, fStarTp, fStarTr

def integralValues(fTp, fTr, step=0.01):
    """
        Returns P, R, F1 for thresholds (tp, 0) and (0, tr)

        Inputs:
            fTp: a numpy array of F1*(tp,0)
            fTr: a numpy array of F1*(0,tr)
            step: between tp and tr values

        Outputs:
            Ip, Ir, Ipr=avg
    """
    Ip = 0
    Ir = 0
    t = np.arange(0,1+step,step)
    nbValues = t.size
    for i in range(nbValues-1):
        midTp = 0.5 * (fTp[i] + fTp[i+1])
        midTr = 0.5 * (fTr[i] + fTr[i+1])
        Ip += midTp*step
        Ir += midTr*step
    return Ip, Ir, 0.5*(Ip+Ir)
