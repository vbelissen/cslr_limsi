import numpy as np
import sys

def framewiseAccuracy(trueData, predData, trueIsCategorical, predIsCategorical, predIsProbabilities):
    """
        Computes accuracy between annotations and predictions.

        Inputs:
            trueData: a numpy array of annotations, shape [timeSteps] (values are classes)
                or [timeSteps, nbClasses] (categorical data)
            predData: a numpy array of predictions, shape [timeSteps] (values are classes),
                or [timeSteps, nbClasses] (probabilities or categorical)
            trueIsCategorical, predIsCategorical, predIsProbabilities: bool

        Outputs:
            a single accuracy value
    """

    trueLength = trueData.shape[0]
    predLength = predData.shape[0]

    if trueLength != predLength:
        sys.exit('Annotation and prediction data should have the same length')
    if trueIsCategorical:
        trueData = np.argmax(trueData,axis=1)
    if predIsCategorical or predIsProbabilities:
        predData = np.argmax(predData,axis=1)

    return np.sum(trueData == predData)/trueLength

def framewisePRF1(trueData, predData, trueIsCategorical, predIsCategorical, predIsProbabilities):
    """
        Computes precision, recall and f1-score between annotations and predictions.
        Data must be binary.

        Inputs:
            trueData: a numpy array of annotations, shape [timeSteps] (values are classes)
                or [timeSteps, 2] (categorical data)
            predData: a numpy array of predictions, shape [timeSteps] (values are classes),
                or [timeSteps, 2] (probabilities or categorical)
            trueIsCategorical, predIsCategorical, predIsProbabilities: bool

        Outputs:
            a single accuracy value
    """

    trueLength = trueData.shape[0]
    predLength = predData.shape[0]

    if trueLength != predLength:
        sys.exit('Annotation and prediction data should have the same length')
    if np.max(trueData) > 1 or np.max(predData) > 1:
        sys.exit('Binary data required')
    if trueIsCategorical:
        if trueData.shape[1] > 2:
            sys.exit('Binary data required (2 classes)')
    if predIsCategorical or predIsProbabilities:
        if predIsCategorical.shape[1] > 2:
            sys.exit('Binary data required (2 classes)')
    if trueIsCategorical:
        trueData = np.argmax(trueData,axis=1)
    if predIsCategorical or predIsProbabilities:
        predData = np.argmax(predData,axis=1)

    TP = np.sum(trueData*predData)
    FP = np.sum((1-trueData)*predData)
    TN = np.sum((1-trueData)*(1-predData))
    FN = np.sum(trueData*(1-predData))

    if TP+FP > 0:
        P = TP/(TP+FP)
    else:
        P = 0

    if TP+FN > 0:
        R = TP/(TP+FN)
    else:
        R = 0

    if P > 0 and R > 0:
        F1 = 2*P*R/(P+R)
    else:
        F1 = 0

    return P, R, F1
