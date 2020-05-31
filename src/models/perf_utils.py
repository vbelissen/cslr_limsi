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
