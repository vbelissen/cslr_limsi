#!/bin/bash

OUTPUTS=("fls" "FS" "DS" "PT" "FBUOY")
BATCH_SIZE=("50" "100" "200" "400")
DROPOUT=("0" "0.25" "0.5" "0.75")
RNN_NUMBER=("1" "2" "3")
HIDDEN_UNITS=("10" "50" "90")
CONV_FILTERS=("50" "200" "350")
WEIGHT_CORRECTION=("0" "0.5" "1")
SEQ_LENGTH=("50" "100" "200")
EXCLUDE_TASK_9=("0" "1")
CONFIG_SI=("0" "1" "0" "1")
CONFIG_TI=("0" "0" "1" "1")
RANDOM_SEEDS=("1" "2" "3" "4")

FIXED_SIGNERS_TRAIN="0 5 6 7 8 9 10 11 12 13 14 15"
FIXED_SIGNERS_VALID="1 2"
FIXED_SIGNERS_TEST="3 4"

lenOut=${#OUTPUTS[@]}
lenBatch=${#BATCH_SIZE[@]}
lenDrop=${#DROPOUT[@]}
lenRNNn=${#RNN_NUMBER[@]}
lenHUnits=${#HIDDEN_UNITS[@]}
lenCFilt=${#CONV_FILTERS[@]}
lenWeightC=${#WEIGHT_CORRECTION[@]}
lenSeqL=${#SEQ_LENGTH[@]}
lenExcl=${#EXCLUDE_TASK_9[@]}
lenConfigs=${#CONFIG_SI[@]}
lenRandS=${#RANDOM_SEEDS[@]}


cd ..
source activate py36_tf1
#python src/recognitionMultiDictaSignFromScript.py --epochs 30 --idxTrainBypass $({ echo {1..10}; echo {50..60}; } | tr "\n" " ") --idxValidBypass {10..20} --idxTestBypass {20..30}

for (( iB=0; iB<${lenBatch}; iB++ ));
do
  batchSize=${BATCH_SIZE[$iB]}
  python src/recognitionMultiDictaSignFromScript.py --comment "variation batch" --batchSize $batchSize --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  rm *.hdf5
done

for (( iD=0; iD<${lenDrop}; iD++ ));
do
  dropout=${DROPOUT[$iD]}
  python src/recognitionMultiDictaSignFromScript.py --comment "variation dropout" --dropout $dropout --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  rm *.hdf5
done

for (( iR=0; iR<${lenRNNn}; iR++ ));
do
  rnnNumber=${RNN_NUMBER[$iR]}
  python src/recognitionMultiDictaSignFromScript.py --comment "variation rnn number" --rnnNumber $rnnNumber --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  rm *.hdf5
done

for (( iR=0; iR<${lenHUnits}; iR++ ));
do
  rnnHiddenUnits=${HIDDEN_UNITS[$iR]}
  python src/recognitionMultiDictaSignFromScript.py --comment "variation rnn units" --rnnHiddenUnits $rnnHiddenUnits --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  rm *.hdf5
done

for (( iCOnvF=0; iCOnvF<${lenCFilt}; iCOnvF++ ));
do
  convFilt=${CONV_FILTERS[$iCOnvF]}
  python src/recognitionMultiDictaSignFromScript.py --comment "variation conv filters" --convFilt $convFilt --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  rm *.hdf5
done

for (( iWeight=0; iWeight<${lenWeightC}; iWeight++ ));
do
  weightCorrection=${WEIGHT_CORRECTION[$iWeight]}
  python src/recognitionMultiDictaSignFromScript.py --comment "variation weight correction" --weightCorrection $weightCorrection --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  rm *.hdf5
done

for (( iSeqL=0; iSeqL<${lenSeqL}; iSeqL++ ));
do
  seqLength=${SEQ_LENGTH[$iSeqL]}
  python src/recognitionMultiDictaSignFromScript.py --comment "variation seq length" --seqLength $seqLength --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  rm *.hdf5
done

for (( iExcl=0; iExcl<${lenExcl}; iExcl++ ));
do
  excludeTask9=${EXCLUDE_TASK_9[$iExcl]}
  python src/recognitionMultiDictaSignFromScript.py --comment "exclude or include task 9" --excludeTask9 $excludeTask9 --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  rm *.hdf5
done


source deactivate
