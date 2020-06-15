#!/bin/bash

OUTPUTS=("fls" "DS" "PT" "FBUOY")
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
#python src/recognitionUniqueDictaSignFromScript.py --epochs 30 --idxTrainBypass $({ echo {1..10}; echo {50..60}; } | tr "\n" " ") --idxValidBypass {10..20} --idxTestBypass {20..30}

#for (( iOutput=0; iOutput<${tLen}; iOutput++ ));
for (( iOutput=0; iOutput<${lenOut}; iOutput++ ));
do
  output=${OUTPUTS[$iOutput]}
  echo Output: $output#
  #output=${OUTPUTS[$iOutput]}
  for (( iConfig=0; iConfig<${lenConfigs}; iConfig++ ));
  do
    config_SI=${CONFIG_SI[$iConfig]}
    config_TI=${CONFIG_TI[$iConfig]}
    for (( iRand=0; iRand<${lenRandS}; iRand++ ));
    do
      randSeed=${RANDOM_SEEDS[$iRand]}
      python src/recognitionUniqueDictaSignFromScript.py --outputName $output --comment "variation config SI TI v2" --randSeed $randSeed --epochs 150 --videoSplitMode auto --signerIndependent $config_SI --taskIndependent $config_TI --excludeTask9 1 --fractionTest 0.2 --fractionValid 0.2
    done
  done

  mv *.hdf5 models/
done


source deactivate
