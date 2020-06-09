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

lenConfigs=${#CONFIG_SI[@]}

cd ..
source activate py36_tf1
#python src/recognitionUniqueDictaSignFromScript.py --epochs 30 --idxTrainBypass $({ echo {1..10}; echo {50..60}; } | tr "\n" " ") --idxValidBypass {10..20} --idxTestBypass {20..30}

#for (( iOutput=0; iOutput<${tLen}; iOutput++ ));
for output in OUTPUTS
do
  echo Output: $output#${OUTPUTS[$iOutput]}
  #output=${OUTPUTS[$iOutput]}

  for batchSize in BATCH_SIZE:
  do
    python src/recognitionUniqueDictaSignFromScript.py --outputName output --comment "variation batch" --batchSize batchSize --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  done

  for dropout in DROPOUT:
  do
    python src/recognitionUniqueDictaSignFromScript.py --outputName output --comment "variation dropout" --dropout dropout --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  done

  for rnnNumber in RNN_NUMBER:
  do
    python src/recognitionUniqueDictaSignFromScript.py --outputName output --comment "variation rnn number" --rnnNumber rnnNumber --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  done

  for rnnHiddenUnits in HIDDEN_UNITS:
  do
    python src/recognitionUniqueDictaSignFromScript.py --outputName output --comment "variation rnn units" --rnnHiddenUnits rnnHiddenUnits --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  done

  for convFilt in CONV_FILTERS:
  do
    python src/recognitionUniqueDictaSignFromScript.py --outputName output --comment "variation conv filters" --convFilt convFilt --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  done

  for weightCorrection in WEIGHT_CORRECTION:
  do
    python src/recognitionUniqueDictaSignFromScript.py --outputName output --comment "variation weight correction" --weightCorrection weightCorrection --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  done

  for seqLength in SEQ_LENGTH:
  do
    python src/recognitionUniqueDictaSignFromScript.py --outputName output --comment "variation seq length" --seqLength seqLength --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  done

  for excludeTask9 in EXCLUDE_TASK_9:
  do
    python src/recognitionUniqueDictaSignFromScript.py --outputName output --comment "exclude or include task 9" --excludeTask9 excludeTask9 --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  done

  for (( iConfig=0; iConfig<${lenConfigs}; iConfig++ ));
  do
    config_SI=${CONFIG_SI[$iConfig]}
    config_TI=${CONFIG_TI[$iConfig]}
    for randSeed in RANDOM_SEEDS:
    do
      python src/recognitionUniqueDictaSignFromScript.py --outputName output --comment "variation config SI TI" --randSeed randSeed --epochs 150 --videoSplitMode auto --signerIndependent config_SI --taskIndependent config_TI
    done
  done

  mv *.hdf5 models/
done


source deactivate
