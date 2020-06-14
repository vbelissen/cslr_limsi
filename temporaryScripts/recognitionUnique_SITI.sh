#!/bin/bash

OUTPUTS=("fls" "DS" "PT" "FBUOY")

CONFIG_SI=(0 1 0 1)
CONFIG_TI=(0 0 1 1)

COMMENT=("variation_config_SD_TD" "variation_config_SI_TD" "variation_config_SD_TI" "variation_config_SI_TI")

FIXED_SIGNERS_TRAIN_SI="0 7 8 9 10 11 12 13 14 15"
FIXED_SIGNERS_VALID_SI="1 2 3"
FIXED_SIGNERS_TEST_SI="4 5 6"

FIXED_TASKS_TRAIN_TI="4 5 6"
FIXED_TASKS_VALID_TI="1 2 3"
FIXED_TASKS_TEST_TI="7 8"

iterSDTD=4

lenOut=${#OUTPUTS[@]}
lenConfigs=${#CONFIG_SI[@]}



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
    comment=${COMMENT[$iConfig]}
    if [ $config_SI = 0 ] && [ $config_TI = 0 ]
    then
      for ((i=0; i<$iterSDTD; i++));
      do
        python src/recognitionUniqueDictaSignFromScript.py --outputName $output --comment $comment --epochs 150 --videoSplitMode auto --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST --fractionValid 0.2 --fractionTest 0.2
      done
    elif [ $config_SI = 1 ] && [ $config_TI = 0 ]
    then
      python src/recognitionUniqueDictaSignFromScript.py --outputName $output --comment $comment --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN_SI --signersValid $FIXED_SIGNERS_VALID_SI --signersTest $FIXED_SIGNERS_TEST_SI --excludeTask9 1
    elif [ $config_SI = 0 ] && [ $config_TI = 1 ]
    then
      python src/recognitionUniqueDictaSignFromScript.py --outputName $output --comment $comment --epochs 150 --videoSplitMode manual --tasksTrain $FIXED_TASKS_TRAIN_TI --tasksValid $FIXED_TASKS_VALID_TI --tasksTest $FIXED_TASKS_TEST_TI
    else
      python src/recognitionUniqueDictaSignFromScript.py --outputName $output --comment $comment --epochs 150 --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN_SI --signersValid $FIXED_SIGNERS_VALID_SI --signersTest $FIXED_SIGNERS_TEST_SI --tasksTrain $FIXED_TASKS_TRAIN_TI --tasksValid $FIXED_TASKS_VALID_TI --tasksTest $FIXED_TASKS_TEST_TI
    fi
  done

  mv *.hdf5 models/
done


source deactivate
