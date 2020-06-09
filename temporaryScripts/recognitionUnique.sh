#!/bin/bash

OUTPUTS=("fls" "FS" "DS" "PT" "FBUOY")


lenOutputs=${#OUTPUTS[@]}

# use for loop read all nameservers
for (( iOutput=0; iOutput<${tLen}; iOutput++ ));
do
  echo ${OUTPUTS[$iOutput]}
done

cd ..
source activate py36_tf1
#python src/recognitionUniqueDictaSignFromScript.py --epochs 30 --idxTrainBypass $({ echo {1..10}; echo {50..60}; } | tr "\n" " ") --idxValidBypass {10..20} --idxTestBypass {20..30}




source deactivate
