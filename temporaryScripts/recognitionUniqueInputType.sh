#!/bin/bash

OUTPUTS=("fls" "FS" "DS" "PT" "FBUOY")
INPUT_TYPE=("2Draw_noHands" "2Dfeatures_noHands" "3Draw_noHands" "3Dfeatures_noHands")
#INPUT_TYPE=("2Draw" "2Draw_HS" "2Draw_HS_noOP" "2Dfeatures" "2Dfeatures_HS" "2Dfeatures_HS_noOP" "3Draw" "3Draw_HS" "3Draw_HS_noOP" "3Dfeatures" "3Dfeatures_HS" "3Dfeatures_HS_noOP")

FIXED_SIGNERS_TRAIN="0 5 6 7 8 9 10 11 12 13 14 15"
FIXED_SIGNERS_VALID="1 2"
FIXED_SIGNERS_TEST="3 4"

lenOut=${#OUTPUTS[@]}
lenIn=${#INPUT_TYPE[@]}


cd ..
source activate py36_tf1
#python src/recognitionUniqueDictaSignFromScript.py --epochs 30 --idxTrainBypass $({ echo {1..10}; echo {50..60}; } | tr "\n" " ") --idxValidBypass {10..20} --idxTestBypass {20..30}

#for (( iOutput=0; iOutput<${tLen}; iOutput++ ));
for (( iOutput=0; iOutput<${lenOut}; iOutput++ ));
do
  output=${OUTPUTS[$iOutput]}
  echo Output: $output#
  #output=${OUTPUTS[$iOutput]}

  for (( iI=0; iI<${lenIn}; iI++ ));
  do
    inputType=${INPUT_TYPE[$iI]}
    python src/recognitionUniqueDictaSignFromScript.py --outputName $output --comment "variation input" --epochs 150 --inputType $inputType --videoSplitMode manual --signersTrain $FIXED_SIGNERS_TRAIN --signersValid $FIXED_SIGNERS_VALID --signersTest $FIXED_SIGNERS_TEST
  done

  mv *.hdf5 models/
done


source deactivate
