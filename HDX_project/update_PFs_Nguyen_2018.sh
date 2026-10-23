#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

cp $code_path/PF_Nguyen_2018.py ./

python PF_Nguyen_2018.py proteins_PF.txt

rm PF_Nguyen_2018.py

echo "Protection factors (Nguyen 2018 parameters) calculated and updated!"

conda deactivate

