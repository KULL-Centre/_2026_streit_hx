#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

cp $code_path/PF_Stofella_2025.py ./

python PF_Stofella_2025.py proteins_PF.txt

rm PF_Stofella_2025.py

echo "Protection factors (Nguyen 2018 parameters) calculated and updated!"

conda deactivate

