#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

cp $code_path/run_stats_rewM9DH.py ./

python run_stats_rewM9DH.py proteins.txt PF_Nguyen_2018

rm run_stats_rewM9DH.py

echo "Calculated statistics for forward models!"

conda deactivate

