#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

python $code_path/run_stats_rewM12DH_scaled.py proteins.txt PF_Nguyen_2018

echo "Calculated statistics for forward models!"

conda deactivate

