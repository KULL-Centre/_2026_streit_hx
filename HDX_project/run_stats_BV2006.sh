#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

cp $code_path/run_stats_BV2006.py ./

python run_stats_BV2006.py proteins.txt PF_Nguyen_2018

rm run_stats_BV2006.py

echo "Calculated statistics for forward models!"

conda deactivate

