#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

cp $code_path/optimise_BV2006_params.py ./

python optimise_BV2006_params.py proteins.txt PF_Nguyen_2018

rm optimise_BV2006_params.py

echo "Forward models optimised!"

conda deactivate

