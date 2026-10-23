#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

python $code_path/optimise_M9_params.py proteins.txt

echo "Forward models optimised!"

conda deactivate

