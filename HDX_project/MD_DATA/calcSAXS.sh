#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

cp $code_path/calc_SAXS.py ./

python calc_SAXS.py 

rm calc_SAXS.py

conda deactivate

