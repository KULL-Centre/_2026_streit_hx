#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

python calc_SAXS.py 

conda deactivate

