#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

python CS_cross_validation_parallel_safe.py --n-workers 8

conda deactivate


