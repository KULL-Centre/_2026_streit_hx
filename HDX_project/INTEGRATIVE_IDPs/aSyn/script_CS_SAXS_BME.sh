#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

python run_CS_SAXS_BMErew.py 


deactivate

