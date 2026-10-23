#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

for DIR in */; do
	DIR="${DIR%/}"   # Remove trailing slash
	echo $DIR

	python $code_path/get_rewPF_elbow.py --name $DIR

	echo "Done!"

done

conda deactivate


