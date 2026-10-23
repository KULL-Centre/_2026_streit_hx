#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

save_dir=HDXrew_M14DH_scaled/

for DIR in {aSyn,}; do
	DIR="${DIR%/}"   # Remove trailing slash
	echo $DIR
	mkdir $DIR/$save_dir

	python run_rew_aSyn.py --protein_dir $DIR --save_dir $save_dir

	echo "Done!"

done

conda deactivate


