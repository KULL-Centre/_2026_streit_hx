#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=PCA/

for DIR in {Ubiquitin,}; do
	DIR="${DIR%/}"   # Remove trailing slash

	echo $DIR
	cd $DIR
	mkdir $save_dir

	python $code_path/run_PCA.py -t protein.pdb -o $save_dir
	
	cd ..
	echo "Done!"

done

conda deactivate


