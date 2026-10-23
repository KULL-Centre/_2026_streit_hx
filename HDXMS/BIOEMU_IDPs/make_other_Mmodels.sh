#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDXMS/code/'



for DIR in {aSyn,}; do

	echo $DIR
	cd $DIR
	
	for model in {M14,M14DH,M14DH_scaled}; do 
		mkdir $model

		python $code_path/make_$model.py --pdb protein.pdb --name $DIR --save_dir $model/ 

	done
		
	cd ..
	echo "Done!"

done

conda deactivate


