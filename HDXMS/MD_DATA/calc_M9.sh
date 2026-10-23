#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDXMS/code/'

save_dir=M9/

for DIR in {Angiotensin,Bradykinin,aSyn,Ab42,Tau}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	python $code_path/calc_M9.py --pdb protein.pdb --name $DIR --save_dir $save_dir --max_frames 3000000

	cd ..
	echo "Done!"

done

conda deactivate


