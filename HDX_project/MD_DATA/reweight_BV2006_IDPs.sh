#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=rewBV2006/

for DIR in {AAQAA3,plastocyanin,aSyn,ACTR,BecN,ChiZ,drkN,d131d,NTail,Calmodulin}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	python $code_path/reweighting_BV2006.py --pdb protein.pdb --name $DIR --save_dir $save_dir \
		--min_neff 0.1 --max_frames 20000001

	cd ..
	echo "Done!"

done

conda deactivate


