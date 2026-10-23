#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDXMS/code/'

save_dir=M10DH/

for DIR in {Ab42,Tau}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	# Set ionic strength depending on system
	if [[ "$DIR" == "aSyn" || "$DIR" == "Ab42" || "$DIR" == "Tau" ]]; then
		ionic=0.17
	else
		ionic=0.02
	fi

	python $code_path/calc_M10DH.py --pdb protein.pdb --topology protein.pdb --name $DIR --save_dir $save_dir --temperature 293.0 --ionic $ionic --max_frames 3000000

	cd ..
	echo "Done!"

done

conda deactivate


