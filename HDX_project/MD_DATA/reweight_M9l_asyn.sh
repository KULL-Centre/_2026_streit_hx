#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/Rocklin_NMRHDX/code/'

save_dir=rewM9l/

for DIR in {aSyn,}; do
	DIR="${DIR%/}"   # Remove trailing slash
	echo $DIR
	cd $DIR
	mkdir $save_dir

	# find line in metadata file containing temperature and ionic strength
	line=$(grep "^$DIR " ../../EXP_DATA/metadata_for_DH.txt) 
	read name temp ionic pH exchange conc <<< "$line"

	cp $code_path/reweighting_M9l.py ./

	python reweighting_M9l.py --pdb protein.pdb --name $DIR --save_dir $save_dir --min_neff 0.1 --max_frames 20000

	rm reweighting_M9l.py

	cd ..
	echo "Done!"

done

conda deactivate


