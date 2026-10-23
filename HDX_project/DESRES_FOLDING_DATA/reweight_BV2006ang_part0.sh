#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=rewBV2006ang/

for DIR in */; do
	DIR="${DIR%/}"   # Remove trailing slash
	echo $DIR
	cd $DIR
	mkdir $save_dir

	# find line in metadata file containing temperature and ionic strength
	line=$(grep "^$DIR " ../../EXP_DATA/metadata_for_DH.txt) 
	read name temp ionic pH exchange conc <<< "$line"

	cp $code_path/reweighting_BV2006ang.py ./

	python reweighting_BV2006ang.py --pdb protein.pdb --name $DIR --beta_c 0.93 --beta_h 4.24 --save_dir $save_dir \
		--min_neff 0.1 --max_frames 20000

	rm reweighting_BV2006ang.py

	cd ..
	echo "Done!"

done

conda deactivate


