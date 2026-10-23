#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=rewM9/

for DIR in {CI2_I29A_I37H_293K,CI2_wt_298K,CI2_wt_306K,DAXX,FKBP12,fynSH3,GB3,GCN4,H14,hFGF1,HP36,Jarastatin}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	# find line in metadata file containing temperature and ionic strength
	line=$(grep "^$DIR " ../../EXP_DATA/metadata_for_DH.txt) 
	read name temp ionic pH exchange conc <<< "$line"

	cp $code_path/reweighting_M9.py ./

	python reweighting_M9.py --pdb protein.pdb --name $DIR --save_dir $save_dir --min_neff 0.1 --max_frames 20001

	rm reweighting_M9.py

	cd ..
	echo "Done!"

done

conda deactivate


