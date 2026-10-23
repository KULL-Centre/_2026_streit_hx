#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=S2blocks/

for DIR in {alpha3D,Barnase,betalactamasePSE4,CI2,FKBP12,GB3,HP36,Im7,Jarastatin,Lysozyme,PeroxiredoxinQ,Ubiquitin,HIVintegrase}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	# find line in metadata file containing temperature and ionic strength
	line=$(grep "^$DIR " ../../EXP_DATA/metadata_for_DH.txt) 
	read name temp ionic pH exchange conc <<< "$line"

	python $code_path/calc_S2blocks.py --pdb protein.pdb --name $DIR --save_dir $save_dir --max_frames 20001

	cd ..
	echo "Done!"

done

conda deactivate


