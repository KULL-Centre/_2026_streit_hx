#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=DH/

for DIR in {AAQAA3,}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	# find line in metadata file containing temperature and ionic strength
	line=$(grep "^$DIR " ../../EXP_DATA/metadata_for_DH.txt) 
	read name temp ionic pH exchange conc <<< "$line"
	
	cp $code_path/debye_huckel_weights.py ./

	for i in {0..2}; do echo traj$i.xtc; python debye_huckel_weights.py protein.pdb traj$i.xtc $temp $ionic $save_dir weights$i.txt True; done

	cd ..
	echo "Done!"

done

conda deactivate


