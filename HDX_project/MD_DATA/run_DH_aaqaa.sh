#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=DH/

for DIR in {AAQAA3,}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/debye_huckel_weights.py ./

	for i in {0..2}; do echo traj$i.xtc; python debye_huckel_weights.py protein.pdb traj$i.xtc 274.0 0.01 $save_dir weights$i.txt; done

	cd ..
	echo "Done!"

done

conda deactivate


