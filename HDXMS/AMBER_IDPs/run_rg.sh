#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=rg_data/

for DIR in {Ab42,}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/calc_rg.py ./

	for traj in *.xtc; do echo $traj; python calc_rg.py --pdb protein.pdb --traj $traj --save_dir $save_dir --stride 1 ; done

	rm calc_rg.py

	cd ..
	echo "Done!"

done

conda deactivate


