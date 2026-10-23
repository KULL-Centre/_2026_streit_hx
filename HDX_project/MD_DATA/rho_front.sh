#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=rho_front/

for DIR in {aSyn,plastocyanin,AAQAA3,Ubiquitin,}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	for traj in *.xtc; do echo $traj; python $code_path/compute_rho_front.py --pdb protein.pdb --traj $traj --save_dir $save_dir; done

	cd ..
	echo "Done!"

done

conda deactivate


