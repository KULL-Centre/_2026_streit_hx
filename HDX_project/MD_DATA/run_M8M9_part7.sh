#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=M8M9/

for DIR in {T4_lysozyme,GCN4}; do
	
	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/calc_PF_M8_M9.py ./

	for traj in *.xtc; do echo $traj; python calc_PF_M8_M9.py protein.pdb $traj $save_dir; done

	cd ..
	echo "Done!"

done

conda deactivate



