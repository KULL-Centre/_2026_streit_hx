#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=SASA/

for DIR in {Ubiquitin,}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/amide_SASA.py ./

	for traj in *.xtc; do echo $traj; python amide_SASA.py protein.pdb $traj $save_dir; done

	cd ..
	echo "Done!"

done

conda deactivate


