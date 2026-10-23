#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=BV2006ang/

for DIR in {AAQAA3,}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/calc_PF_BV2006ang_weights.py ./

	for i in {0..2}; do echo traj$i.xtc; python calc_PF_BV2006ang_weights.py protein.pdb traj$i.xtc 0.35 2.0 $save_dir weights$i.txt; done

	cd ..
	echo "Done!"

done

conda deactivate


