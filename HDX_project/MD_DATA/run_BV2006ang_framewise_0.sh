#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=BV2006ang/

for DIR in {Im7,}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/calc_PF_BV2006ang.py ./

	for traj in *.xtc; do echo $traj; python calc_PF_BV2006ang.py --pdb protein.pdb --traj $traj --beta_c 0.35 --beta_h 2.0 --save_dir $save_dir --save_frames True --stride 1 ; done

	rm calc_PF_BV2006ang.py
	cd ..
	echo "Done!"

done

conda deactivate


