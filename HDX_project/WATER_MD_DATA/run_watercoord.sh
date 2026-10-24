#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=AmideHcoord/

for DIR in {AAQAA3,plastocyanin}; do

	echo $DIR
	cd $DIR/water_trajs_processed
	mkdir $save_dir

	cp $code_path/amide_water_coord.py ./

	for traj in *.xtc; do echo $traj; python amide_water_coord.py box.pdb $traj $save_dir True 0.26; done
	rm amide_water_coord.py

	cd ../../
	echo "Done!"

done

conda deactivate


