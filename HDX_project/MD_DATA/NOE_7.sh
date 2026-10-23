#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=NOE/

for DIR in {Lysozyme,}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/calc_noes_from_table.py ./

	for traj in *.xtc; do echo $traj; name=$(basename "$traj" .xtc); python calc_noes_from_table.py protein.pdb $traj ../../ORTHOGONAL_EXP_DATA/$DIR/NOE.dat $save_dir/d_$name.npy ; done

	rm calc_noes_from_table.py

	cd ..
	echo "Done!"

done

conda deactivate


