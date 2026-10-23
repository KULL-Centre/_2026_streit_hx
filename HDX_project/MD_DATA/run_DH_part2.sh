#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=DH/

for DIR in {BPTI,Calbindin,CalcineurinB,CI2_wt_306K,CI2_I29A_I37H_293K,CI2_wt_298K,drkN}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	# find line in metadata file containing temperature and ionic strength
	line=$(grep "^$DIR " ../../EXP_DATA/metadata_for_DH.txt) 
	read name temp ionic pH exchange conc <<< "$line"

	cp $code_path/debye_huckel.py ./

	for traj in *.xtc; do echo $traj; python debye_huckel.py protein.pdb topology.pdb $traj $temp $ionic $save_dir; done

	cd ..
	echo "Done!"

done

conda deactivate


