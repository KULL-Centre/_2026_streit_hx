#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=native_contacts_s4/

for DIR in {Ubiquitin,}; do
	DIR="${DIR%/}"   # Remove trailing slash

	echo $DIR
	cd $DIR
	mkdir $save_dir

	for traj in *.xtc; do python $code_path/native_contacts_resfilter.py --pdb protein.pdb --xtc $traj --lambda_param 1.8 --beta 50.0 --cutoff 0.45 --res_min 66 --res_max 71 --output $save_dir/Q_$traj.txt; done       

	cd ..
	echo "Done!"

done

conda deactivate


