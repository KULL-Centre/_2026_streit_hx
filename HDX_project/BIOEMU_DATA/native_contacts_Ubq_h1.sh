#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=native_contacts_h1/

for DIR in {Ubiquitin,}; do
	DIR="${DIR%/}"   # Remove trailing slash

	echo $DIR
	cd $DIR
	mkdir $save_dir

	for i in {0..99}; do python $code_path/native_contacts_resfilter.py --pdb protein.pdb --xtc traj$i.xtc --lambda_param 1.8 --beta 50.0 --cutoff 0.45 --res_min 22 --res_max 35 --output $save_dir/Q$i.txt; done       

	cd ..
	echo "Done!"

done

conda deactivate


