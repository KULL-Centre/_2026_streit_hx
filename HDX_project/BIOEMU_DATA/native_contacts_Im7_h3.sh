#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=native_contacts_h3/

for DIR in {Im7,}; do
	DIR="${DIR%/}"   # Remove trailing slash

	echo $DIR
	cd $DIR
	mkdir $save_dir

	for i in {0..59}; do python $code_path/native_contacts_resfilter.py --pdb protein.pdb --xtc traj$i.xtc --lambda_param 1.8 --beta 50.0 --cutoff 0.45 --res_min 49 --res_max 56 --output $save_dir/Q$i.txt; done       

	cd ..
	echo "Done!"

done

conda deactivate


