#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=native_contacts_h2/

for DIR in {ACBP,}; do
	DIR="${DIR%/}"   # Remove trailing slash

	echo $DIR
	cd $DIR
	mkdir $save_dir

	for i in {0..29}; do python $code_path/native_contacts_resfilter.py --pdb protein.pdb --xtc traj$i.xtc --lambda_param 1.8 --beta 50.0 --cutoff 0.45 --res_min 21 --res_max 36 --output $save_dir/Q$i.txt; done       

	cd ..
	echo "Done!"

done

conda deactivate


