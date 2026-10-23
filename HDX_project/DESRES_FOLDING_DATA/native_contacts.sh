#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=native_contacts/

for DIR in */; do 

	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/native_contacts.py ./

	for traj in *.xtc; do python native_contacts.py --pdb exp_native_renum.pdb --xtc $traj --lambda_param 1.8 --beta 50.0 --cutoff 0.45 --output $save_dir/Q_$traj.txt; done       
	
	rm native_contacts.py
	cd ..
	echo "Done!"

done

conda deactivate


