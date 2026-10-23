#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=rho_front_noexcl/

for DIR in {Angiotensin,Bradykinin,aSyn,Ab42,Tau}; do 
	echo $DIR
	cd $DIR
	mkdir $save_dir

	python $code_path/compute_rho_front_noexcl.py --pdb protein.pdb --name $DIR --save_dir $save_dir

	cd ..
	echo "Done!"

done

conda deactivate


