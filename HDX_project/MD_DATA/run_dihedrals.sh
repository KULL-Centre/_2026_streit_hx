#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=dihedrals/

for DIR in {AAQAA3,plastocyanin,ACTR,aSyn,BecN,ChiZ,d131d,drkN,NTail}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/compute_phi_psi.py ./

	for traj in *.xtc; do echo $traj; name=$(basename "$traj" .xtc); python compute_phi_psi.py --pdb protein.pdb --traj $traj --out $name --save_dir $save_dir --stride 1 ; done

	rm compute_phi_psi.py

	cd ..
	echo "Done!"

done

conda deactivate


