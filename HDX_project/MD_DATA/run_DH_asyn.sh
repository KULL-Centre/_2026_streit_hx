#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=DH/

for DIR in {aSyn,BecN,ChiZ,d131d}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	# find line in metadata file containing temperature and ionic strength
	line=$(grep "^$DIR " ../../EXP_DATA/metadata_for_DH.txt) 
	read name temp ionic pH exchange conc <<< "$line"

	cp $code_path/debye_huckel.py ./

	for traj in *.xtc; do echo $traj; python debye_huckel.py --pdb protein.pdb --topfile topology.pdb --traj $traj --temperature $temp --ionic $ionic --save_dir $save_dir; done

	rm debye_huckel.py
	cd ..
	echo "Done!"

done

conda deactivate


