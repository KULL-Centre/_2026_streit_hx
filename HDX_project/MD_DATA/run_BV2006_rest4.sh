#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=BV2006ang/

for DIR in {HIVintegrase,Jarastatin,Lactalbumin,lDHFR,Lysozyme,NTail}; do
	
	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/calc_PF_BV2006ang.py ./

	for traj in *.xtc; do echo $traj; python calc_PF_BV2006ang.py protein.pdb $traj 0.35 2.0 $save_dir; done

	cd ..
	echo "Done!"

done

conda deactivate



