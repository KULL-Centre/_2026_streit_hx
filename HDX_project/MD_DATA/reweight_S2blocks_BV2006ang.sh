#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=S2blocks_rewBV2006ang/

for DIR in {GB3,}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/reweighting_S2blocks_BV2006ang.py ./

	python reweighting_S2blocks_BV2006ang.py --pdb protein.pdb --name $DIR --beta_c 0.93 --beta_h 4.24 --save_dir $save_dir --min_neff 0.1 --max_frames 20001

	cd ..
	echo "Done!"

done

conda deactivate


