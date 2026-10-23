#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=rewM12DH_scaled/

for DIR in */; do
	DIR="${DIR%/}"   # Remove trailing slash

	echo $DIR
	cd $DIR
	mkdir $save_dir

	# find line in metadata file containing temperature and ionic strength
	line=$(grep "^$DIR " ../../EXP_DATA/metadata_for_DH.txt) 
	read name temp ionic pH exchange conc <<< "$line"

	python $code_path/reweighting_M12DH_scale.py --pdb protein.pdb --topology topology.pdb --name $DIR --save_dir $save_dir --temperature $temp --ionic $ionic --min_neff 0.1 --max_frames 20001 --rew_method log_lambda --filter

	cd ..
	echo "Done!"

done

conda deactivate


