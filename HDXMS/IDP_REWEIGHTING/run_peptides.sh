#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject


for DIR in {Angiotensin,Bradykinin}; do
	
	echo $DIR

	cd $DIR

	for MOD in {M9,M9DH,M9DH_scaled,M10,M10DH,M10DH_scaled,M11,M11DH,M11DH_scaled,M12,M12DH,M12DH_scaled,M14,M14DH,M14DH_scaled}; do

		mkdir -p rew$MOD
		cd rew$MOD

		python ../../reweighting_peptides.py --protein $DIR --forward_model $MOD --ref_rates Stofella2025

		cd ..

	done

	cd ..

done

conda deactivate


