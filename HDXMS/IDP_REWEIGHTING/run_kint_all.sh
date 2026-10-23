#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject


for DIR in {Angiotensin,Bradykinin,aSyn,Ab42,Tau}; do
	
	echo $DIR
	cd $DIR

	mkdir kint 
	cd kint

	# use pD 7.38 for Angiotensin and Bradykinin, pD 7.3732 for aSyn, Ab42, Tau
	if [[ "$DIR" == "Angiotensin" || "$DIR" == "Bradykinin" ]]; then
		pD=7.38
	else
		pD=7.3732
	fi

	python ../../kint_baseline.py --protein $DIR --ref_rates Stofella2025 --pD $pD --temperature 293.15
	python ../../kint_baseline.py --protein $DIR --ref_rates Nguyen2018 --pD $pD --temperature 293.15

	cd ../../



done

conda deactivate


