#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject


cd aSyn

for MOD in {M14,M14DH,M14DH_scaled}; do

	mkdir bioemu_$MOD
	cd bioemu_$MOD

	python ../../ensemble_comparison.py --protein aSyn --forward_model $MOD --ref_rates Stofella2025 --pD 7.3732 --temperature 293.15 --ensemble BIOEMU
	python ../../ensemble_comparison.py --protein aSyn --forward_model $MOD --ref_rates Nguyen2018 --pD 7.3732 --temperature 293.15 --ensemble BIOEMU

	cd ..

	mkdir desres_$MOD
	cd desres_$MOD

	python ../../ensemble_comparison.py --protein aSyn --forward_model $MOD --ref_rates Stofella2025 --pD 7.3732 --temperature 293.15 --ensemble DESRES
	python ../../ensemble_comparison.py --protein aSyn --forward_model $MOD --ref_rates Nguyen2018 --pD 7.3732 --temperature 293.15 --ensemble DESRES

	cd ..

done

cd ..



conda deactivate


