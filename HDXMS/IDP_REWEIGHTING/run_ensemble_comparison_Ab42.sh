#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject


cd Ab42

for MOD in {M14,M14DH,M14DH_scaled}; do

	mkdir hyres_gpu_$MOD
	cd hyres_gpu_$MOD

	python ../../ensemble_comparison.py --protein Ab42 --forward_model $MOD --ref_rates Stofella2025 --pD 7.3732 --temperature 293.15 --ensemble HYRES_GPU
	python ../../ensemble_comparison.py --protein Ab42 --forward_model $MOD --ref_rates Nguyen2018 --pD 7.3732 --temperature 293.15 --ensemble HYRES_GPU

	cd ..

	mkdir amber_$MOD
	cd amber_$MOD

	python ../../ensemble_comparison.py --protein Ab42 --forward_model $MOD --ref_rates Stofella2025 --pD 7.3732 --temperature 293.15 --ensemble AMBER
	python ../../ensemble_comparison.py --protein Ab42 --forward_model $MOD --ref_rates Nguyen2018 --pD 7.3732 --temperature 293.15 --ensemble AMBER

	cd ..

done

cd ..



conda deactivate


