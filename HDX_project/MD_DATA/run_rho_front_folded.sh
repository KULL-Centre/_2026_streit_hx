#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=rho_front_noexcl/

for DIR in {ACBP,ACP,alpha3D,At2g44940,Barnase,BarstarK60,betalactamasePSE4,BPTI,Calbindin,CalcineurinB,ci2_I29A_I37H,CI2,DAXX,FKBP12,fynSH3,GB3,GCN4,H14,hFGF1,HP36,Im7,Jarastatin,Galectin1,Hisactophilin1,HIVintegrase,Lactalbumin,lDHFR,Lysozyme,NHP2L1,PeroxiredoxinQ,RNaseA,RNaseH,RNaseT1,SAMP1,Staph_nuclease_P117G_H124L,T4_lysozyme,Ubiquitin}; do
	echo $DIR
	cd $DIR
	mkdir $save_dir

	# find line in metadata file containing temperature and ionic strength
	line=$(grep "^$DIR " ../../EXP_DATA/metadata_for_DH.txt) 
	read name temp ionic pH exchange conc <<< "$line"

	python $code_path/compute_rho_front_noexcl.py --pdb protein.pdb --name $DIR --save_dir $save_dir

	cd ..
	echo "Done!"

done

conda deactivate


