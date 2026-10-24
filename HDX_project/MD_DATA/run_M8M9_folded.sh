#!/bin/bash

module load miniconda/py39_23.1
source activate HDXproject

code_path='/projects/lindorff_group/people/thd711/HDX_project/code/'

save_dir=M8M9/

for DIR in {ACBP,ACP,alpha3D,At2g44940,Barnase,BarstarK60,betalactamasePSE4,BPTI,Calbindin,CalcineurinB,ci2_I29A_I37H,CI2,DAXX,FKBP12,fynSH3,GB3,GCN4,H14,hFGF1,HP36,Im7,Jarastatin,Galectin1,Hisactophilin1,HIVintegrase,Lactalbumin,lDHFR,Lysozyme,NHP2L1,PeroxiredoxinQ,RNaseA,RNaseH,RNaseT1,SAMP1,Staph_nuclease_P117G_H124L,T4_lysozyme,Ubiquitin}; do

	echo $DIR
	cd $DIR
	mkdir $save_dir

	cp $code_path/calc_PF_M8_M9.py ./

	for traj in *.xtc; do echo $traj; python calc_PF_M8_M9.py --pdb protein.pdb --traj $traj --save_dir $save_dir; done

	cd ..
	echo "Done!"

done

conda deactivate


