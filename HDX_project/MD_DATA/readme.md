
Trajectory data archives: MD_DATA_IDPs.tar, MD_DATA_dimers.tar, MD_DATA_large.tar, MD_DATA_small.tar
Available here:
https://zenodo.org/records/23569858
https://zenodo.org/records/23569607
https://zenodo.org/uploads/23568205

The following will be missing:
ACTR/
drkN/
GCN4/
NTail/
Calmodulin/

These are available from D.E Shaw Research: Trajectories@DEShawResearch.com 

1. Scripts starting with 'run' calculate forward model descriptors/values (rho_front needs to be run before M11, M12 and M14 can be calculated). This is needed to run the benchmark and BV2006 optimisation on the dataset without reweighting.
2. Scripts starting with 'reweight' calculate forward model data (per-frame) and reweighting calculations. M10 needs to be run before M11 and M12 can be run. M13 needs to be run before M14 can be run. The rho_front scripts use compute_rho_front_noexcl.py (needed for M12 and M14). After, switch the called python script to compute_rho_front_neigh.py and store the results in rho_front_neigh4/ which is needed for model M11.
3. All other scripts calculate associated orthogonal data.
