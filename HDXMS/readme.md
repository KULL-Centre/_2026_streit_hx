# HDX-MS Forward Model Analysis and Ensemble Reweighting

This repository contains scripts and code to process the simulation data and experimental datasets to analyse forward models and ensemble reweighting with HDX-MS data.

## Directory structure

### Simulation data
- `MD_DATA/` – Scripts for MD trajecories.
- `BIOEMU_DATA/` – Scripts for BIOEMU ensembles.
- `AMBER_IDPs/`, `DESRES_IDPs/`, `HYRES_GPU_IDPs/`, `IDPO_IDPs/`, `BIOEMU_IDPs/` – Intrinsically disordered protein (IDP) ensembles generated using different simulation methods and models.

### Experimental data
- `EXP_DATA/` – Experimental HDX-MS uptake data used for forward-model benchmarking and ensemble reweighting.
- `ORTHOGONAL_EXP_DATA/` – Independent experimental measurements used for cross-validation.
- `CD_DATA/` – Cluster data (HDX-MS of aSyn, Ab42 and Tau).

### Code and analysis
- `code/` – Scripts for HX forward-model calculations, ensemble reweighting, and analysis.
- `analysis_IDPs/` – HDX uptake analyses of intrinsically disordered proteins.
- `analysis_EEHEE/` – Analyses of the EEHEE protein.
- `IDP_REWEIGHTING/` – Reweighting and benchmarking of IDP ensembles with HDX-MS data.
- `INTEGRATIVE_IDPs/` – NMR/SAXS-reweighted IDP ensembles of aSyn, Ab42, Tau used for benchmarking HDX predictions.
- `COMPARING_AB42_ENSEMBLES/`, `COMPARING_ASYN_ENSEMBLES/`, `COMPARING_TAU_ENSEMBLES/` – Comparisons of alternative conformational ensembles for Ab42, aSyn and Tau with HDX-MS data and their conformational properties.

