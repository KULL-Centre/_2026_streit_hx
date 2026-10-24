# HX-NMR Forward Model Analysis and Ensemble Reweighting

This repository contains simulation data, experimental datasets, and analysis scripts used to benchmark HX-NMR forward models and refine protein conformational ensembles using experimental protection factors (PFs).

## Workflow

The analysis notebooks in in `analysis/` require precomputed HX observables and data associated with the ensembles. Run the relevant scripts in `MD_DATA/`, `BIOEMU_DATA/`, `DESRES_FOLDING_DATA/`, and `DESRES_NATIVE_DATA/` before running the top-level analysis scripts. 

1. **Prepare intrinsic exchange rates and experimental PFs.** Run the relevant `kint` and PF scripts, including `update_PFs_Nguyen_2018.sh` or `update_PFs_Stofella_2025.sh` as appropriate. Use `update_seqs_for_kint.sh` to prepare sequences first.
2. **Optimise and assess forward models.** Run the relevant `optimise_*.sh` scripts to optimise forward model parameters (BV2006, BV2006ang, M8, M9) on unreweighted ensembles, and followed by `run_stats_*.sh` scripts to evaluate forward models before and after ensemble reweighting.

## Directory structure

### Simulation and ensemble data

- `MD_DATA/` – MD trajectories of ground-state ensemble and scripts for processing.
- `BIOEMU_DATA/` – BioEmu-generated conformational ensembles and associated scripts for analyses of folding landscapes.
- `DESRES_NATIVE_DATA/` – DESRES simulations of native-state ensembles with scripts.
- `DESRES_FOLDING_DATA/` – DESRES simulations sampling folding landscapes with scripts.
- `PEPTIDE_MD/`, `WATER_MD_DATA/` – Additional simulation datasets of peptides in explicit solvent (trippeptides, AAQAA3, plastocyanin). These also contain precomputed data required for analyses. 
- `INTEGRATIVE_IDPs/`, `INTEGRATIVE_IDPs_PRE/` – Integrative IDP ensembles reweighted with NMR and SAXS data used to evaluate forward models.

### Experimental data

- `EXP_DATA/` – Experimental HX-NMR data used for model benchmarking and ensemble reweighting.
- `ORTHOGONAL_EXP_DATA/` – Independent experimental measurements used for cross-validation.

### Code, analysis, and outputs

- `code/` – Python code for forward models, reweighting, and analysis.
- `analysis/` – Analysis notebooks and results.
- `output_results/` – Collected results and summary outputs of forward model benchmark on ground-state ensembles.

## Top-level files

- `optimise_*.sh` – Optimisation of forward-model parameters.
- `run_stats_*.sh` – Model assessment and statistics, including reweighted-ensemble analyses and variants with or without DH corrections.
- `update_PFs_Nguyen_2018.sh`, `update_PFs_Stofella_2025.sh` – Preparation or updating of PF datasets.
- `update_seqs_for_kint.sh` – Sequence preparation for intrinsic exchange-rate calculations.
- `update_summary_metadata.sh` – Updates summary metadata in `EXP_DATA/`.
- `proteins.txt`, `proteins_PF.txt` – Protein lists used by the analyses.
