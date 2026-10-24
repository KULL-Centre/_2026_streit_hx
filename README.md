# _2026_streit_hx

This repository contains the experimental datasets, custom code, and analysis scripts associated with our study of hydrogen exchange (HX) forward models and their use in refining protein conformational ensembles. It covers both residue-resolved HX-NMR protection factors and peptide-resolved hydrogen–deuterium exchange mass spectrometry (HDX-MS).

Streit J.O., Peterle D., Tesei G., Invernizzi M., Bottaro S., Schulze T.K., Kunz P., Tamiola K., Lindorff-Larsen K. (2026). *Integrating hydrogen-exchange experiments with conformational ensembles*. **bioRxiv**. Link TBA.

## Repository structure

- **[`HDX_project/`](HDX_project/readme.md)** – HX-NMR forward-model benchmarking, parameter optimisation, and ensemble reweighting using experimental protection factors. Includes simulation datasets, experimental data, and analysis scripts.
- **[`HDXMS/`](HDXMS/readme.md)** – HDX-MS forward-model benchmarking and ensemble reweighting, including analyses of folded and intrinsically disordered proteins, comparisons of alternative ensembles, and cross-validation against independent measurements.
- **`scripts_for_NMR_SAXS/`** – Scripts for analyses involving NMR (CS, RDC, PRE) and small-angle X-ray scattering (SAXS) data.
- **`HDXproject_env.yml`** – Conda environment file.


## Getting started

### Setting up the Conda environment

The Python dependencies required for the analyses are specified in `HDXproject_env.yml`. 

From the root directory of this repository, create and activate the environment:

```bash
conda env create -f HDXproject_env.yml -n HDXproject
conda activate HDXproject
```

### Running the analyses

See the READMEs in [`HDX_project/`](HDX_project/readme.md) and [`HDXMS/`](HDXMS/readme.md) for their respective directory structures, input data, and analysis workflows.


## Data availability

- **Intrinsically disordered proteins:** Molecular dynamics ensembles are available on [Zenodo](https://zenodo.org/records/23569858).
- **Folded proteins:** Molecular dynamics data are available in two Zenodo records: [23569607](https://zenodo.org/records/23569607) and [23568205](https://zenodo.org/records/23568205).
- **Peptide simulations:** Trajectories including water coordinates are available on [Zenodo](https://zenodo.org/records/23539119).
- **BioEmu ensembles:** All BioEmu-generated ensembles are available on [Zenodo](https://zenodo.org/records/23564156).

**DESRES trajectories:** Previously reported molecular dynamics trajectories used in this study are available for non-commercial use by contacting D. E. Shaw Research directly at [Trajectories@DEShawResearch.com](mailto:Trajectories@DEShawResearch.com).

