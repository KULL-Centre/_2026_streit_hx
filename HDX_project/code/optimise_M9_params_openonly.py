import numpy as np
import sys
import os
from os import listdir
import pandas as pd
import mdtraj as mdj
import scipy.stats
import re

sys.path.insert(1, '/projects/lindorff_group/people/thd711/HDX_project/code/')


def RMSE(actual, predicted):

    # Convert inputs to numpy arrays
    actual = np.array(actual)
    predicted = np.array(predicted)

    # Calculate the squared differences
    squared_diff = (actual - predicted) ** 2

    # Compute the mean of squared differences
    mean_squared_diff = np.mean(squared_diff)

    # Take the square root of the mean squared difference
    return np.sqrt(mean_squared_diff)


def find_xtc_filenames(path_to_dir, suffix=".xtc"):
    filenames = listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


# Read list of proteins to calculate PFs from a text file
input_proteins_file = sys.argv[1]

with open(input_proteins_file, 'r') as f:
    lines = f.readlines()

proteins = []

for line in lines:
    proteins.append(line.strip())


# SASA cut-off values to check
sasa_thresholds = [1.0, 2.5, 4.0, 5.5, 7.0]


for sasa_threshold in sasa_thresholds:

    print('SASA threshold: ', sasa_threshold)

    # Setup lists for final dataframe with optimal parameters
    PROTEIN = []
    MODEL_RESIDUE = []
    PF_EXP = []
    PF_EXP_ERR = []
    PF_SIM = []

    # Match calculated and experimental values/residues
    for protein in proteins:

        print(protein)

        # Experimental data
        exp = pd.read_csv(
            'EXP_DATA/PF_Nguyen_2018/{}.csv'.format(protein),
            index_col=0
        )

        model_residues = np.array(exp['Model_residue']).astype(int)

        # Model PDB
        u = mdj.load(
            'MD_DATA/{}/protein.pdb'.format(protein),
            top='MD_DATA/{}/protein.pdb'.format(protein)
        )

        amides = u.top.select(
            'name CA and not '
            '(resname PRO or resname NH2 or resname ACE or resname NME)'
        )

        amides_resnames = np.array([
            u.top.atom(i).residue.name for i in amides
        ])

        amides_resids = np.array([
            u.top.atom(i).residue.resSeq for i in amides
        ])

        # Load amide residue indices
        amide_res_idx = np.loadtxt(
            'MD_DATA/{}/M8M9/amideresidx.txt'.format(protein)
        )

        assert amide_res_idx.shape[0] == amides_resids.shape[0]

        exp_idxs = []

        for res in model_residues:

            hits = np.where(amides_resids == res)[0]

            if len(hits) == 1:
                exp_idxs.append(hits[0])

            elif len(hits) > 1:
                exp_idxs.append([hit for hit in hits])

            elif len(hits) == 0:
                print(
                    'Warning: there is no match for residue {} '
                    'in the pdb file of {}'.format(res, protein)
                )

        exp_idxs = np.array(exp_idxs).astype(int)

        # Treat monomer and dimer differently
        try:
            n_prot = exp_idxs.shape[1]
        except IndexError:
            n_prot = 1

        # Check how many simulations/files there are
        trajs = find_xtc_filenames(
            'MD_DATA/{}/'.format(protein)
        )

        n_trajs = len(trajs)

        # Load and average data
        tmp_nopen = []
        tmp_nclosed = []

        for traj in trajs:

            tmp_nopen.append(
                np.loadtxt(
                    'MD_DATA/{}/M8M9/'
                    'n_open_sasa_{}Ang_{}.txt'.format(
                        protein,
                        sasa_threshold,
                        traj[:-4]
                    )
                )
            )

            tmp_nclosed.append(
                np.loadtxt(
                    'MD_DATA/{}/M8M9/'
                    'n_closed_sasa_{}Ang_{}.txt'.format(
                        protein,
                        sasa_threshold,
                        traj[:-4]
                    )
                )
            )

        # sum open/closed state counts
        tmp_nopen = np.array(tmp_nopen)
        tmp_nclosed = np.array(tmp_nclosed)

        nopen = np.sum(tmp_nopen, axis=0)
        nclosed = np.sum(tmp_nclosed, axis=0)

        # PF = 1 / fraction_open
        # Residues with zero observed open frames are assigned NaN
        PF = np.full(nopen.shape, np.nan, dtype=float)

        has_open_frame = nopen >= 1

        PF[has_open_frame] = (
            1.0
            + nclosed[has_open_frame] / nopen[has_open_frame]
        )

        # Monomer vs dimer
        if n_prot == 1:  # Monomer

            PF = PF[exp_idxs]

        elif n_prot == 2:  # Dimer: average the chains

            exp_idxs_1 = exp_idxs[:, 0]
            exp_idxs_2 = exp_idxs[:, 1]

            assert exp_idxs_1.shape[0] == exp_idxs_2.shape[0]

            PF1 = PF[exp_idxs_1]
            PF2 = PF[exp_idxs_2]

            # Average the chains.
            # If either chain has no open frames, the resulting PF is NaN.
            PF = PF1 * 0.5 + PF2 * 0.5

        else:

            print(
                'Error: more than two protein copies '
                'are not yet supported'
            )
            break

        # Append all data to lists
        for i in range(len(model_residues)):

            PROTEIN.append(protein)
            MODEL_RESIDUE.append(model_residues[i])
            PF_EXP.append(float(exp['PF'].iloc[i]))
            PF_EXP_ERR.append(float(exp['PF_error'].iloc[i]))
            PF_SIM.append(PF[i])

    # Convert to arrays
    PROTEIN = np.array(PROTEIN)
    MODEL_RESIDUE = np.array(MODEL_RESIDUE)
    PF_EXP = np.array(PF_EXP)
    PF_EXP_ERR = np.array(PF_EXP_ERR)
    PF_SIM = np.array(PF_SIM)

    # Only include valid simulated PFs in the statistics
    valid = (
        np.isfinite(PF_EXP)
        & np.isfinite(PF_SIM)
        & (PF_EXP > 0)
        & (PF_SIM > 0)
    )

    # Calculate statistics in log space
    x = np.log10(PF_EXP[valid])
    y = np.log10(PF_SIM[valid])

    rmse = RMSE(x, y)
    pearson_r = scipy.stats.pearsonr(x, y)[0]

    # Make final dataframe to save.
    # Residues with no open frames remain present with SIM PF = NaN.
    df = pd.DataFrame()

    df['Protein'] = PROTEIN
    df['Model_residue'] = MODEL_RESIDUE
    df['EXP PF'] = PF_EXP
    df['EXP PF ERR'] = PF_EXP_ERR
    df['SIM PF'] = PF_SIM

    df.to_csv(
        'output_results/'
        'data_EXP_vs_SIM_M9_sasa_{}Ang.csv'.format(
            sasa_threshold
        )
    )

    print('M9 performance:')
    print('SASA threshold: {}'.format(sasa_threshold))
    print(
        'Valid datapoints: {} / {}'.format(
            np.sum(valid),
            len(valid)
        )
    )
    print('RMSE (log10 PF): {:.2f}'.format(rmse))
    print('Pearson r: {:.2f}'.format(pearson_r))