import os

# Important: set these before importing numpy/scipy/pandas where possible.
# Prevents each worker process from spawning many BLAS/OpenMP threads.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # safe for parallel figure generation

import matplotlib.pyplot as plt
import matplotlib as mpl

import ast
import scipy
import mdtraj as md


# matplotlib settings
plt.rcParams.update({
    # Font
    "font.family": "Times New Roman",
    "font.serif": ["Arial"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "pdf.fonttype": 42,

    # Figure
    "figure.figsize": (6, 4),
    "figure.dpi": 150,

    # Axes
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.linewidth": 1.2,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.7,

    # Ticks
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,

    # Lines
    "lines.linewidth": 2,
    "lines.markersize": 6,

    # Legend
    "legend.fontsize": 10,
    "legend.frameon": False
})


def find_xtc_filenames(path_to_dir, suffix=".xtc"):
    filenames = os.listdir(path_to_dir)
    return [filename for filename in filenames if filename.endswith(suffix)]


# --- Utility functions ---
def RMSE(actual, predicted):
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.sqrt(np.mean((actual - predicted) ** 2))


def load_pf_data(protein, model_name, ensemble_dir="MD_DATA", exp_dir="EXP_DATA/PF_Nguyen_2018"):
    exp = pd.read_csv(f"../../{exp_dir}/{protein}.csv")
    residues = np.loadtxt(f"../../{ensemble_dir}/{protein}/{model_name}/exp_residues.txt")
    priorPF = np.exp(np.loadtxt(f"../../{ensemble_dir}/{protein}/{model_name}/prior_lnPF.txt"))

    # stats is currently unused, so this read is commented out.
    # stats = pd.read_csv(f"../../{ensemble_dir}/{protein}/{model_name}/stats_df.csv")
    stats = None

    if model_name in ("rewBV2006", "rewBV2006ang"):
        rewdata = np.loadtxt(f"../../{ensemble_dir}/{protein}/{model_name}/reweighting_chi2_phi_theta.txt")
        chi2, phi, thetas = rewdata[0, :], rewdata[1, :], rewdata[2, :]
        rewPF = np.exp(np.loadtxt(f"../../{ensemble_dir}/{protein}/{model_name}/rew_lnPF.txt"))
    else:
        rewdata = np.loadtxt(f"../../{ensemble_dir}/{protein}/{model_name}/log_lambda_reweighting_chi2_phi_theta.txt")
        chi2, phi, thetas = rewdata[0, :], rewdata[1, :], rewdata[2, :]
        rewPF = np.exp(np.loadtxt(f"../../{ensemble_dir}/{protein}/{model_name}/log_lambda_rew_lnPF.txt"))

    idxs = [list(exp["Model_residue"]).index(ele) for ele in residues]

    return exp, residues, priorPF, rewPF, chi2, phi, thetas, stats, idxs


def find_elbow(x, y):
    x = np.array(x)
    y = np.array(y)

    if not np.all(np.diff(x) >= 0):
        idx_sort = np.argsort(x)
        x_sorted = x[idx_sort]
        y_sorted = y[idx_sort]
    else:
        idx_sort = np.arange(len(x))
        x_sorted, y_sorted = x, y

    p1 = np.array([x_sorted[0], y_sorted[0]])
    p2 = np.array([x_sorted[-1], y_sorted[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    vec_from_p1 = np.vstack([x_sorted - p1[0], y_sorted - p1[1]]).T
    proj_len = np.dot(vec_from_p1, line_vec_norm)
    proj_point = np.outer(proj_len, line_vec_norm) + p1
    dist = np.linalg.norm(vec_from_p1 - (proj_point - p1), axis=1)

    idx_elbow_sorted = np.argmax(dist)
    idx_elbow_original = idx_sort[idx_elbow_sorted]

    return idx_elbow_original, x[idx_elbow_original], y[idx_elbow_original]


def helical_fraction(ss_line):
    helix_chars = {'H', 'G', 'I'}
    n_residues = len(ss_line)
    n_helical = sum(1 for c in ss_line if c in helix_chars)
    return n_helical / n_residues


def dssp_to_helicity(dssp_lines):
    """
    Convert DSSP strings into a binary helicity matrix.

    Parameters
    ----------
    dssp_lines : list of str
        Each entry corresponds to one frame.
        Each string is the DSSP assignment per residue for that frame.

    Returns
    -------
    helicity : np.ndarray
        Array of shape (n_frames, n_residues).
        1 = helix ('H', 'G', 'I'), 0 = non-helix.
    """
    helix_codes = {'H', 'G', 'I'}

    n_frames = len(dssp_lines)
    n_residues = len(dssp_lines[0])

    helicity = np.zeros((n_frames, n_residues), dtype=int)

    for i, line in enumerate(dssp_lines):
        helicity[i, :] = [1 if c in helix_codes else 0 for c in line]

    return helicity


# Constants
CS_UNCERTAINTIES = {
    "Sparta+": {"C": 1.09, "CA": 0.94, "CB": 1.14, "N": 2.45, "H": 0.49, "HA": 0.25}
}


AA_dict = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "E": "GLU",
    "Q": "GLN",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
}


ATOM_TYPES = ["CA"]


def load_experimental_shifts(path: str, ATOM_TYPES) -> Dict[Tuple[int, str], float]:
    df = pd.read_csv(path, delim_whitespace=True)
    residues = []
    names = []
    values = []

    for i in range(len(df)):
        res = int(df['#RESID'][i])

        for col in df.columns[2:]:
            if col in ATOM_TYPES:
                if col == 'HN':
                    try:
                        value = float(df[col][i])
                        if value > 0:
                            residues.append(res)
                            names.append('H')
                            values.append(value)
                    except Exception:
                        continue
                else:
                    try:
                        value = float(df[col][i])
                        if value > 0:
                            residues.append(res)
                            names.append(col)
                            values.append(value)
                    except Exception:
                        continue

    return {
        (int(res), str(name)): float(value)
        for res, name, value in zip(residues, names, values)
    }


def load_and_filter_experimental_shifts(
    path: str,
    bmrb_stats_path: str,
    sigma_cutoff=3.0,
    ATOM_TYPES=ATOM_TYPES
) -> Dict[Tuple[int, str], float]:

    df = pd.read_csv(path, delim_whitespace=True)
    bmrb = pd.read_csv(bmrb_stats_path)

    residues = []
    names = []
    values = []

    for i in range(len(df)):
        res = int(df['#RESID'][i])
        resname = str(df['RESNAME'][i])

        if len(resname) == 1:
            resname = AA_dict[resname]

        for col in df.columns[2:]:
            if col in ATOM_TYPES:
                try:
                    value = float(df[col][i])

                    if value > 0:
                        if col == 'HN':
                            row = bmrb[(bmrb["comp_id"] == resname) & (bmrb["atom_id"] == 'H')]
                            bmrb_mu = row["avg"].values[0]
                            bmrb_sd = row["std"].values[0]

                            if np.abs(value - bmrb_mu) < bmrb_sd * sigma_cutoff:
                                residues.append(res)
                                names.append('H')
                                values.append(value)

                        else:
                            row = bmrb[(bmrb["comp_id"] == resname) & (bmrb["atom_id"] == col)]
                            bmrb_mu = row["avg"].values[0]
                            bmrb_sd = row["std"].values[0]

                            if np.abs(value - bmrb_mu) < bmrb_sd * sigma_cutoff:
                                residues.append(res)
                                names.append(col)
                                values.append(value)

                except Exception:
                    continue

    return {
        (int(res), str(name)): float(value)
        for res, name, value in zip(residues, names, values)
    }


def load_potenci_shifts(path: str, ATOM_TYPES=ATOM_TYPES) -> Dict[Tuple[int, str], float]:
    df = pd.read_csv(path, delim_whitespace=True)

    residues = []
    names = []
    values = []

    for i in range(len(df)):
        res = int(df['#NUM'][i])

        for col in df.columns[2:]:
            if col in ATOM_TYPES:
                try:
                    value = float(df[col][i])

                    if value > 0:
                        residues.append(res)
                        names.append(col)
                        values.append(value)

                except Exception:
                    continue

    return {
        (int(res), str(name)): float(value)
        for res, name, value in zip(residues, names, values)
    }


def load_generated_shifts(path: str) -> pd.DataFrame:
    """
    Load generated shifts from a CSV file.
    Assumes columns: resSeq, name, 0, 1, 2, ...
    Returns DataFrame indexed by (resSeq, name), with frames as columns.
    """
    df = pd.read_csv(path)
    df["resSeq"] = df["resSeq"].astype(int)
    df["name"] = df["name"].astype(str)
    df.set_index(["resSeq", "name"], inplace=True)
    return df


def standardize_deltas(
    gen_df: pd.DataFrame,
    exp_dict: Dict[Tuple[int, str], float],
    predictor: str = "Sparta+"
) -> Tuple[np.ndarray, list]:

    cs_uncertainties = CS_UNCERTAINTIES[predictor]
    keys = sorted(set(exp_dict).intersection(set(gen_df.index)))

    std_devs = np.array([cs_uncertainties[a] for (_, a) in keys]).reshape(-1, 1)
    exp_vals = np.array([exp_dict[k] for k in keys]).reshape(-1, 1)
    gen_vals = gen_df.loc[keys].values

    std_delta = (gen_vals - exp_vals) / std_devs

    return std_delta, keys


def raw_diff(gen_df: pd.DataFrame, exp_dict: Dict[Tuple[int, str], float]):
    keys = sorted(set(exp_dict).intersection(set(gen_df.index)))
    exp_vals = np.array([exp_dict[k] for k in keys]).reshape(-1, 1)
    gen_vals = gen_df.loc[keys].values
    diff = gen_vals - exp_vals

    return diff, keys


def get_RMSE(
    std_delta_cs: np.ndarray,
    weights: np.ndarray = None,
    order: int = 2,
) -> float:
    """
    Root Mean Square Error.
    """
    assert len(std_delta_cs.shape) == 2

    mask_nan = ~np.isnan(std_delta_cs).any(axis=0)

    if weights is None:
        return np.linalg.norm(
            np.average(std_delta_cs[:, mask_nan], axis=1),
            ord=order
        ) / np.power(len(std_delta_cs), 1 / order)

    mask_nan_weights = ~np.isnan(weights)

    if weights[mask_nan_weights & (~mask_nan)].sum() != 0:
        print("+++ WARNING: samples with NaN chemical shifts should have 0 weight", flush=True)

    mask_nan = mask_nan & mask_nan_weights

    if not mask_nan.any():
        return np.nan

    return np.linalg.norm(
        np.average(std_delta_cs[:, mask_nan], weights=weights[mask_nan], axis=1),
        ord=order,
    ) / np.power(len(std_delta_cs), 1 / order)


def analyse_protein(protein):
    """
    Run the complete analysis for one protein.

    This function is called in parallel by ProcessPoolExecutor.
    Each protein writes independent output files, so this is safe.
    """
    try:
        print(f"[START] {protein}", flush=True)

        outpath = './CS_analyses/'
        os.makedirs(outpath, exist_ok=True)

        models = [
            'rewBV2006',
            'rewBV2006ang',
            'rewM9',
            'rewM9DH',
            'rewM9DH_scaled',
            'rewM10',
            'rewM10DH',
            'rewM10DH_scaled',
            'rewM11',
            'rewM11DH',
            'rewM11DH_scaled',
            'rewM12',
            'rewM12DH',
            'rewM12DH_scaled',
            'rewM14',
            'rewM14DH',
            'rewM14DH_scaled'
        ]

        nucleus = 'CA'
        all_types = ['CA', 'CB', 'HA', 'C', 'H', 'HN', 'N']

        label_dict = {'CA': 'Cα', 'HA': 'Hα', 'CB': 'Cβ', 'C': "C'"}
        axlim_dict = {'CA': 4, 'HA': 1, 'CB': 4, 'C': 4}

        # Work out which model/protein combinations actually need to be run.
        # Safeguard 1: do not redo work if the final per-model CSV already exists.
        # Safeguard 2: do not run a model unless reweighting results exist.
        runnable_models = []

        for model in models:
            model_dir = f"../../MD_DATA/{protein}/{model}"
            final_csv = os.path.join(outpath, f"{protein}_{model}_CS_validation.csv")

            if model in ("rewBV2006", "rewBV2006ang"):
                wopt_filename = "wopt.npy"
                wopt_array_filename = "wopt_array.npy"
            else:
                wopt_filename = "log_lambda_wopt.npy"
                wopt_array_filename = "log_lambda_wopt_array.npy"

            wopt_path = os.path.join(model_dir, wopt_filename)
            wopt_array_path = os.path.join(model_dir, wopt_array_filename)

            if os.path.exists(final_csv):
                print(
                    f"[SKIP] {protein} {model}: final CSV already exists: {final_csv}",
                    flush=True
                )
                continue

            if not os.path.exists(wopt_path):
                print(
                    f"[SKIP] {protein} {model}: no reweighting result found: {wopt_path}",
                    flush=True
                )
                continue

            # The CS validation loop also needs the full theta-scan weight array.
            # Skipping here avoids failing the whole protein if only wopt.npy exists.
            if not os.path.exists(wopt_array_path):
                print(
                    f"[SKIP] {protein} {model}: missing required file: {wopt_array_path}",
                    flush=True
                )
                continue

            runnable_models.append(model)

        if len(runnable_models) == 0:
            print(f"[DONE] {protein}: no new model/protein combinations to run", flush=True)
            return protein, None

        traj_names = np.sort(find_xtc_filenames(f'../../MD_DATA/{protein}/'))

        n_models = len(runnable_models)
        ncols = 4

        fig, axes = plt.subplots(
            nrows=n_models,
            ncols=ncols,
            figsize=(ncols * 4., 3.5 * n_models),
            squeeze=False
        )

        # load CS data
        gen_dfs = []

        for traj in traj_names:
            cs_file = f'../../MD_DATA/{protein}/CS/CS_{traj[:-4]}.csv'
            gen_dfs.append(load_generated_shifts(cs_file))

        gen_df = pd.concat(gen_dfs, axis=1, join='outer', ignore_index=True)

        # experimental CS data
        exp_dict = load_and_filter_experimental_shifts(
            f'../../ORTHOGONAL_EXP_DATA/{protein}/CS.dat',
            '../../code/cs_stat_aa_filt.csv',
            ATOM_TYPES=all_types
        )

        # get standardised deltas
        std_delta_cs, keys = standardize_deltas(
            gen_df,
            exp_dict,
            predictor='Sparta+'
        )

        # load RC shifts
        potenci = load_potenci_shifts(
            f'../../ORTHOGONAL_EXP_DATA/potenci_shifts/{protein}_RCshifts.dat',
            ATOM_TYPES=all_types
        )

        # calculate experimental secondary shifts
        potenci_residues = []
        potenci_shifts = []

        for key in list(potenci.keys()):
            if key[1] == nucleus:
                potenci_residues.append(key[0])
                potenci_shifts.append(potenci[key])

        exp_residues = []
        exp_shifts = []

        for key in list(exp_dict.keys()):
            if key[1] == nucleus:
                exp_residues.append(key[0])
                exp_shifts.append(exp_dict[key])

        potenci_residues = np.array(potenci_residues)
        potenci_shifts = np.array(potenci_shifts)
        exp_residues = np.array(exp_residues)
        exp_shifts = np.array(exp_shifts)

        common_residues = np.intersect1d(potenci_residues, exp_residues)

        indices_p = np.where(np.isin(potenci_residues, common_residues))[0]
        indices_e = np.where(np.isin(exp_residues, common_residues))[0]

        SSexp = exp_shifts[indices_e] - potenci_shifts[indices_p]

        for i, model in enumerate(runnable_models):
            exp, residues, priorPF, rewPF, chi2, phi, thetas, stats, idxs = load_pf_data(
                protein,
                model_name=model
            )

            model_dir = f"../../MD_DATA/{protein}/{model}"

            if model in ("rewBV2006", "rewBV2006ang"):
                wopt_filename = "wopt.npy"
                wopt_array_filename = "wopt_array.npy"
            else:
                wopt_filename = "log_lambda_wopt.npy"
                wopt_array_filename = "log_lambda_wopt_array.npy"

            wopt = np.load(os.path.join(model_dir, wopt_filename))
            wopt_array = np.load(os.path.join(model_dir, wopt_array_filename))

            savedf = pd.DataFrame()
            savedf['Neff'] = phi

            expPF = np.array(exp["PF"][idxs])

            rmse_prior = RMSE(np.log10(expPF), np.log10(priorPF))
            rmse_post = RMSE(np.log10(expPF), np.log10(rewPF))

            r_prior = scipy.stats.pearsonr(
                np.log10(expPF),
                np.log10(priorPF)
            )[0]

            r_post = scipy.stats.pearsonr(
                np.log10(expPF),
                np.log10(rewPF)
            )[0]

            # calculate RMSE to validation data
            validation_rmse = []

            for W in wopt_array:
                rmse = get_RMSE(std_delta_cs, weights=W)
                validation_rmse.append(rmse)

            cs_prior_rmse = get_RMSE(
                std_delta_cs,
                weights=np.ones(std_delta_cs.shape[1]) / std_delta_cs.shape[1]
            )

            savedf['CS_RMSE'] = validation_rmse

            # --- Plot 1: neff vs RMSE with validation data ---
            ax = axes[i, 0]
            ax.scatter(phi, validation_rmse, c="tab:red")
            ax.set_xlabel("N$_{eff}$")
            ax.set_ylabel("CS RMSE")
            ax.set_title(f"{model}")
            ax.set_ylim(0.5 * validation_rmse[0], 1.5 * validation_rmse[0])

            # --- Plot 2: Per-residue PFs ---
            ax = axes[i, 1]
            ax.set_title(
                f"Prior: RMSE={rmse_prior:.2f}\n"
                f"Posterior: RMSE={rmse_post:.2f}"
            )
            ax.bar(residues, expPF, color="lightblue", alpha=0.5, label="Exp.")
            ax.scatter(residues, priorPF, color="tab:blue", label="Prior")
            ax.scatter(residues, rewPF, color="tab:red", label="Posterior")
            ax.set_yscale("log")
            ax.set_ylim(0, 1e8)
            ax.set_xlabel("Residue")
            ax.set_ylabel("PF")

            # --- Plot 3: PF exp vs calc prior and posterior ---
            ax = axes[i, 2]
            ax.set_title(
                f"Prior: r={r_prior:.2f}\n"
                f"Posterior: r={r_post:.2f}"
            )
            ax.scatter(expPF, priorPF, color="tab:blue", label="Prior")
            ax.scatter(expPF, rewPF, color="tab:red", label="Posterior")
            ax.plot(
                [-10, 1e10],
                [-10, 1e10],
                color='black',
                linestyle='dashed',
                linewidth=1
            )
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_ylim(0, 1e8)
            ax.set_xlim(0, 1e8)
            ax.set_xlabel("Exp. PF")
            ax.set_ylabel("Calc. PF")

            # --- Plot 4: secondary CA shifts exp vs calculated prior and posterior ---
            ax = axes[i, 3]

            ax.set_title(
                f"Prior: RMSE={cs_prior_rmse:.2f}\n"
                f"Posterior: RMSE={validation_rmse[-1]:.2f}"
            )

            ax.bar(
                exp_residues[indices_e],
                SSexp,
                color='lightblue',
                alpha=0.5,
                label='Exp.'
            )

            ax.set_ylim(-axlim_dict[nucleus], axlim_dict[nucleus])
            ax.set_xlabel("Residue")
            ax.set_ylabel(f"Δδ({label_dict[nucleus]}) (ppm)")

            calc_vals = []
            calc_residues = []

            for res in potenci_residues:
                try:
                    calc_vals.append(list(gen_df.loc[res, nucleus]))
                    calc_residues.append(res)
                except KeyError:
                    continue

            calc_residues = np.array(calc_residues)
            calc_vals = np.array(calc_vals).T

            prior = np.average(calc_vals, axis=0)
            posterior = np.average(calc_vals, weights=wopt, axis=0)

            indices_p2 = np.where(np.isin(potenci_residues, calc_residues))[0]
            potenci_for_calc = potenci_shifts[indices_p2]

            SSprior = prior - potenci_for_calc
            SSposterior = posterior - potenci_for_calc

            ax.plot(
                calc_residues,
                SSprior,
                color="tab:blue",
                label="Prior",
                linewidth=1.5
            )

            ax.plot(
                calc_residues,
                SSposterior,
                color="tab:red",
                label="Posterior",
                linewidth=1.5
            )

            ax.legend(loc='lower right')

            # save validation diff data
            savedf.to_csv(outpath + f"{protein}_{model}_CS_validation.csv")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        plt.savefig(
            outpath + f"{protein}_HDXrew_CS_all.png",
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(f"[DONE] {protein}", flush=True)
        return protein, None

    except Exception:
        err = traceback.format_exc()

        try:
            plt.close("all")
        except Exception:
            pass

        return protein, err


def main():
    parser = argparse.ArgumentParser(
        description="Run HDX-reweighted CS validation plots in parallel over proteins."
    )

    parser.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes. Default: 4."
    )

    args = parser.parse_args()

    outpath = './CS_analyses/'
    os.makedirs(outpath, exist_ok=True)

    proteins = ['AAQAA3', 'ACTR', 'aSyn', 'BecN', 'ChiZ', 'd131d', 'drkN', 'NTail']

    n_workers = min(args.n_workers, len(proteins))

    print(f"Running {len(proteins)} proteins with {n_workers} workers", flush=True)

    failures = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(analyse_protein, protein): protein
            for protein in proteins
        }

        for future in as_completed(futures):
            protein = futures[future]

            try:
                result_protein, error = future.result()
            except Exception:
                result_protein = protein
                error = traceback.format_exc()

            if error is None:
                print(f"[OK] {result_protein}", flush=True)
            else:
                print(f"[FAILED] {result_protein}", flush=True)
                print(error, flush=True)
                failures.append(result_protein)

    if len(failures) > 0:
        print("\nThe following proteins failed:", flush=True)
        for protein in failures:
            print(f"  - {protein}", flush=True)
    else:
        print("\nAll proteins completed successfully.", flush=True)


if __name__ == "__main__":
    main()


