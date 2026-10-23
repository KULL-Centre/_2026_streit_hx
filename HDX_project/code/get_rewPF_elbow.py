import os
import numpy as np
import argparse


parser = argparse.ArgumentParser(
    description="Extract reweighted lnPF values for experimental residues from the elbow weights."
)
parser.add_argument("--name", type=str, required=True, help="Name of the protein.")
args = parser.parse_args()

protname = args.name
directory = protname

models = [
    'rewBV2006', 'rewBV2006ang',
    'rewM9', 'rewM9DH', 'rewM9DH_scaled',
    'rewM10', 'rewM10DH', 'rewM10DH_scaled',
    'rewM11', 'rewM11DH', 'rewM11DH_scaled',
    'rewM12', 'rewM12DH', 'rewM12DH_scaled',
    'rewM14', 'rewM14DH', 'rewM14DH_scaled'
]

model_params = {
    'rewBV2006': {'beta_c': 0.35, 'beta_h': 2.00},
    'rewBV2006ang': {'beta_c': 0.89, 'beta_h': 4.39}
}

for model in models:
    model_dir = os.path.join(directory, model)

    exp_residues = np.loadtxt(
        os.path.join(model_dir, 'exp_residues.txt'),
        dtype=int,
        ndmin=1
    )

    all_amides = np.loadtxt(
        os.path.join(model_dir, 'amideresidues.txt'),
        dtype=int,
        ndmin=1
    )

    if model in model_params:
        weights = np.load(
            os.path.join(model_dir, 'wopt_elbow.npy')
        )
        contacts = np.load(
            os.path.join(model_dir, 'contacts_res_frames.npy')
        )
        hbonds = np.load(
            os.path.join(model_dir, 'hbonds_res_frames.npy')
        )

        # BV reweighting uses the first occurrence in the full residue list
        exp_idxs = np.array(
            [list(all_amides).index(resid) for resid in exp_residues],
            dtype=int
        )

        lnPF = (
            model_params[model]['beta_c'] * contacts
            + model_params[model]['beta_h'] * hbonds
        )

        rew_lnPF = np.average(
            lnPF[:, exp_idxs],
            axis=0,
            weights=weights
        )

        output_file = os.path.join(
            model_dir,
            'rew_lnPF_elbow.txt'
        )

    else:
        weights = np.load(
            os.path.join(model_dir, 'log_lambda_wopt_elbow.npy')
        )

        O = None
        for filename in ['O.npy', 'fopen.npy', 'fopenDH.npy']:
            filepath = os.path.join(model_dir, filename)
            if os.path.exists(filepath):
                O = np.load(filepath)
                break

        if O is None:
            raise FileNotFoundError(
                f"No O.npy, fopen.npy or fopenDH.npy found in {model_dir}"
            )

        # Reproduce the filtering used in reweight_log_lambda:
        # remove columns that never sample an open state
        valid = np.where(np.mean(O, axis=0) > 0)[0]

        valid_amides = all_amides[valid]

        # First occurrence in the filtered residue list
        idxs_in_valid = np.array(
            [list(valid_amides).index(resid) for resid in exp_residues],
            dtype=int
        )

        # Map filtered indices back onto columns of the full O array
        exp_idxs = valid[idxs_in_valid]

        mean_fopen = np.average(
            O[:, exp_idxs],
            axis=0,
            weights=weights
        )

        rew_lnPF = -np.log(mean_fopen)

        output_file = os.path.join(
            model_dir,
            'log_lambda_rew_lnPF_elbow.txt'
        )

    if not np.array_equal(all_amides[exp_idxs], exp_residues):
        raise ValueError(
            f"{model}: selected residues do not match exp_residues.txt"
        )

    np.savetxt(output_file, rew_lnPF)