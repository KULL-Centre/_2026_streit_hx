import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# to do/implement
# warning and save residues in the covered peptide range that never sample the open state so the user knows it
# make it easier to combine peptides from multiple conditions i.e. multiple kint values will be needed (e.g. have condition indices assigned and connected with
# different sets of kint)

class HDXReweighterFast:
    def __init__(self, k, O, peptides, times, F_exp, sigma, theta=1.0, n_skip=2, dtype=np.float64):
        self.k = np.asarray(k, dtype=dtype)
        self.O = np.asarray(O, dtype=dtype)
        self.peptides = peptides
        self.times = np.asarray(times, dtype=dtype)
        self.F_exp = np.clip(np.asarray(F_exp, dtype=dtype), 0.0, 1.0)
        self.sigma = np.asarray(sigma, dtype=dtype)
        self.theta = dtype(theta)
        self.n_skip = n_skip
        self.dtype = dtype

        self.n_res, self.n_frames = self.O.shape
        self.n_data = len(self.peptides)

        if len(self.times) != self.n_data: raise ValueError("times must have the same length as peptides.")
        if len(self.F_exp) != self.n_data: raise ValueError("F_exp must have the same length as peptides.")
        if len(self.sigma) != self.n_data: raise ValueError("sigma must have the same length as peptides.")
        if np.any(self.sigma <= 0): raise ValueError("All sigma values must be > 0.")

        # Precompute valid residue indices and k-values for each datapoint
        self.valid_idx = []
        self.valid_k = []
        self.N_alpha = np.empty(self.n_data, dtype=np.int32)

        for a, pep in enumerate(self.peptides):
            pep = np.asarray(pep, dtype=np.int32)
            idx = pep[self.n_skip:]
            mask = self.k[idx] > 0
            idx = idx[mask]

            if idx.size == 0: raise ValueError(f"Datapoint {a} has no exchanging residues after n_skip/masking.")

            self.valid_idx.append(idx)
            self.valid_k.append(self.k[idx])
            self.N_alpha[a] = idx.size

        # Prior used for KL regularisation; set by fit()
        self.w_prior = None

    def _weights_from_params(self, u):
        u = np.asarray(u, dtype=self.dtype)
        u = u - np.max(u)
        e = np.exp(u)
        return e / np.sum(e)

    def _predict_from_o(self, o):
        F_pred = np.empty(self.n_data, dtype=self.dtype)
        exp_cache = [None] * self.n_data

        for a in range(self.n_data):
            idx = self.valid_idx[a]
            kk = self.valid_k[a]
            t = self.times[a]

            e = np.exp(-kk * t * o[idx])
            exp_cache[a] = e
            F_pred[a] = 1.0 - np.mean(e)

        return F_pred, exp_cache

    def _predicted_fraction(self, w):
        """Compute predicted fraction deuterated for each peptide/timepoint."""
        w = np.asarray(w, dtype=self.dtype)
        w /= np.sum(w)

        o_i = self.O @ w
        F_pred, _ = self._predict_from_o(o_i)

        return F_pred, o_i

    def predict_with_weights(self, w):
        w = np.asarray(w, dtype=self.dtype)
        w /= np.sum(w)
        o = self.O @ w
        F_pred, _ = self._predict_from_o(o)
        return F_pred

    def _loss_and_grad(self, u):
        if self.w_prior is None: raise RuntimeError("Prior weights have not been set. Call fit() first.")

        w = self._weights_from_params(u)

        # Ensemble-averaged open probabilities
        o = self.O @ w

        # Predicted uptake
        F_pred, exp_cache = self._predict_from_o(o)
        r = F_pred - self.F_exp

        # Data-term derivative in residue space
        coeff = np.zeros(self.n_res, dtype=self.dtype)

        for a in range(self.n_data):
            idx = self.valid_idx[a]
            kk = self.valid_k[a]
            t = self.times[a]
            e = exp_cache[a]

            pref = (r[a] / (self.sigma[a] ** 2)) / self.N_alpha[a]
            coeff[idx] += pref * (kk * t * e)

        # d(loss_data)/dw
        grad_w = self.O.T @ coeff

        # Maximum-entropy regularisation relative to the ACTUAL PRIOR:
        #
        # D_KL(w || w0) = sum_i w_i log(w_i / w0_i)
        #
        log_ratio = np.log(w / self.w_prior)
        grad_w += self.theta * (1.0 + log_ratio)

        # Softmax chain rule
        grad_u = w * (grad_w - np.sum(w * grad_w))

        # Loss
        loss_data = 0.5 * np.sum((r / self.sigma) ** 2)
        loss_reg = self.theta * np.sum(w * log_ratio)
        loss = loss_data + loss_reg

        return float(loss), grad_u.astype(np.float64)

    def fit(self, w0=None, method="L-BFGS-B", options=None):
        if options is None:
            options = {"maxiter": 50000, "maxfun": 100000, "gtol": 1e-7, "ftol": 1e-12, "maxcor": 10, "maxls": 20}

        if w0 is None:
            w0 = np.ones(self.n_frames, dtype=self.dtype) / self.n_frames
        else:
            w0 = np.asarray(w0, dtype=self.dtype)

            if w0.ndim != 1 or len(w0) != self.n_frames: raise ValueError(f"w0 must have shape ({self.n_frames},), got {w0.shape}.")
            if np.any(~np.isfinite(w0)): raise ValueError("w0 contains non-finite values.")
            if np.any(w0 <= 0): raise ValueError("All prior weights w0 must be > 0 for KL reweighting with this implementation.")

            w0 /= np.sum(w0)

        # Store the prior used by the KL term
        self.w_prior = w0.copy()

        # Starting point corresponds exactly to the prior
        u0 = np.log(self.w_prior)
        u0 -= np.max(u0)

        result = minimize(
            self._loss_and_grad,
            u0.astype(np.float64),
            jac=True,
            method=method,
            options=options,
        )

        self.u_opt = result.x.astype(self.dtype)
        self.w_opt = self._weights_from_params(self.u_opt)

        return result

    def predict(self):
        """Predict using optimised weights."""
        return self._predicted_fraction(self.w_opt)[0]

    def calc_chi2(self, w):
        """
        Compute reduced chi^2:
        chi2_red = chi2 / N_data
        """
        w = np.asarray(w, dtype=float)
        w /= np.sum(w)

        F_pred, _ = self._predicted_fraction(w)
        r = F_pred - self.F_exp

        chi2 = np.sum((r / self.sigma) ** 2)
        chi2_red = chi2 / len(self.F_exp)

        return chi2_red

    def calc_rmse(self, w):
        """Compute RMSE between calculated and experimental values."""
        w = np.asarray(w, dtype=float)
        w /= np.sum(w)

        F_pred, _ = self._predicted_fraction(w)
        rmse = np.sqrt(np.mean((F_pred - self.F_exp) ** 2))

        return rmse

    def calc_neff(self, w, w0=None):
        """
        Relative effective ensemble size:

            phi = exp[-D_KL(w || w0)]

        where

            D_KL(w || w0) = sum_i w_i log(w_i / w0_i)

        phi = 1 for the unchanged prior.
        """
        w = np.asarray(w, dtype=float)
        w /= np.sum(w)

        if w0 is None:
            w0 = np.ones(self.n_frames, dtype=float) / self.n_frames
        else:
            w0 = np.asarray(w0, dtype=float)
            if w0.ndim != 1 or len(w0) != self.n_frames: raise ValueError(f"w0 must have shape ({self.n_frames},), got {w0.shape}.")
            if np.any(w0 <= 0): raise ValueError("All prior weights w0 must be > 0 when calculating KL divergence.")
            w0 /= np.sum(w0)

        mask = w > 0
        D_kl = np.sum(w[mask] * np.log(w[mask] / w0[mask]))
        Neff = np.exp(-D_kl)

        return Neff

    def plot_uptake_curves(self, w0=None, wopt=None, savefile=None, dpi=100, ncols=4, nrows=5):
        """Plot uptake curves in pages with a fixed number of peptides per figure."""

        if w0 is None:
            w0 = np.ones(self.n_frames) / self.n_frames
        else:
            w0 = np.asarray(w0, dtype=float)
            w0 /= np.sum(w0)

        if not hasattr(self, "w_opt") and wopt is None: raise ValueError("No optimised weights found. Run fit() first or supply wopt.")

        # Detect peptide boundaries from time resets
        peptide_groups = []
        current_group = [0]

        for i in range(1, len(self.times)):
            if self.times[i] < self.times[i - 1]:
                peptide_groups.append(current_group)
                current_group = [i]
            else:
                current_group.append(i)

        peptide_groups.append(current_group)

        F_prior_all = self.predict_with_weights(w0)
        F_opt_all = self.predict() if wopt is None else self.predict_with_weights(wopt)

        n_per_page = nrows * ncols
        n_pages = int(np.ceil(len(peptide_groups) / n_per_page))

        for page in range(n_pages):
            groups = peptide_groups[page * n_per_page:(page + 1) * n_per_page]

            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharex=False, sharey=True)
            axes = np.atleast_1d(axes).ravel()

            for ax, group in zip(axes, groups):
                times = self.times[group]
                F_exp = self.F_exp[group]
                sigma = self.sigma[group]
                pep = self.peptides[group[0]]
                start, end = pep[0] + 1, pep[-1] + 1

                F_prior = F_prior_all[group]
                F_opt = F_opt_all[group]

                ax.errorbar(times, F_exp, yerr=sigma, fmt='o', capsize=3, color='black', label='Exp.')
                ax.plot(times, F_prior, '--', color='tab:blue', label='Prior')
                ax.plot(times, F_opt, '-', color='tab:red', label='Post.')

                ax.set_title(f"Peptide {start}-{end}")
                ax.set_xscale('log')
                ax.set_ylim(0, 1.05)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Fractional uptake")
                ax.grid(True, which="both", linestyle=":")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            for ax in axes[len(groups):]:
                ax.remove()

            if len(groups) > 0:
                axes[0].legend()

            plt.tight_layout()

            if savefile is not None:
                if n_pages == 1:
                    filename = savefile
                else:
                    if '.' in savefile:
                        base, ext = savefile.rsplit('.', 1)
                        filename = f"{base}_page{page + 1}.{ext}"
                    else:
                        filename = f"{savefile}_page{page + 1}"

                plt.savefig(filename, dpi=dpi, bbox_inches='tight')

            plt.show()
            plt.close(fig)