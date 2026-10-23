import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# to do/implement
# warning and save residues in the covered peptide range that never sample the open state so the user knows it
# make it easier to combine peptides from multiple conditions i.e. multiple kint values will be needed (e.g. have condition indices assigned and connected with
# different sets of kint)

class HDXReweighterFast:
    def __init__(self, k, O, peptides, times, F_exp, sigma, theta=1.0, n_skip=1, dtype=np.float64):
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

        # Precompute valid residue indices and k-values for each datapoint
        self.valid_idx = []
        self.valid_k = []
        self.N_alpha = np.empty(self.n_data, dtype=np.int32)

        for a, pep in enumerate(self.peptides):
            pep = np.asarray(pep, dtype=np.int32)
            idx = pep[self.n_skip:]
            mask = self.k[idx] > 0
            idx = idx[mask]

            if idx.size == 0:
                raise ValueError(f"Datapoint {a} has no exchanging residues after n_skip/masking.")

            self.valid_idx.append(idx)
            self.valid_k.append(self.k[idx])
            self.N_alpha[a] = idx.size

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
        F_pred = []
        o_i = self.O @ w  # ensemble-averaged open fractions
        for alpha, pep in enumerate(self.peptides):
            t = self.times[alpha]
            mask = self.k[pep][self.n_skip:] > 0 # non-proline mask (their kint is 0 s-1), first n_skip residues ignored due back exchange
            F_alpha = np.mean(1 - np.exp(-self.k[pep][self.n_skip:][mask] * t * o_i[pep][self.n_skip:][mask])) # predicted fractional uptake
            F_pred.append(F_alpha)
        return np.array(F_pred), o_i

    def predict_with_weights(self, w):
        w = np.asarray(w, dtype=self.dtype)
        w = w / np.sum(w)
        o = self.O @ w
        F_pred, _ = self._predict_from_o(o)
        return F_pred

    def _loss_and_grad(self, u):
        w = self._weights_from_params(u)

        # First BLAS call
        o = self.O @ w

        # Cheap loop over datapoints/peptides
        F_pred, exp_cache = self._predict_from_o(o)

        r = F_pred - self.F_exp

        # Accumulate residue-space coefficients
        coeff = np.zeros(self.n_res, dtype=self.dtype)

        for a in range(self.n_data):
            idx = self.valid_idx[a]
            kk = self.valid_k[a]
            t = self.times[a]
            e = exp_cache[a]

            pref = (r[a] / (self.sigma[a] ** 2)) / self.N_alpha[a]
            coeff[idx] += pref * (kk * t * e)

        # Second BLAS call
        grad_w = self.O.T @ coeff

        # Maximum-entropy term
        grad_w += self.theta * (1.0 + np.log(w * self.n_frames))

        # Softmax chain rule
        grad_u = w * (grad_w - np.sum(w * grad_w))

        loss_data = 0.5 * np.sum((r / self.sigma) ** 2)
        loss_reg = self.theta * np.sum(w * np.log(w * self.n_frames))
        loss = loss_data + loss_reg

        return float(loss), grad_u.astype(np.float64)

    def fit(self, w0=None, method="L-BFGS-B", options=None):
        if options is None:
            options = {"maxiter": 50000, "maxfun": 100000, "gtol": 1e-7, "ftol": 1e-12, "maxcor": 10, "maxls": 20}

        if w0 is None:
            w0 = np.ones(self.n_frames, dtype=self.dtype) / self.n_frames
        else:
            w0 = np.asarray(w0, dtype=self.dtype)
            w0 = w0 / np.sum(w0)

        u0 = np.log(w0)
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

        Parameters
        ----------
        w : array_like, shape (n_frames,)
            Weights (non-negative, sum to 1).

        Returns
        -------
        chi2_red : float
        """
        w = np.array(w, dtype=float)
        w /= np.sum(w)  # just in case
        F_pred, _ = self._predicted_fraction(w)
        r = F_pred - self.F_exp
        chi2 = np.sum((r / self.sigma) ** 2)
        chi2_red = chi2 / len(self.F_exp)
        return chi2_red
    
    def calc_rmse(self, w):
        """
        Compute RMSE between calculated and experimental values:
        
        Parameters
        ----------
        w : array_like, shape (n_frames,)
            Weights (non-negative, sum to 1).

        Returns
        -------
        rmse : float
        """
        w = np.array(w, dtype=float)
        w /= np.sum(w)  # just in case
        F_pred, _ = self._predicted_fraction(w)
        rmse = np.sqrt(np.mean((F_pred - self.F_exp)**2))
        return rmse
    
    def calc_neff(self, w, w0=None):
        """
        Compute effective number of frames (Neff) from KL divergence.

        Parameters
        ----------
        w : array_like, shape (n_frames,)
            Weights (non-negative, sum to 1).
        w0 : array_like, shape (n_frames,)
            Prior weights (non-negative, sum to 1)

        Returns
        -------
        Neff : float
        """
        w = np.array(w, dtype=float)
        w /= np.sum(w)
        # Uniform prior
        if w0 is None:
            w0 = 1.0 / self.n_frames
        w0 = np.array(w0, dtype=float)
        w0/= np.sum(w0)
        # Avoid log(0)
        mask = w > 0
        D_kl = np.sum(w[mask] * np.log(w[mask] / w0))
        Neff = np.exp(-D_kl)
        return Neff
    
    
    def plot_uptake_curves(self, w0=None, wopt=None, savefile=None):
        """
        Plot uptake curves (one subplot per peptide).

        Parameters
        ----------
        w0 : array_like or None
            Prior weights. If None, uniform weights are used.
        """

        if w0 is None:
            w0 = np.ones(self.n_frames) / self.n_frames
        else:
            w0 = np.array(w0, dtype=float)
            w0 /= np.sum(w0)

        if not hasattr(self, "w_opt") and wopt is None:
            raise ValueError("No optimised weights found. Run fit() first or supply wopt.")

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

        n_peptides = len(peptide_groups)

        fig, axes = plt.subplots(
            n_peptides, 1,
            figsize=(6, 4 * n_peptides),
            sharex=False
        )

        if n_peptides == 1:
            axes = [axes]

        for ax, group in zip(axes, peptide_groups):

            times = self.times[group]
            F_exp = self.F_exp[group]
            sigma = self.sigma[group]
            pep = self.peptides[group[0]]
            start, end = pep[0] + 1, pep[-1] + 1
            
            # Predictions
            F_prior = self.predict_with_weights(w0)[group]
            if wopt is None:
                F_opt = self.predict()[group]
            else:
                F_opt = self.predict_with_weights(wopt)[group]

            # Plot experimental
            ax.errorbar(
                times, F_exp,
                yerr=sigma,
                fmt='o',
                capsize=3,
                label='Exp.',
                color = 'black'
            )

            # Plot prior
            ax.plot(
                times, F_prior,
                linestyle='--',
                label='Prior',
                color='tab:blue'
            )

            # Plot optimised
            ax.plot(
                times, F_opt,
                linestyle='-',
                label='Post.',
                color='tab:red'
            )

            ax.set_title(f"Peptide {start}-{end}")
            ax.set_xscale('log')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Fractional uptake")
            ax.set_ylim(0, 1.05)
            ax.legend()
            ax.grid(True, which="both", linestyle=":")

        if savefile is None:
            plt.tight_layout()
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(savefile, dpi=200)
            plt.show()
        



