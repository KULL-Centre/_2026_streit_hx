import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# to do/implement
# warning and save residues in the covered peptide range that never sample the open state so the user knows it
# make it easier to combine peptides from multiple conditions i.e. multiple kint values will be needed (e.g. have condition indices assigned and connected with
# different sets of kint)


class HDXReweighter:
    """
    Bayesian reweighting for HDX-MS data using maximum-entropy regularisation.
    """

    def __init__(self, k, O, peptides, times, F_exp, sigma, theta=1.0, n_skip=1):
        self.k = np.array(k)
        self.O = np.array(O)
        self.peptides = peptides
        self.times = np.array(times)
        # ensures spurious datapoints above 1.0 fractional uptake are not penalising and all values are minimum 0.0
        self.F_exp = np.maximum(np.minimum(np.array(F_exp), float(1.0)), float(0.0)) # CONSIDER REMOVING THIS AS IT MAY CAUSE ISSUES WITH RESULTING IN FAKE PLATEAUS
        self.sigma = np.array(sigma)
        self.theta = theta

        self.n_frames = O.shape[1]
        self.n_res = O.shape[0]
        self.n_skip = n_skip

    def _weights_from_params(self, u):
        """Softmax reparameterisation: u -> normalised positive weights w."""
        e_u = np.exp(u - np.max(u))  # numerical stability
        return e_u / np.sum(e_u)

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
        """
        Predict fractions for a set of weights w.

        Parameters
        ----------
        w : array_like, shape (n_frames,)
            Weights must sum to 1 and be non-negative.

        Returns
        -------
        F_pred : array_like
            Predicted fractions for each peptide/timepoint.
        """

        # normalise just in case
        w = w/np.sum(w)
        
        return self._predicted_fraction(np.array(w))[0]

    def predict_with_params(self, u):
        """
        Predict fractions from **unconstrained parameters** u (softmax).

        Parameters
        ----------
        u : array_like, shape (n_frames,)
            Unconstrained parameters.

        Returns
        -------
        F_pred : array_like
            Predicted fractions for each peptide/timepoint.
        """
        w = self._weights_from_params(np.array(u))
        return self._predicted_fraction(w)[0]


    def _loss_and_grad(self, u):
        w = self._weights_from_params(u)
        F_pred, o_i = self._predicted_fraction(w)
        r = F_pred - self.F_exp

        grad_w = np.zeros_like(w)

        for alpha, pep in enumerate(self.peptides):
            t = self.times[alpha]

            # skip first n_skip residues, mask non-exchanging
            pep_idx = np.array(pep)  # ensure numpy array
            mask = self.k[pep_idx][self.n_skip:] > 0
            k_masked = self.k[pep_idx][self.n_skip:][mask]
            o_masked = o_i[pep_idx][self.n_skip:][mask]
            O_masked = self.O[pep_idx[self.n_skip:][mask], :]  # safe indexing

            exp_term = np.exp(-k_masked * t * o_masked)[:, np.newaxis]

            N_alpha = len(k_masked)

            grad_w += (r[alpha] / self.sigma[alpha]**2) * (1.0 / N_alpha) * np.sum(
                k_masked[:, np.newaxis] * t * O_masked * exp_term,
                axis=0)

        # Maximum-entropy term
        grad_w += self.theta * (1 + np.log(w / (1.0 / self.n_frames)))

        # Softmax chain rule
        grad_u = w * (grad_w - np.sum(w * grad_w))

        # Loss function
        loss = 0.5 * np.sum((r / self.sigma)**2) + self.theta * np.sum(w * np.log(w * self.n_frames))
        return loss, grad_u


    def fit(self, w0=None, method='L-BFGS-B', options={'maxiter': 5000}):
        """Optimise frame weights."""
        # Default: uniform initial weights
        if w0 is None:
            w0 = np.ones(self.n_frames) / self.n_frames
        else:
            w0 = np.array(w0, dtype=float)
            w0 /= np.sum(w0)  # normalise
        # Convert initial weights to unconstrained parameters for softmax
        u0 = np.log(w0)  
        u0 -= np.max(u0)  # shift so max(u0) = 0

        # Objective function for optimizer
        def fun(u):
            loss, grad = self._loss_and_grad(u)
            return loss, grad
        # Minimise
        result = minimize(fun, u0, jac=True, method=method, options=options)
        # Store optimal parameters and weights
        self.u_opt = result.x
        self.w_opt = self._weights_from_params(result.x)
        return result

    # fit from initial softmax u values
    def oldfit(self, u0=None, method='L-BFGS-B', options={'maxiter': 5000}):
        """Optimise frame weights."""
        if u0 is None:
            u0 = np.zeros(self.n_frames)

        def fun(u):
            loss, grad = self._loss_and_grad(u)
            return loss, grad

        result = minimize(fun, u0, jac=True, method=method, options=options)
        self.u_opt = result.x
        self.w_opt = self._weights_from_params(result.x)
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
        



