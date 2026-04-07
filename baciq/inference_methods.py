from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import pymc as pm
import sys
from typing import Tuple


def get_proteins_and_indices(data: pd.DataFrame) -> Tuple[pd.Series, np.array]:
    if 'Protein ID' not in data:
        raise KeyError('Expected "Protein ID" in input data')

    proteins = data['Protein ID'].unique()
    idx = np.zeros(len(data), dtype=int)
    for i, p in enumerate(proteins):
        idx[data['Protein ID'] == p] = i

    return proteins, idx


class Base_Modeler(ABC):
    def __init__(self,
                 samples: int,
                 chains: int,
                 tuning: int,
                 channel: str):
        super().__init__()
        self.samples = samples
        self.chains = chains
        self.tuning = tuning
        self.channel = channel

    @abstractmethod
    def fit_quantiles(self,
                      data: pd.DataFrame,
                      quantiles: np.array) -> pd.DataFrame:
        pass

    @abstractmethod
    def fit_histogram(self,
                      data: pd.DataFrame,
                      bin_width: float) -> pd.DataFrame:
        pass


class PYMC_Model(Base_Modeler):
    def __init__(self, samples, chains, tuning, channel):
        super().__init__(samples, chains, tuning, channel)

    def fit_quantiles(self, data, quantiles):
        '''
        Fit the model with supplied data returning quantiles
        '''
        bin_width = 0.0001
        proteins, samples = self.mcmc_sample(data, bin_width=bin_width)

        bins = np.array(range(samples.shape[1])) * bin_width
        # scale counts by total, get cumsum
        quants = np.cumsum(samples / np.sum(samples, axis=1)[:, None], axis=1)

        result = None
        for q in quantiles:
            idx = np.argmax(quants >= q, axis=1)
            if result is None:
                result = bins[idx].reshape((len(idx), 1))
            else:
                result = np.hstack((result, bins[idx].reshape((len(idx), 1))))

        result = pd.DataFrame(result,
                              index=proteins,
                              columns=['{:.9g}'.format(q) for q in quantiles])
        result.index.name = "Protein ID"
        return result

    def fit_histogram(self, data, bin_width):
        '''
        Fit the model with supplied data returning histogram
        '''
        proteins, result = self.mcmc_sample(data, bin_width)

        result = pd.DataFrame(result,
                              index=proteins,
                              columns=[bin_width*i
                                       for i in range(result.shape[1])])
        result.index.name = "Protein ID"
        return result

    def mcmc_sample(self, data, bin_width):
        proteins, idx = get_proteins_and_indices(data)
        with pm.Model():
            τ = pm.Gamma('τ', alpha=7.5, beta=1)
            μ = pm.TruncatedNormal('μ', mu=0.5, sigma=1, lower=0, upper=1,
                                   shape=len(proteins))
            κ = pm.Exponential('κ', τ, shape=len(proteins))
            pm.BetaBinomial('y', alpha=μ[idx]*κ[idx], beta=(1.0-μ[idx])*κ[idx],
                            n=data['sum'], observed=data[self.channel])
            idata = pm.sample(
                draws=self.samples,
                tune=self.tuning,
                chains=self.chains,
                nuts_sampler='numpyro',
                progressbar=sys.stdout.isatty(),
                compute_convergence_checks=False,
            )

        # Extract μ posterior samples: shape (chains, draws, n_proteins)
        mu_samples = idata.posterior['μ'].values
        n_chains, n_draws, n_proteins = mu_samples.shape
        # Flatten to (total_samples, n_proteins)
        mu_flat = mu_samples.reshape(n_chains * n_draws, n_proteins)

        # Build histogram over [0, 1] for each protein
        num_bins = int(1 / bin_width)
        hist = np.zeros((n_proteins, num_bins), dtype=int)
        for i in range(n_proteins):
            vals = mu_flat[:, i]
            inds = np.floor(vals / bin_width).astype(int)
            inds[vals == 1.0] -= 1
            invalid = np.where((inds < 0) | (inds >= num_bins))
            if invalid[0].size != 0:
                raise ValueError(
                    f'μ sample out of [0, 1] bounds: {vals[invalid[0][0]]}'
                )
            np.add.at(hist[i], inds, 1)

        return proteins, hist
