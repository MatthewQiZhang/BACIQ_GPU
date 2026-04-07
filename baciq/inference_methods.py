from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
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
        n_proteins = len(proteins)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        idx_t = torch.tensor(idx, dtype=torch.long, device=device)
        n_t = torch.tensor(data['sum'].values, dtype=torch.float32, device=device)
        obs_t = torch.tensor(
            data[self.channel].values, dtype=torch.float32, device=device)

        def model(n, obs):
            tau = pyro.sample('tau', dist.Gamma(
                torch.tensor(7.5, device=device),
                torch.tensor(1.0, device=device)))
            mu = pyro.sample('mu', dist.Uniform(
                torch.zeros(n_proteins, device=device),
                torch.ones(n_proteins, device=device)).to_event(1))
            kappa = pyro.sample('kappa', dist.Exponential(
                tau * torch.ones(n_proteins, device=device)).to_event(1))
            alpha = mu[idx_t] * kappa[idx_t]
            beta_v = (1.0 - mu[idx_t]) * kappa[idx_t]
            with pyro.plate('obs', len(n)):
                pyro.sample('y', dist.BetaBinomial(alpha, beta_v, n), obs=obs)

        # Run chains sequentially for Windows GPU compatibility
        all_mu = []
        for _ in range(self.chains):
            pyro.clear_param_store()
            nuts_kernel = NUTS(model)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=self.samples,
                warmup_steps=self.tuning,
                num_chains=1,
                disable_progbar=not sys.stdout.isatty(),
            )
            mcmc.run(n_t, obs_t)
            all_mu.append(mcmc.get_samples()['mu'].cpu().numpy())

        # shape: (chains * draws, n_proteins)
        mu_flat = np.concatenate(all_mu, axis=0)

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
