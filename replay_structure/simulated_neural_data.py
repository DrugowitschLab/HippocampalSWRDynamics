import numpy as np
import scipy.stats as sp
from typing import List, Dict, Optional

from replay_structure.metadata import Neg_Binomial_Params, Poisson_Params

from replay_structure.simulated_trajectories import Simulated_Trajectory
from replay_structure.metadata import Likelihood_Function_Params


class Simulated_Spikes_Parameters:
    """Defines parameters for generating neural spikes from a simulated spatial
    trajectory.
    """

    def __init__(
        self,
        pf_matrix: np.ndarray,
        time_window_ms: int,
        likelihood_function_params: Likelihood_Function_Params,
        time_window_advance_ms: Optional[int] = None,
        bin_size_cm: int = 4,
        random_seed=1,
    ):
        self.pf_matrix = pf_matrix
        self.random_seed = random_seed
        self.n_cells = pf_matrix.shape[0]
        self.n_bins_x = np.sqrt(pf_matrix.shape[1]).astype(int)
        self.n_bins_y = np.sqrt(pf_matrix.shape[1]).astype(int)
        self.bin_size_cm = bin_size_cm
        self.time_window_ms = time_window_ms
        self.time_window_s = time_window_ms / 1000
        self.time_window_advance_ms = time_window_advance_ms
        self.time_window_advance_s = (
            time_window_advance_ms / 1000
            if time_window_advance_ms is not None
            else None
        )
        self.likelihood_function_params = likelihood_function_params


class Simulated_Data_Preprocessing:
    """Generates simulated neural spikes from a simulated spatial trajectory.
    """

    def __init__(
        self,
        trajectory_set: List[Simulated_Trajectory],
        simulated_spikes_params: Simulated_Spikes_Parameters,
    ):
        self.trajectory_set = trajectory_set
        self.params = simulated_spikes_params
        self.trajectory_spikes, self.generative_fr = self.generate_trajectory_spikes(
            self.trajectory_set, self.params.pf_matrix
        )
        if self.params.time_window_advance_ms is None:
            self.spikemats = self.bin_spikes_nonoverlapping(self.trajectory_spikes)
        else:
            self.spikemats = self.bin_spikes_overlapping(self.trajectory_spikes)

    def generate_trajectory_spikes(
        self, trajectories: List[Simulated_Trajectory], pf_matrix: np.ndarray
    ) -> tuple:
        np.random.seed(self.params.random_seed)
        trajectory_spikes: Dict[int, np.ndarray] = dict()
        trajectory_generative_fr_ind: Dict[int, np.ndarray] = dict()
        for i, trajectory in enumerate(trajectories):
            spikes, generative_fr_ind = self.generate_spikes(trajectory, pf_matrix)
            trajectory_spikes[i] = spikes
            trajectory_generative_fr_ind[i] = generative_fr_ind
        return trajectory_spikes, trajectory_generative_fr_ind

    def generate_spikes(
        self, trajectory: Simulated_Trajectory, pf_matrix: np.ndarray
    ) -> np.ndarray:
        n_timesteps = np.shape(trajectory.trajectory_cm)[0]
        # multiple by ms time window
        if isinstance(self.params.likelihood_function_params, Neg_Binomial_Params):
            fr_scaling_factors = sp.gamma.rvs(
                a=self.params.likelihood_function_params.alpha,
                loc=0,
                scale=1 / self.params.likelihood_function_params.beta,
                size=self.params.n_cells,
            )
        spikes = np.zeros((n_timesteps, self.params.n_cells))
        generative_fr_ind = np.zeros((n_timesteps, self.params.n_cells))
        for t, (x, y) in enumerate(trajectory.trajectory_cm):
            # extract firing rates for this
            x = np.floor(
                x
                * (self.params.n_bins_x / trajectory.trajectory_params.arena_length_cm)
            )
            y = np.floor(
                y
                * (self.params.n_bins_y / trajectory.trajectory_params.arena_length_cm)
            )
            ind = (x * self.params.n_bins_x) + y
            ind = int(np.floor(ind))
            if ind < 0:
                print("invalid trajectory location!")
            if isinstance(self.params.likelihood_function_params, Poisson_Params):
                fr = (
                    pf_matrix[:, ind]
                    * trajectory.trajectory_params.time_window_s
                    * self.params.likelihood_function_params.rate_scaling
                )
            elif isinstance(
                self.params.likelihood_function_params, Neg_Binomial_Params
            ):
                fr = (
                    pf_matrix[:, ind]
                    * trajectory.trajectory_params.time_window_s
                    * fr_scaling_factors
                )
            # generate spikes according to poisson distribution
            spikes[t] = np.random.poisson(fr, size=self.params.n_cells)
            generative_fr_ind[t] = ind  # (
        return spikes, generative_fr_ind

    def bin_spikes_nonoverlapping(
        self, trajecory_spikes: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        spikemats: Dict[int, np.ndarray] = dict()
        for i in range(len(trajecory_spikes)):
            n_timesteps = np.shape(trajecory_spikes[i])[0]
            timebin = self.params.time_window_ms
            n_timesteps_bin = int(np.floor(n_timesteps / timebin))
            spikemat = np.zeros((n_timesteps_bin, self.params.n_cells))
            for t in range(n_timesteps_bin):
                start = t * timebin
                end = start + timebin
                spikemat[t] = trajecory_spikes[i][start:end].sum(axis=0)
            spikemats[i] = spikemat.astype(np.int8)
        return spikemats

    def bin_spikes_overlapping(
        self, trajecory_spikes: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        assert self.params.time_window_advance_ms is not None
        spikemats: Dict[int, np.ndarray] = dict()
        for i in range(len(trajecory_spikes)):
            n_timesteps = np.shape(trajecory_spikes[i])[0]
            timebin = self.params.time_window_ms
            timebin_advance = self.params.time_window_advance_ms
            n_timesteps_bin = int(np.floor((n_timesteps - timebin) / timebin_advance))
            spikemat = np.zeros((n_timesteps_bin, self.params.n_cells))
            for t in range(n_timesteps_bin):
                start = t * timebin_advance
                end = start + timebin
                spikemat[t] = trajecory_spikes[i][start:end].sum(axis=0)
            spikemats[i] = spikemat.astype(np.int8)
        return spikemats
