import numpy as np

import replay_structure.utils as utils
from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.config import Structure_Analysis_Input_Parameters
from replay_structure.metadata import Poisson_Params, Neg_Binomial_Params
from replay_structure.viterbi import Viterbi


class Most_Likely_Trajectories:
    """Finds the most likely trajectory given a sequence of neural activity using the
    Viterbi algorithm."""

    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        sd_meters: float,
        run_all: bool = True,
    ):
        self.params: Structure_Analysis_Input_Parameters = structure_data.params
        self.sd_meters = sd_meters
        self.sd_bins = utils.meters_to_bins(
            sd_meters, bin_size_cm=structure_data.params.bin_size_cm
        )
        if isinstance(self.params.likelihood_function_params, Poisson_Params):
            if self.params.likelihood_function_params.rate_scaling is not None:
                self.emission_prob_time_window = (
                    self.params.time_window_s
                    * self.params.likelihood_function_params.rate_scaling
                )
            else:
                self.emission_prob_time_window = self.params.time_window_s
        self.viterbi_input = self._initialize_viterbi_input()
        if run_all:
            print("Getting most likely trajectories")
            self.most_likely_trajectories = self.run_all(structure_data)

    def _initialize_viterbi_input(self):
        viterbi_input = dict()
        viterbi_input["initial_state_prior"] = (
            np.ones(self.params.n_grid) / self.params.n_grid
        )
        return viterbi_input

    def run_all(self, structure_data: Structure_Analysis_Input):
        most_likely_trajectories = dict()
        for i in range(len(structure_data.spikemats)):
            print(i)
            most_likely_trajectories[i] = self.get_most_likely_trajectory(
                structure_data, i
            )
        return most_likely_trajectories

    def get_most_likely_trajectory(
        self, structure_data: Structure_Analysis_Input, spikemat_ind: int
    ) -> np.ndarray:
        if structure_data.spikemats[spikemat_ind] is not None:
            most_likely_trajectory_flattened = self._get_most_likely_trajectory(
                structure_data, spikemat_ind
            )
            most_likely_trajectory = np.array(
                [
                    most_likely_trajectory_flattened
                    % self.params.n_bins_x
                    * self.params.bin_size_cm,
                    most_likely_trajectory_flattened
                    // self.params.n_bins_x
                    * self.params.bin_size_cm,
                ]
            ).T
        else:
            most_likely_trajectory = None
        return most_likely_trajectory

    def _get_most_likely_trajectory(
        self, structure_data: Structure_Analysis_Input, spikemat_ind: int
    ) -> np.ndarray:
        if structure_data.spikemats[spikemat_ind] is not None:
            self.viterbi_input[
                "emission_probabilities"
            ] = self._calc_emission_probabilities(structure_data, spikemat_ind)
            self.viterbi_input["transition_matrix"] = self._calc_transition_matrix(
                self.sd_bins
            )
            viterbi_outputs = Viterbi(self.viterbi_input).run_viterbi_algorithm()
            most_likely_trajectory = viterbi_outputs["z_max"]
        else:
            most_likely_trajectory = np.array([np.nan, np.nan])
        return most_likely_trajectory

    def _calc_transition_matrix(self, sd_bins: float) -> np.ndarray:
        """(NxN)x(NxN) matrix"""
        transition_mat = np.zeros(
            (
                self.params.n_bins_x * self.params.n_bins_y,
                self.params.n_bins_x * self.params.n_bins_y,
            )
        )
        m = np.arange(self.params.n_bins_x)
        n = np.arange(self.params.n_bins_y)
        mm, nn = np.meshgrid(m, n)  # t x,y
        for i in range(self.params.n_bins_x):  # t-1 x
            for j in range(self.params.n_bins_y):  # t-1 y
                this_transition = np.exp(
                    -((nn - i) ** 2 + (mm - j) ** 2)
                    / (2 * sd_bins ** 2 * self.params.time_window_s)
                )
                flat_transition = this_transition.reshape(-1)
                transition_mat[
                    :, i * self.params.n_bins_x + j
                ] = flat_transition / np.sum(flat_transition)
        return transition_mat

    def _calc_emission_probabilities(
        self, structure_data: Structure_Analysis_Input, spikemat_ind: int
    ):
        if isinstance(self.params.likelihood_function_params, Neg_Binomial_Params):
            emission_probabilities = utils.calc_neg_binomial_emission_probabilities(
                structure_data.spikemats[spikemat_ind],
                structure_data.pf_matrix,
                self.params.time_window_s,
                self.params.likelihood_function_params.alpha,
                self.params.likelihood_function_params.beta,
            )
        elif isinstance(self.params.likelihood_function_params, Poisson_Params):
            emission_probabilities = utils.calc_poisson_emission_probabilities(
                structure_data.spikemats[spikemat_ind],
                structure_data.pf_matrix,
                self.emission_prob_time_window,
            )
        else:
            raise Exception("Invalid likelihood function")
        return emission_probabilities
