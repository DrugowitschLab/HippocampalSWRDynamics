from replay_structure.metadata import Neg_Binomial_Params, Poisson_Params
import numpy as np
import scipy.stats as sp
import torch
from typing import Optional
from scipy.special import logsumexp

import replay_structure.utils as utils
import replay_structure.forward_backward as fb
from replay_structure.structure_analysis_input import Structure_Analysis_Input


class Structure_Model:
    """Base class implementation of structure model. This class defines the
    methods that are common across all models (getting the model evidence ond marginals
    across all SWRs in a session). The model-specific calculation of the model
    evidence is defined in each model child class."""

    def __init__(self, structure_data: Structure_Analysis_Input):
        self.structure_data = structure_data
        # get effective time_window_s if using scaled Poisson likelihood
        if isinstance(
            self.structure_data.params.likelihood_function_params, Poisson_Params
        ):
            if (
                self.structure_data.params.likelihood_function_params.rate_scaling
                is not None
            ):
                self.emission_prob_time_window = (
                    self.structure_data.params.time_window_s
                    * self.structure_data.params.likelihood_function_params.rate_scaling
                )
            else:
                self.emission_prob_time_window = (
                    self.structure_data.params.time_window_s
                )

    def get_model_evidences(self) -> np.ndarray:
        model_evidence = np.zeros(len(self.structure_data.spikemats))
        for spikemat_ind in range(len(self.structure_data.spikemats)):
            model_evidence[spikemat_ind] = self.get_spikemat_model_evidence(
                spikemat_ind
            )
        return model_evidence

    def get_spikemat_model_evidence(self, spikemat_ind: int) -> float:
        if self.structure_data.spikemats[spikemat_ind] is not None:
            model_evidence, _ = self._calc_model_evidence(spikemat_ind)
        else:
            model_evidence = np.nan
        return model_evidence

    def get_marginals(self) -> dict:
        marginals = dict()
        for spikemat_ind in range(len(self.structure_data.spikemats)):
            marginals[spikemat_ind] = self.get_spikemat_marginals(spikemat_ind)
        return marginals

    def get_spikemat_marginals(self, spikemat_ind: int) -> np.ndarray:
        if self.structure_data.spikemats[spikemat_ind] is not None:
            _, marginals = self._calc_model_evidence(spikemat_ind)
        else:
            marginals = np.nan
        return marginals

    def _calc_model_evidence(self, spikemat_ind: int):
        """Implemented in child classes."""
        pass

    def _calc_emission_probabilities(self, spikemat_ind: int) -> np.ndarray:
        if isinstance(
            self.structure_data.params.likelihood_function_params, Neg_Binomial_Params
        ):
            emission_probabilities = utils.calc_neg_binomial_emission_probabilities(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.structure_data.params.time_window_s,
                self.structure_data.params.likelihood_function_params.alpha,
                self.structure_data.params.likelihood_function_params.beta,
            )
        elif isinstance(
            self.structure_data.params.likelihood_function_params, Poisson_Params
        ):
            emission_probabilities = utils.calc_poisson_emission_probabilities(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.emission_prob_time_window,
            )
        else:
            raise Exception("Invalid likelihood function")
        return emission_probabilities

    def _calc_emission_probabilities_log(self, spikemat_ind: int) -> np.ndarray:
        if isinstance(
            self.structure_data.params.likelihood_function_params, Neg_Binomial_Params
        ):
            emission_probabilities_log = utils.calc_neg_binomial_emission_probabilities_log(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.structure_data.params.time_window_s,
                self.structure_data.params.likelihood_function_params.alpha,
                self.structure_data.params.likelihood_function_params.beta,
            )
        elif isinstance(
            self.structure_data.params.likelihood_function_params, Poisson_Params
        ):
            emission_probabilities_log = utils.calc_poisson_emission_probabilities_log(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.emission_prob_time_window,
            )
        else:
            raise Exception("Invalid likelihood function")
        return emission_probabilities_log

    def _calc_emission_probability_log(self, spikemat_ind: int) -> np.ndarray:
        if isinstance(
            self.structure_data.params.likelihood_function_params, Neg_Binomial_Params
        ):
            emission_probability_log = utils.calc_neg_binomial_emission_probability_log(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.structure_data.params.time_window_s,
                self.structure_data.params.likelihood_function_params.alpha,
                self.structure_data.params.likelihood_function_params.beta,
            )
        elif isinstance(
            self.structure_data.params.likelihood_function_params, Poisson_Params
        ):
            emission_probability_log = utils.calc_poisson_emission_probability_log(
                self.structure_data.spikemats[spikemat_ind],
                self.structure_data.pf_matrix,
                self.emission_prob_time_window,
            )
        else:
            raise Exception("Invalid likelihood function")
        return emission_probability_log


class Diffusion(Structure_Model):
    def __init__(self, structure_data: Structure_Analysis_Input, sd_meters: float):
        super().__init__(structure_data)
        self.sd_meters = sd_meters
        self.sd_bins = utils.meters_to_bins(
            sd_meters, bin_size_cm=structure_data.params.bin_size_cm
        )
        self.forward_backward_input = self._initialize_forward_backward_input()

    def _calc_model_evidence(self, spikemat_ind: int):
        self.forward_backward_input[
            "emission_probabilities"
        ] = self._calc_emission_probabilities(spikemat_ind)
        forward_backward_output = fb.Forward_Backward_xy(
            self.forward_backward_input
        ).run_forward_backward_algorithm("no joints")
        model_ev = forward_backward_output["data_likelihood"]
        marginals = forward_backward_output["latent_marginals"].T
        return model_ev, marginals

    def _initialize_forward_backward_input(self) -> dict:
        forward_backward_input = dict()
        n_grid = (
            self.structure_data.params.n_bins_x * self.structure_data.params.n_bins_y
        )
        forward_backward_input["initial_state_prior"] = np.ones(n_grid) / n_grid
        forward_backward_input["transition_matrix"] = self._calc_transition_matrix(
            self.sd_bins
        )
        return forward_backward_input

    def _calc_transition_matrix(self, sd_bins: float) -> np.ndarray:
        """NxN matrix"""
        transition_mat = np.zeros(
            (self.structure_data.params.n_bins_x, self.structure_data.params.n_bins_y)
        )
        j = np.arange(self.structure_data.params.n_bins_y)  # t
        for i in range(self.structure_data.params.n_bins_x):  # t-1
            this_transition = np.exp(
                -((j - i) ** 2)
                / (2 * sd_bins ** 2 * self.structure_data.params.time_window_s)
            )
            transition_mat[:, i] = this_transition / np.sum(this_transition)
        return transition_mat


class Momentum(Structure_Model):
    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        sd_0_meters: float,
        sd_meters: float,
        decay: float,
        emission_probabilities: Optional[np.ndarray] = None,
        plotting: bool = False,
        plotting_folder: Optional[str] = None,
    ):
        super().__init__(structure_data)
        self.sd_0_meters = sd_0_meters
        self.sd_meters = sd_meters
        self.sd_0_bins = utils.meters_to_bins(
            sd_0_meters, bin_size_cm=structure_data.params.bin_size_cm
        )
        self.sd_bins = utils.meters_to_bins(
            sd_meters, bin_size_cm=structure_data.params.bin_size_cm
        )
        self.decay = decay
        self.emission_probabilities = emission_probabilities
        self.forward_backward_input = self._initialize_forward_backward_input()
        self.plotting = plotting
        self.plotting_folder = plotting_folder

    def _calc_model_evidence(self, spikemat_ind: int):
        self.forward_backward_input[
            "emission_probabilities"
        ] = self.get_emission_probabilities(spikemat_ind)
        forward_backward_output = fb.Forward_Backward_order2(
            self.forward_backward_input
        ).run_forward_backward_algorithm(
            plotting=self.plotting, plotting_folder=self.plotting_folder
        )
        model_ev = forward_backward_output["data_likelihood"]  # .numpy()
        if self.plotting:
            marginals = forward_backward_output["latent_marginals"].numpy().T
        else:
            marginals = forward_backward_output["alphas"].numpy().T
        return model_ev, marginals

    def _initialize_forward_backward_input(self) -> dict:
        forward_backward_input = dict()
        forward_backward_input["initial_state_prior"] = torch.from_numpy(
            np.ones(self.structure_data.params.n_grid)
            / self.structure_data.params.n_grid
        )
        forward_backward_input["initial_transition"] = torch.from_numpy(
            self._calc_order1_transition_matrix(self.sd_0_bins)
        )
        forward_backward_input["transition_matrix"] = torch.from_numpy(
            self._calc_order2_transition_matrix(self.sd_bins, self.decay)
        )
        return forward_backward_input

    def _calc_order1_transition_matrix(self, sd: float):
        """(NxN)x(NxN) matrix"""
        initial_transition = np.zeros(
            (self.structure_data.params.n_grid, self.structure_data.params.n_grid)
        )
        m = np.arange(self.structure_data.params.n_bins_x)
        n = np.arange(self.structure_data.params.n_bins_y)
        mm, nn = np.meshgrid(m, n)  # t x,y
        for i in range(self.structure_data.params.n_bins_x):  # t-1 x
            for j in range(self.structure_data.params.n_bins_y):  # t-1 y
                this_transition = np.exp(
                    -((nn - i) ** 2 + (mm - j) ** 2)
                    / (2 * sd ** 2 * self.structure_data.params.time_window_s)
                )
                flat_transition = this_transition.reshape(-1)
                initial_transition[
                    :, i * self.structure_data.params.n_bins_x + j
                ] = flat_transition / np.sum(flat_transition)
        return initial_transition

    def _calc_order2_transition_matrix(self, sd: float, decay: float):
        """(n x n x n) matrix"""
        var_scaled = (
            (sd ** 2 * self.structure_data.params.time_window_s ** 2)
            / (2 * decay)
            * (1 - np.exp(-2 * decay * self.structure_data.params.time_window_s))
        )
        transition_mat = np.zeros(
            (
                self.structure_data.params.n_bins_x,
                self.structure_data.params.n_bins_x,
                self.structure_data.params.n_bins_x,
            )
        )
        m = np.arange(self.structure_data.params.n_bins_x)  # t x/y
        for i in range(self.structure_data.params.n_bins_x):  # t-2 x/y
            for k in range(self.structure_data.params.n_bins_x):  # t-1 x/y
                mean = (
                    1 + np.exp(-self.structure_data.params.time_window_s * decay)
                ) * k - (np.exp(-self.structure_data.params.time_window_s * decay)) * i
                this_transition = np.exp(-((m - mean) ** 2) / (2 * var_scaled))
                norm_sum = np.sum(this_transition)
                if norm_sum == 0:
                    max_prob_ind = (
                        0 if mean < 0 else self.structure_data.params.n_bins_x - 1
                    )
                    this_transition[max_prob_ind] = 1
                    transition_mat[:, k, i] = this_transition
                else:
                    transition_mat[:, k, i] = this_transition / norm_sum
        return transition_mat

    def get_emission_probabilities(self, spikemat_ind: int):
        if self.emission_probabilities is None:
            emission_probabilities = torch.from_numpy(
                self._calc_emission_probabilities(spikemat_ind)
            )
        else:
            emission_probabilities = torch.from_numpy(self.emission_probabilities)
        return emission_probabilities


class Stationary(Structure_Model):
    def __init__(self, structure_data: Structure_Analysis_Input):
        super().__init__(structure_data)

    def _calc_model_evidence(self, spikemat_ind: int):
        emission_probability_log = self._calc_emission_probability_log(spikemat_ind)
        joint_probability = emission_probability_log - np.log(
            self.structure_data.params.n_grid
        )
        model_evidence = logsumexp(joint_probability)
        marginals = np.exp(emission_probability_log - emission_probability_log.max())
        return model_evidence, marginals


class Stationary_Gaussian(Structure_Model):
    def __init__(self, structure_data: Structure_Analysis_Input, sd_meters: float):
        super().__init__(structure_data)
        self.sd_meters = sd_meters
        self.sd_bins = utils.meters_to_bins(
            sd_meters, bin_size_cm=structure_data.params.bin_size_cm
        )
        self.latent_probabilities_normalized = self._calc_latent_probabilities()

    def _calc_model_evidence(self, spikemat_ind: int):
        emission_probabilities = self._calc_emission_probabilities(spikemat_ind)
        sum_z = np.matmul(
            emission_probabilities.T, self.latent_probabilities_normalized
        )
        sum_t = np.sum(np.log(sum_z), axis=0)
        model_evidence = logsumexp(-np.log(self.structure_data.params.n_grid) + sum_t)
        marginals = np.matmul(
            emission_probabilities.T, self.latent_probabilities_normalized.T
        ).T
        return model_evidence, marginals

    def _calc_latent_probabilities(self):
        # initialze matrix (Nz x Nu)
        latent_mat = np.zeros(
            (
                self.structure_data.params.n_bins_x,
                self.structure_data.params.n_bins_y,
                self.structure_data.params.n_bins_x,
                self.structure_data.params.n_bins_y,
            )
        )
        x = np.arange(self.structure_data.params.n_bins_x)
        y = np.arange(self.structure_data.params.n_bins_y)
        xx, yy = np.meshgrid(x, y)
        for m in range(self.structure_data.params.n_bins_x):
            for n in range(self.structure_data.params.n_bins_y):
                this_prob = sp.multivariate_normal(
                    [n, m], [[self.sd_bins ** 2, 0], [0, self.sd_bins ** 2]]
                ).pdf(np.transpose([xx, yy]))
                latent_mat[:, :, n, m] = this_prob / this_prob.sum()
        latent_mat = latent_mat.reshape(
            (self.structure_data.params.n_grid, self.structure_data.params.n_grid)
        )
        return latent_mat


class Random(Structure_Model):
    def __init__(self, structure_data: Structure_Analysis_Input):
        super().__init__(structure_data)

    def _calc_model_evidence(self, spikemat_ind: int):
        emission_probabilities_log = self._calc_emission_probabilities_log(spikemat_ind)
        full_sum = emission_probabilities_log - np.log(
            self.structure_data.params.n_grid
        )
        sum_z = logsumexp(full_sum, axis=0)
        model_evidence = np.sum(sum_z)
        marginals = np.exp(emission_probabilities_log)
        return model_evidence, marginals
