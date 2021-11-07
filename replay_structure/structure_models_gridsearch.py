#!/usr/bin/env python3
import numpy as np
import replay_structure.utils as utils
import replay_structure.structure_models as models
from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.config import Structure_Model_Gridsearch_Parameters
from replay_structure.metadata import Poisson_Params, Neg_Binomial_Params


class Structure_Gridsearch:
    """Base class for running a parameter gridsearch."""

    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        params: Structure_Model_Gridsearch_Parameters,
    ):
        self.gridsearch_params = params.params
        self.gridsearch_results = self.run_gridsearch(structure_data)

    def run_gridsearch(self, structure_data):
        pass


class Stationary_Gaussian(Structure_Gridsearch):
    """Stationary Guassian model gridsearch over the standard deviation parameter."""

    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        params: Structure_Model_Gridsearch_Parameters,
    ):
        super().__init__(structure_data, params)

    def run_gridsearch(self, structure_data: Structure_Analysis_Input) -> np.ndarray:
        gridsearch_results = np.zeros(
            (
                len(structure_data.spikemats),
                len(self.gridsearch_params["sd_array_meters"]),
            )
        )
        for i, sd_meters in enumerate(self.gridsearch_params["sd_array_meters"]):
            print("sd = {:.2}".format(sd_meters))
            model_evidence = models.Stationary_Gaussian(
                structure_data, sd_meters
            ).get_model_evidences()
            gridsearch_results[:, i] = model_evidence
        return gridsearch_results


class Diffusion(Structure_Gridsearch):
    """Diffusion model gridsearch over the standard deviation parameter."""

    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        params: Structure_Model_Gridsearch_Parameters,
    ):
        super().__init__(structure_data, params)

    def run_gridsearch(self, structure_data: Structure_Analysis_Input) -> np.ndarray:
        gridsearch_results = np.zeros(
            (
                len(structure_data.spikemats),
                len(self.gridsearch_params["sd_array_meters"]),
            )
        )
        for i, sd_meters in enumerate(self.gridsearch_params["sd_array_meters"]):
            print("sd = {:.2}".format(sd_meters))
            model_evidence = models.Diffusion(
                structure_data, sd_meters
            ).get_model_evidences()
            print(model_evidence[:5])
            gridsearch_results[:, i] = model_evidence
        return gridsearch_results


class Momentum(Structure_Gridsearch):
    """Momentum model gridsearch over the standard deviation and decay parameters."""

    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        params: Structure_Model_Gridsearch_Parameters,
        spikemat_ind: int,
        adjust_params: bool = True,
    ):
        self.spikemat_ind = spikemat_ind
        self.adjust_params = adjust_params
        params.params["sd_0_meters"] = (
            params.params["initial_sd_m_per_s"] * structure_data.params.time_window_s
        )
        super().__init__(structure_data, params)

    def run_gridsearch(self, structure_data: Structure_Analysis_Input) -> np.ndarray:
        if structure_data.spikemats[self.spikemat_ind] is None:
            gridsearch_results = (
                np.ones(
                    (
                        len(self.gridsearch_params["sd_array_meters"]),
                        len(self.gridsearch_params["decay_array"]),
                    )
                )
                * np.nan
            )
        else:
            emission_probabilities = self._pre_calc_emission_probablities(
                structure_data
            )
            gridsearch_results = np.zeros(
                (
                    len(self.gridsearch_params["sd_array_meters"]),
                    len(self.gridsearch_params["decay_array"]),
                )
            )
            sd_0_meters = self.gridsearch_params["sd_0_meters"]
            for i, sd_meters in enumerate(self.gridsearch_params["sd_array_meters"]):
                print("sd = {:.2}".format(sd_meters))
                for j, decay in enumerate(self.gridsearch_params["decay_array"]):
                    print("decay = {}".format(decay))

                    if self.adjust_params:
                        (decay, sd_meters) = utils.get_adjusted_parameters(
                            decay, sd_meters, structure_data.params.time_window_s
                        )
                        print(
                            "sd adjusted: {}, decay adjusted: {}".format(
                                sd_meters, decay
                            )
                        )
                    model_evidence = models.Momentum(
                        structure_data,
                        sd_0_meters,
                        sd_meters,
                        decay,
                        emission_probabilities=emission_probabilities,
                    ).get_spikemat_model_evidence(self.spikemat_ind)
                    gridsearch_results[i, j] = model_evidence
        return gridsearch_results

    def _pre_calc_emission_probablities(self, structure_data: Structure_Analysis_Input):
        if isinstance(
            structure_data.params.likelihood_function_params, Neg_Binomial_Params
        ):
            emission_probabilities = utils.calc_neg_binomial_emission_probabilities(
                structure_data.spikemats[self.spikemat_ind],
                structure_data.pf_matrix,
                structure_data.params.time_window_s,
                structure_data.params.likelihood_function_params.alpha,
                structure_data.params.likelihood_function_params.beta,
            )
        elif isinstance(
            structure_data.params.likelihood_function_params, Poisson_Params
        ):
            if (
                structure_data.params.likelihood_function_params.rate_scaling
                is not None
            ):
                emission_prob_time_window = (
                    structure_data.params.time_window_s
                    * structure_data.params.likelihood_function_params.rate_scaling
                )
            else:
                emission_prob_time_window = structure_data.params.time_window_s
            emission_probabilities = utils.calc_poisson_emission_probabilities(
                structure_data.spikemats[self.spikemat_ind],
                structure_data.pf_matrix,
                emission_prob_time_window,
            )
        else:
            raise Exception("Invalid likelihood function")
        return emission_probabilities
