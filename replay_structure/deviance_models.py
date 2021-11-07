"""
This file contains the code for calculating the deviance explained of each dynamics
model.
"""

import numpy as np
import scipy.stats as sp
import pandas as pd
from typing import Optional

from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.model_comparison import Model_Comparison
from replay_structure.metadata import MODELS_AS_STR


class Deviance_Explained:
    """Calculates the deviance explained of each dynamics model for each
    SWR within a session.
    """

    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        model_comparison_results: Model_Comparison,
        structure_data_for_null: Optional[Structure_Analysis_Input] = None,
    ):
        self.saturated_model = Saturated(structure_data).run_all_spikemats()
        self.null_model = Null_AcrossNeurons_WithinSpikemats(
            structure_data, structure_data_for_null=structure_data_for_null
        ).run_all_spikemats()
        self.results = self.run_deviance_explained(model_comparison_results)

    def run_deviance_explained(self, model_comparison_results: Model_Comparison):

        deviance_results = pd.DataFrame()
        for model in MODELS_AS_STR:
            deviance_results[model] = self.calc_deviance_explained(
                model_comparison_results.results_dataframe[model]
            )
        return deviance_results

    def calc_deviance_explained(self, model_ev: np.ndarray) -> np.ndarray:
        deviance_model = 2 * (self.saturated_model - model_ev)
        deviance_null = 2 * (self.saturated_model - self.null_model)
        deviance_explained = 1 - (deviance_model / deviance_null)
        return deviance_explained


class Deviance_Model:
    """Base class for calculating the model evidence of the saturated and null models
    for each SWR within a session."""

    def __init__(self, structure_data: Structure_Analysis_Input):
        self.structure_data = structure_data

    def run_all_spikemats(self):
        n_spikemats = len(self.structure_data.spikemats)
        model_evidence = np.zeros(n_spikemats)
        for i in range(n_spikemats):
            model_evidence[i] = self.run_one_spikemat(i)
        return model_evidence

    def run_one_spikemat(self, spikemat_ind: int):
        if self.structure_data.spikemats[spikemat_ind] is not None:
            model_evidence = self._calc_model_evidence(spikemat_ind)
        else:
            model_evidence = np.nan
        return model_evidence

    def _calc_model_evidence(self, spikemat_ind: int):
        pass


class Saturated(Deviance_Model):
    """Calculates the model evidence of the saturated model for each SWR within a
    session. Inherits from Deviance Model."""

    def __init__(self, structure_data: Structure_Analysis_Input):
        super().__init__(structure_data)

    def _calc_model_evidence(self, spikemat_ind: int):
        spikemat = self.structure_data.spikemats[spikemat_ind]
        model_ev = np.sum(np.log(sp.poisson(spikemat).pmf(spikemat)))
        return model_ev


class Null_AcrossNeurons_WithinSpikemats(Deviance_Model):
    """Calculates the model evidence of the null model for each SWR within a
    session. Inherits from Deviance Model"""

    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        structure_data_for_null: Optional[Structure_Analysis_Input] = None,
    ):
        super().__init__(structure_data)
        if structure_data_for_null is None:
            self.spikemat_avg_fr = self.get_spikemat_avg_fr(structure_data)
        else:
            self.spikemat_avg_fr = self.get_spikemat_avg_fr(structure_data_for_null)

    @staticmethod
    def get_spikemat_avg_fr(structure_data: Structure_Analysis_Input):
        n_cells = structure_data.spikemats[0].shape[1]
        total_spikes = np.zeros(n_cells)
        total_time = 0
        for i in range(len(structure_data.spikemats)):
            if structure_data.spikemats[i] is not None:
                total_spikes += structure_data.spikemats[i].sum(axis=0)
                total_time += (
                    structure_data.spikemats[i].shape[0]
                    * structure_data.params.time_window_s
                )
        spikemat_avg_frs = total_spikes / total_time
        return spikemat_avg_frs.mean()

    def _calc_model_evidence(self, spikemat_ind: int):
        spikemat = self.structure_data.spikemats[spikemat_ind]
        model_ev = np.sum(
            np.log(
                sp.poisson(
                    self.spikemat_avg_fr * self.structure_data.params.time_window_s
                ).pmf(spikemat)
            )
        )
        return model_ev
