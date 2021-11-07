import numpy as np
import scipy.stats as sp
from typing import Optional

import replay_structure.utils as utils
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.config import Ripple_Preprocessing_Parameters


class Ripple_Preprocessing:
    """
    Extracts neural activity within SWRs.
    """

    def __init__(
        self,
        ratday: RatDay_Preprocessing,
        params: Ripple_Preprocessing_Parameters,
        popburst_times_s: Optional[np.ndarray] = None,
    ):
        self.params = params
        self.data = self.select_relavent_data(ratday)
        self.pf_matrix = utils.get_pf_matrix(
            ratday.place_field_data["place_fields"],
            ratday.place_field_data["place_cell_ids"],
        )
        print("Getting ripple spikemats")
        self.ripple_info = self.get_ripple_info(
            ratday, popburst_times_s=popburst_times_s
        )

    @staticmethod
    def select_relavent_data(ratday: RatDay_Preprocessing) -> dict:
        data = dict()
        data["ripple_times_s"] = ratday.data["ripple_times_s"]
        data["n_ripples"] = ratday.data["n_ripples"]
        data["n_place_cells"] = ratday.place_field_data["n_place_cells"]
        return data

    def get_ripple_info(
        self,
        ratday: RatDay_Preprocessing,
        popburst_times_s: Optional[np.ndarray] = None,
    ) -> dict:
        ripple_info = dict()
        spike_ids = ratday.data["spike_ids"]
        print(spike_ids)
        if self.params.shuffle_placefieldIDs:
            np.random.seed(0)
            np.random.shuffle(spike_ids)
        print(spike_ids)
        (
            ripple_info["spikemats_fullripple"],
            ripple_info["spikemats_popburst"],
            ripple_info["popburst_times_s"],
            ripple_info["avg_spikes_per_s_smoothed"],
        ) = self.get_spikemats(
            spike_ids,
            ratday.data["spike_times_s"],
            ratday.place_field_data["place_cell_ids"],
            ratday.data["ripple_times_s"],
            popburst_times_s=popburst_times_s,
        )

        (
            ripple_info["popburst_mean_firing_rate_array"],
            ripple_info["popburst_mean_firing_rate_matrix"],
        ) = self.calc_popburst_firing_rate_array(ripple_info["spikemats_popburst"])
        ripple_info["firing_rate_scaling"] = self.calc_firing_rate_scaling(
            ratday.place_field_data["mean_firing_rate_array"][
                ratday.place_field_data["place_cell_ids"]
            ],
            ripple_info["popburst_mean_firing_rate_array"],
        )
        return ripple_info

    def calc_firing_rate_scaling(
        self, run_mean_frs: np.ndarray, ripple_mean_frs: np.ndarray
    ) -> dict:
        scaling_factors = ripple_mean_frs / run_mean_frs
        scaling_factors = scaling_factors[scaling_factors > 0]
        k, _, scale = sp.gamma.fit(scaling_factors, floc=0)
        print(k, scale)
        return {"scaling_factors": scaling_factors, "alpha": k, "beta": 1 / scale}

    def get_spikemats(
        self,
        spike_ids: np.ndarray,
        spike_times: np.ndarray,
        place_cell_ids: np.ndarray,
        ripple_times: np.ndarray,
        popburst_times_s: Optional[np.ndarray] = None,
    ) -> tuple:
        """Get spikemat for full ripple and then for population burst only"""
        spikemats_fullripple = dict()
        spikemats_popburst = dict()
        spikemat_times = np.zeros((self.data["n_ripples"], 2))
        avg_spikes_per_s_smoothed = dict()
        for ripple_num in range(self.data["n_ripples"]):
            ripple_start = ripple_times[ripple_num][0]
            ripple_end = ripple_times[ripple_num][1]
            spikemats_fullripple[ripple_num] = utils.get_spikemat(
                spike_ids,
                spike_times,
                place_cell_ids,
                ripple_start,
                ripple_end,
                self.params.time_window_s,
                self.params.time_window_advance_s,
            )
            # if self.params.select_population_burst:
            if popburst_times_s is None:
                (
                    spikemats_popburst[ripple_num],
                    spikemat_times[ripple_num],
                    avg_spikes_per_s_smoothed[ripple_num],
                ) = self.select_population_burst(
                    spikemats_fullripple[ripple_num], ripple_start, ripple_end
                )
            else:

                spikemats_popburst[ripple_num] = utils.get_spikemat(
                    spike_ids,
                    spike_times,
                    place_cell_ids,
                    popburst_times_s[ripple_num][0],
                    popburst_times_s[ripple_num][1],
                    self.params.time_window_s,
                    self.params.time_window_advance_s,
                )
                spikemat_times[ripple_num] = popburst_times_s[ripple_num]
                avg_spikes_per_s_smoothed[ripple_num] = None
        return (
            spikemats_fullripple,
            spikemats_popburst,
            spikemat_times,
            avg_spikes_per_s_smoothed,
        )

    def select_population_burst(
        self, spikemat_fullripple: np.ndarray, ripple_start: float, ripple_end: float
    ) -> tuple:
        spikes_per_timebin = spikemat_fullripple.sum(axis=1)
        avg_spikes_per_s = (
            spikes_per_timebin / self.data["n_place_cells"] / self.params.time_window_s
        )
        avg_spikes_per_s_smoothed = np.convolve(
            avg_spikes_per_s, self.params.avg_fr_smoothing_convolution, mode="same"
        )
        timebins_above_threshold = (
            avg_spikes_per_s_smoothed > self.params.popburst_avg_spikes_per_s_threshold
        )
        if (timebins_above_threshold).sum() > 1:
            start_timebin = np.argwhere(timebins_above_threshold)[0][0]
            end_timebin = np.argwhere(timebins_above_threshold)[-1][0]
            if (end_timebin - start_timebin) >= self.params.min_popburst_n_time_windows:
                spikemat_popburst = spikemat_fullripple[start_timebin:end_timebin]
                spikemat_popburst_start = (
                    ripple_start + start_timebin * self.params.time_window_s
                )
                spikemat_popburst_end = (
                    ripple_end
                    - (spikemat_fullripple.shape[0] - end_timebin)
                    * self.params.time_window_s
                )
            else:
                spikemat_popburst = None
                spikemat_popburst_start, spikemat_popburst_end = [np.nan, np.nan]
        else:
            spikemat_popburst = None
            spikemat_popburst_start, spikemat_popburst_end = [np.nan, np.nan]
        return (
            spikemat_popburst,
            [spikemat_popburst_start, spikemat_popburst_end],
            avg_spikes_per_s_smoothed,
        )

    def calc_popburst_firing_rate_array(self, spikemats: dict) -> np.ndarray:
        firing_rate_matrix = np.full(
            (self.data["n_place_cells"], len(spikemats)), np.nan
        )
        total_spikes = np.zeros(self.data["n_place_cells"])
        total_time = 0
        for i in range(len(spikemats)):
            if spikemats[i] is not None:
                total_spikes += spikemats[i].sum(axis=0)
                total_time += spikemats[i].shape[0] * self.params.time_window_s
                firing_rate_matrix[:, i] = total_spikes / total_time
        firing_rate_array = total_spikes / total_time
        return firing_rate_array, firing_rate_matrix
