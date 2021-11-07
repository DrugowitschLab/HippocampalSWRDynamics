import numpy as np
import scipy.stats as sp
from scipy.ndimage import gaussian_filter1d

import replay_structure.utils as utils
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.config import HighSynchronyEvents_Preprocessing_Parameters


class HighSynchronyEvents_Preprocessing:
    """
    Extracts neural activity within high-synchrony events (HSEs).
    """

    def __init__(
        self,
        ratday: RatDay_Preprocessing,
        params: HighSynchronyEvents_Preprocessing_Parameters,
    ):
        self.params = params
        self.pf_matrix = utils.get_pf_matrix(
            ratday.place_field_data["place_fields"],
            ratday.place_field_data["place_cell_ids"],
        )
        print("Getting ripple spikemats")
        self.highsynchronyevent_info = self.get_highsynchronyevents(ratday)
        self.spikemat_info = self.get_spikemat_info(ratday)

    def get_highsynchronyevents(self, ratday: RatDay_Preprocessing) -> dict:
        hse_info = dict()
        (
            hse_info["spike_hist"],
            hse_info["spike_hist_smoothed"],
            hse_info["spike_hist_times_s"],
            hse_info["mean_spikes"],
            hse_info["sd_spikes"],
            hse_info["threshold_spikes"],
        ) = self.get_spike_hist(ratday)
        hse_info["hse_times_s"] = self.get_hse_times_s(ratday, hse_info)
        return hse_info

    def get_spike_hist(self, ratday: RatDay_Preprocessing) -> tuple:
        spike_times_ms = ratday.data["spike_times_s"] * 1000
        start_time = np.floor(spike_times_ms.min())
        end_time = np.ceil(spike_times_ms.max())
        n_bins = end_time - start_time
        bins = np.linspace(start_time, end_time, n_bins.astype(int) + 1)
        spike_hist = np.histogram(ratday.data["spike_times_s"] * 1000, bins=bins)
        spike_hist_not_smoothed = spike_hist[0]
        spike_hist_smoothed = gaussian_filter1d(spike_hist[0].astype(float), 10)
        spike_hist_times_ms = (spike_hist[1][1:] + spike_hist[1][:-1]) / 2
        spike_hist_times_s = spike_hist_times_ms / 1000
        # get_hse_selection criteria
        mean = np.mean(spike_hist_smoothed)
        sd = np.std(spike_hist_smoothed)
        threshold = mean + 3 * sd
        return (
            spike_hist_not_smoothed,
            spike_hist_smoothed,
            spike_hist_times_s,
            mean,
            sd,
            threshold,
        )

    def get_hse_times_s(self, ratday: RatDay_Preprocessing, hse_info: dict):
        # get all hse times
        hse_times = np.empty(shape=(0, 2))

        i = 0
        while i < len(hse_info["spike_hist_smoothed"]):
            if hse_info["spike_hist_smoothed"][i] > hse_info["threshold_spikes"]:
                # get start and end time
                hse_start_ind_bool = (
                    hse_info["spike_hist_smoothed"][:i] < hse_info["mean_spikes"]
                )
                if hse_start_ind_bool.sum() > 0:
                    hse_start_ind = np.argwhere(hse_start_ind_bool)[-1][0]
                else:
                    hse_start_ind = 0
                hse_end_ind = (
                    np.argwhere(
                        hse_info["spike_hist_smoothed"][i:] < hse_info["mean_spikes"]
                    )[0][0]
                    + i
                )
                # get end time
                hse_times = np.vstack(
                    (
                        hse_times,
                        [
                            hse_info["spike_hist_times_s"][hse_start_ind],
                            hse_info["spike_hist_times_s"][hse_end_ind],
                        ],
                    )
                )
                i = hse_end_ind
            else:
                i += 1

        # select hse times within rest periods (exclude hse during running)
        rest_starts = ratday.velocity_info["run_ends"][:-1]
        rest_ends = ratday.velocity_info["run_starts"][1:]
        hse_times_within_rest = np.empty(shape=(0, 2))

        for i, (hse_start, hse_end) in enumerate(hse_times):
            if (hse_start > rest_starts).sum() > 0:
                potential_rest_ind = np.argwhere(hse_start > rest_starts)[-1][0]
            else:
                continue
            hse_start_within_rest = hse_start < rest_ends[potential_rest_ind]
            if (hse_end > rest_starts).sum() > 0:
                potential_rest_ind = np.argwhere(hse_end > rest_starts)[-1][0]
            else:
                continue
            # potential_rest_ind = np.argwhere(hse_end > rest_starts)[-1][0]
            hse_end_within_rest = hse_end < rest_ends[potential_rest_ind]
            if hse_start_within_rest & hse_end_within_rest:
                hse_times_within_rest = np.vstack(
                    (hse_times_within_rest, [hse_start, hse_end])
                )
            elif hse_start_within_rest & ~hse_end_within_rest:
                hse_times_within_rest = np.vstack(
                    (hse_times_within_rest, [hse_start, rest_ends[potential_rest_ind]])
                )
            elif ~hse_start_within_rest & hse_end_within_rest:
                hse_times_within_rest = np.vstack(
                    (hse_times_within_rest, [rest_starts[potential_rest_ind], hse_end])
                )
            else:
                pass

        hse_lengths = hse_times_within_rest[:, 1] - hse_times_within_rest[:, 0]
        hse_times_within_rest = hse_times_within_rest[
            hse_lengths > self.params.min_hse_duration_s
        ]
        return hse_times_within_rest

    def get_spikemat_info(self, ratday: RatDay_Preprocessing) -> dict:
        spikemat_info = dict()
        spike_ids = ratday.data["spike_ids"]
        (
            spikemat_info["spikemats_full"],
            spikemat_info["spikemats_popburst"],
            spikemat_info["popburst_times_s"],
            spikemat_info["avg_spikes_per_s_smoothed"],
        ) = self.get_spikemats(
            spike_ids,
            ratday.data["spike_times_s"],
            ratday.place_field_data["place_cell_ids"],
            self.highsynchronyevent_info["hse_times_s"],
        )

        spikemat_info[
            "popburst_mean_firing_rate_array"
        ] = self.calc_popburst_firing_rate_array(spikemat_info["spikemats_popburst"])
        spikemat_info["firing_rate_scaling"] = self.calc_firing_rate_scaling(
            ratday.place_field_data["mean_firing_rate_array"][
                ratday.place_field_data["place_cell_ids"]
            ],
            spikemat_info["popburst_mean_firing_rate_array"],
        )
        return spikemat_info

    def calc_firing_rate_scaling(
        self, run_mean_frs: np.ndarray, ripple_mean_frs: np.ndarray
    ) -> dict:
        scaling_factors = ripple_mean_frs / run_mean_frs
        scaling_factors = scaling_factors[scaling_factors > 0]
        k, _, scale = sp.gamma.fit(scaling_factors, floc=0)
        return {"alpha": k, "beta": 1 / scale}

    def get_spikemats(
        self,
        spike_ids: np.ndarray,
        spike_times: np.ndarray,
        place_cell_ids: np.ndarray,
        hse_times_s: np.ndarray,
    ) -> tuple:
        """Get spikemat for full ripple and then for population burst only"""
        n_highfrequencyevents = np.shape(hse_times_s)[0]
        spikemats_full = dict()
        spikemats_popburst = dict()
        spikemats_popburst_times = np.zeros((n_highfrequencyevents, 2))
        avg_spikes_per_s_smoothed = dict()
        for hse_num in range(n_highfrequencyevents):
            hse_start = hse_times_s[hse_num][0]
            hse_end = hse_times_s[hse_num][1]
            spikemats_full[hse_num] = utils.get_spikemat(
                spike_ids,
                spike_times,
                place_cell_ids,
                hse_start,
                hse_end,
                self.params.time_window_s,
                self.params.time_window_advance_s,
            )
            if self.params.select_population_burst:
                (
                    spikemats_popburst[hse_num],
                    spikemats_popburst_times[hse_num],
                    avg_spikes_per_s_smoothed[hse_num],
                ) = self.select_population_burst(
                    spikemats_full[hse_num], hse_start, hse_end
                )
            else:
                (
                    spikemats_popburst[hse_num],
                    spikemats_popburst_times[hse_num],
                    avg_spikes_per_s_smoothed[hse_num],
                ) = (None, None, None)
        return (
            spikemats_full,
            spikemats_popburst,
            spikemats_popburst_times,
            avg_spikes_per_s_smoothed,
        )

    def select_population_burst(
        self, spikemat_fullripple: np.ndarray, ripple_start: float, ripple_end: float
    ) -> tuple:
        n_place_cells = spikemat_fullripple.shape[1]
        spikes_per_timebin = spikemat_fullripple.sum(axis=1)
        avg_spikes_per_s = (
            spikes_per_timebin / n_place_cells / self.params.time_window_s
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
        n_place_cells = spikemats[0].shape[1]
        total_spikes = np.zeros(n_place_cells)
        total_time = 0
        for i in range(len(spikemats)):
            if spikemats[i] is not None:
                total_spikes += spikemats[i].sum(axis=0)
                total_time += spikemats[i].shape[0] * self.params.time_window_advance_s
        firing_rate_array = total_spikes / total_time
        return firing_rate_array
