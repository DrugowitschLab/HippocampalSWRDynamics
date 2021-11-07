import numpy as np
from scipy.ndimage import gaussian_filter

from replay_structure.config import RatDay_Preprocessing_Parameters
import replay_structure.utils as utils


class RatDay_Preprocessing:
    """
    Preprocesses data as obtained from Pfeiffer & Foster (2013/15). The primary
    functionality is to:
    1. Reformat data from original MATLAB struct format.
    2. Clean the position and spiking recordings of recording gaps.
    2. Calculate velocity.
    3. Calculate place fields.
    """

    def __init__(self, matlab_data, params: RatDay_Preprocessing_Parameters) -> None:
        """
        Reformats and preprocesses the data from Pfeiffer & Foster (2015).
        """
        self.params = params
        print("Reformating data")
        self.raw_data = self.reformat_data(matlab_data)
        print("Cleaning data")
        self.data = self.clean_recording_data(self.raw_data)
        print("Calculating run periods")
        self.velocity_info = self.calculate_velocity_info()
        print("Calculating place fields")
        np.random.seed(0)
        self.place_field_data = self.calculate_place_fields()
        print("DONE")

    @staticmethod
    def reformat_data(matlab_data) -> dict:
        """Reformat original data loaded from Matlab file.
        """
        data = dict()
        data["significant_ripples"] = matlab_data.SignificantRipples - 1
        data["ripple_info"] = matlab_data.RippleTimes
        data["inhibitory_neurons"] = (
            np.array(matlab_data.InhibitoryNeurons) - 1
        )  # account for matlab indexing
        data["excitatory_neurons"] = (
            matlab_data.ExcitatoryNeurons - 1
        )  # account for matlab indexing
        data["well_locations"] = matlab_data.WellLocations
        data["well_sequence"] = matlab_data.WellSequence
        data["spike_ids"] = matlab_data.SpikeData[:, 1].astype(int) - 1
        data["spike_times_s"] = matlab_data.SpikeData[:, 0]
        data["pos_times_s"] = matlab_data.PositionData[:, 0]
        data["pos_xy_cm"] = np.squeeze(matlab_data.PositionData[:, 1:-1])
        data["ripple_times_s"] = data["ripple_info"][:, :2]
        data["n_ripples"] = len(data["ripple_times_s"])
        data["n_cells"] = np.max(data["spike_ids"] + 1)
        return data

    # ----------------------------

    def clean_recording_data(self, raw_data: dict) -> dict:
        """Checks for and cleans any gaps in position recording. Cleaning affects
        "pos_xy", "pos_times", "spike_ids", "spike_times".
        """
        (
            pos_xy_aligned,
            pos_times_aligned,
            spike_ids_aligned,
            spike_times_aligned,
        ) = self.align_spike_and_position_recording_data(
            raw_data["pos_xy_cm"],
            raw_data["pos_times_s"],
            raw_data["spike_ids"],
            raw_data["spike_times_s"],
        )
        (
            pos_xy_gaps_aligned_and_cleaned,
            large_position_gaps_inds,
        ) = self.clean_recording_gaps(pos_xy_aligned, pos_times_aligned)
        # store cleaned data in new dictionary
        cleaned_data = raw_data.copy()
        cleaned_data["pos_xy_cm"] = pos_xy_gaps_aligned_and_cleaned
        cleaned_data["pos_times_s"] = pos_times_aligned
        cleaned_data["spike_ids"] = spike_ids_aligned
        cleaned_data["spike_times_s"] = spike_times_aligned
        cleaned_data["large_position_gaps_inds"] = large_position_gaps_inds
        return cleaned_data

    # -----

    def clean_recording_gaps(self, pos_xy: np.ndarray, pos_times: np.ndarray):
        """The position coordinates near a large gap in the position recording
        interpolate between the positions at the start and end of the gap, giving
        innaccurate position measurements. In order to account for this, we replace the
        positions at +- a buffer around the time of a gap with np.nan, so they are not
        taken into account for velocity or place field calculations.
        """
        (
            position_gap_inds_above_threshold
        ) = self.check_for_position_gaps_above_threshold(pos_times)
        cleaned_pos_xy = pos_xy[:]
        for ind in position_gap_inds_above_threshold:
            cleaned_pos_xy[ind - 5 : ind + 5] = np.nan
        return (cleaned_pos_xy, position_gap_inds_above_threshold)

    def check_for_position_gaps_above_threshold(self, pos_times: np.ndarray):
        pos_times_diff = np.diff(pos_times)
        position_gap_bool = (
            pos_times_diff > self.params.position_recording_gap_threshold_s
        )

        position_gap_inds_above_threshold = np.where(position_gap_bool)[0]
        if len(position_gap_inds_above_threshold):
            print(
                f"{len(position_gap_inds_above_threshold)} position gaps found with"
                f"{np.round(pos_times_diff[position_gap_inds_above_threshold], 2)} s "
                "missing."
            )
        else:
            print(f"No position gaps found.")
        return position_gap_inds_above_threshold

    # ----------------

    def align_spike_and_position_recording_data(
        self,
        pos_xy: np.ndarray,
        pos_times: np.ndarray,
        spike_ids: np.ndarray,
        spike_times: np.ndarray,
    ) -> tuple:
        (spike_ids_aligned, spike_times_aligned) = self.remove_spikes_without_position(
            spike_ids, spike_times, pos_times
        )
        (pos_xy_aligned, pos_times_aligned) = self.remove_position_without_spikes(
            pos_xy, pos_times, spike_times
        )
        return (
            pos_xy_aligned,
            pos_times_aligned,
            spike_ids_aligned,
            spike_times_aligned,
        )

    def remove_spikes_without_position(
        self, spike_ids: np.ndarray, spike_times: np.ndarray, pos_times: np.ndarray
    ):
        spikes_before_position_recording = spike_times < pos_times[0]
        spikes_after_position_recording = spike_times > pos_times[-1]
        spike_ids_aligned_to_position_recording = spike_ids[
            ~spikes_before_position_recording & ~spikes_after_position_recording
        ]
        spike_times_aligned_to_position_recording = spike_times[
            ~spikes_before_position_recording & ~spikes_after_position_recording
        ]
        return (
            spike_ids_aligned_to_position_recording,
            spike_times_aligned_to_position_recording,
        )

    def remove_position_without_spikes(
        self, pos_xy: np.ndarray, pos_times: np.ndarray, spike_times: np.ndarray
    ):
        position_before_spikes_recording = pos_times < spike_times[0]
        position_after_spikes_recording = pos_times > spike_times[-1]
        pos_xy_aligned_to_spikes_recording = pos_xy[
            ~position_before_spikes_recording & ~position_after_spikes_recording
        ]
        pos_times_aligned_to_spikes_recording = pos_times[
            ~position_before_spikes_recording & ~position_after_spikes_recording
        ]
        return (
            pos_xy_aligned_to_spikes_recording,
            pos_times_aligned_to_spikes_recording,
        )

    def confirm_all_30hz(self, pos_times: np.ndarray, gap_inds: np.ndarray) -> None:
        not30hz = np.round(pos_times[1:] - pos_times[:-1], 2) != np.round(
            self.params.POSITION_RECORDING_RESOLUTION_FRAMES_PER_S, 2
        )
        if len(gap_inds) > 0:
            not30hz[gap_inds] = 0
        if np.sum(not30hz) == 0:
            print("Data cleaning check: SUCCESSFUL, all position time frames are 30 Hz")
        else:
            print(
                f"Data cleaning check: WARNING, {np.sum(not30hz)} position time frames "
                "other than 30Hz"
            )
            print((pos_times[1:] - pos_times[:-1])[not30hz])

    # ----------------------------

    def calculate_velocity_info(self) -> dict:
        velocity_info = dict()
        velocity_info["vel_times_s"] = self.calc_velocity_times(
            self.data["pos_times_s"]
        )
        velocity_info["vel_cm_per_s"] = self.calc_velocity(
            self.data["pos_xy_cm"], self.data["large_position_gaps_inds"]
        )
        (velocity_info["run_starts"], velocity_info["run_ends"]) = self.get_run_periods(
            velocity_info["vel_cm_per_s"], velocity_info["vel_times_s"]
        )
        return velocity_info

    @staticmethod
    def calc_velocity_times(pos_times: np.ndarray) -> np.ndarray:
        pos1 = pos_times[:-1]
        pos2 = pos_times[1:]
        return (pos1 + pos2) / 2

    def calc_velocity(
        self, pos_xy: np.ndarray, large_position_gaps_inds: list
    ) -> np.ndarray:
        x_pos_diff = np.diff(pos_xy[:, 0])
        y_pos_diff = np.diff(pos_xy[:, 1])
        distance = np.sqrt(x_pos_diff ** 2 + y_pos_diff ** 2)
        velocity = distance / self.params.POSITION_RECORDING_RESOLUTION_FRAMES_PER_S
        return velocity

    def get_run_periods(
        self, velocity: np.ndarray, velocity_times: np.ndarray
    ) -> tuple:
        # only use spikes from when animal was moving over running velocity threshold
        if np.any(np.isnan(velocity)):
            velocity_times = velocity_times[~np.isnan(velocity)]
            velocity = velocity[~np.isnan(velocity)]
        run_boolean = velocity > self.params.velocity_run_threshold_cm_per_s
        run_starts, run_ends = utils.boolean_to_times(run_boolean, velocity_times)
        return (run_starts, run_ends)

    # ----------------------------

    def calculate_place_fields(self) -> dict:
        """Calculate place fields"""
        place_field_data = dict()
        spike_ids = self.data["spike_ids"].copy()
        place_field_data["run_data"] = self.get_run_spike_and_pos_data(
            spike_ids,
            self.data["spike_times_s"],
            self.data["pos_xy_cm"],
            self.data["pos_times_s"],
            self.velocity_info,
        )
        place_field_data["spatial_grid"] = self.get_spatial_grid()
        place_field_data["position_histogram"] = self.calc_position_histogram(
            place_field_data["run_data"]["pos_xy_cm"], place_field_data["spatial_grid"]
        )
        place_field_data["spike_histograms"] = self.calc_spike_histograms(
            place_field_data["run_data"]["spike_times_s"],
            place_field_data["run_data"]["spike_ids"],
            self.data["pos_xy_cm"],
            self.data["pos_times_s"],
            place_field_data["spatial_grid"],
        )
        place_field_data["place_fields"] = self.calc_place_fields(
            place_field_data["position_histogram"],
            place_field_data["spike_histograms"],
            posterior=True,
        )
        place_field_data["place_fields_likelihood"] = self.calc_place_fields(
            place_field_data["position_histogram"],
            place_field_data["spike_histograms"],
            posterior=False,
        )
        place_field_data["mean_firing_rate_array"] = self.calc_run_mean_firing_rate(
            place_field_data["position_histogram"], place_field_data["spike_histograms"]
        )
        place_field_data["max_firing_rate_array"] = self.calc_max_tuning_curve_array(
            place_field_data["place_fields"]
        )
        (
            place_field_data["excitatory_neurons"],
            place_field_data["inhibitory_neurons"],
        ) = self.check_excitatory_inhibitory_classification(
            place_field_data["mean_firing_rate_array"]
        )
        (
            place_field_data["place_cell_ids"],
            place_field_data["n_place_cells"],
        ) = self.classify_place_cells(
            self.data["excitatory_neurons"], place_field_data["max_firing_rate_array"]
        )

        return place_field_data

    @staticmethod
    def get_run_spike_and_pos_data(
        spike_ids: np.ndarray,
        spike_times: np.ndarray,
        pos_xy: np.ndarray,
        pos_times: np.ndarray,
        velocity_info: dict,
    ) -> dict:
        run_data = dict()
        run_data["spike_times_s"] = np.array([])
        run_data["spike_ids"] = np.array([])
        run_x_pos = np.array([])
        run_y_pos = np.array([])
        for epoch in range(len(velocity_info["run_starts"])):
            start = velocity_info["run_starts"][epoch]
            end = velocity_info["run_ends"][epoch]
            # extract window indices
            spike_window_bool = utils.times_to_bool(spike_times, start, end)
            pos_window_bool = utils.times_to_bool(pos_times, start, end)
            # extract spikes and positions in this window
            window_spike_times = spike_times[spike_window_bool]
            window_spike_ids = spike_ids[spike_window_bool]
            window_x_pos = pos_xy[:, 0][pos_window_bool]
            window_y_pos = pos_xy[:, 1][pos_window_bool]
            # append to list
            run_data["spike_times_s"] = np.append(
                run_data["spike_times_s"], window_spike_times
            )
            run_data["spike_ids"] = np.append(
                run_data["spike_ids"], window_spike_ids
            ).astype(int)
            run_x_pos = np.append(run_x_pos, window_x_pos)
            run_y_pos = np.append(run_y_pos, window_y_pos)
        run_data["pos_xy_cm"] = np.array((run_x_pos, run_y_pos)).T
        return run_data

    def get_spatial_grid(self):
        spatial_grid = dict()
        spatial_grid["x"] = np.linspace(0, 200, self.params.n_bins_x + 1)
        spatial_grid["y"] = np.linspace(0, 200, self.params.n_bins_y + 1)
        return spatial_grid

    def calc_position_histogram(
        self, run_pos_xy: np.ndarray, spatial_grid: dict
    ) -> np.ndarray:
        position_hist, _, _ = np.histogram2d(
            run_pos_xy[:, 0],
            run_pos_xy[:, 1],
            bins=(spatial_grid["x"], spatial_grid["y"]),
        )
        position_hist = (
            position_hist.T * self.params.POSITION_RECORDING_RESOLUTION_FRAMES_PER_S
        )
        return position_hist

    def calc_spike_histograms(
        self,
        spike_times: np.ndarray,
        spike_ids: np.ndarray,
        pos_xy: np.ndarray,
        pos_times: np.ndarray,
        spatial_grid: dict,
    ) -> np.ndarray:
        spike_histograms = np.zeros(
            (self.data["n_cells"], self.params.n_bins_x, self.params.n_bins_y)
        )
        for cell_id in range(self.data["n_cells"]):
            cell_spike_times = spike_times[spike_ids == cell_id]
            cell_spike_pos_xy = self.get_spike_positions(
                cell_spike_times, pos_xy, pos_times
            )
            if (len(cell_spike_times)) > 0:
                spike_hist, _, _ = np.histogram2d(
                    cell_spike_pos_xy[:, 0],
                    cell_spike_pos_xy[:, 1],
                    bins=(spatial_grid["x"], spatial_grid["y"]),
                )
                spike_histograms[cell_id] = spike_hist.T
            else:
                spike_histograms[cell_id] = np.zeros(
                    (self.params.n_bins_x, self.params.n_bins_y)
                )
        return spike_histograms

    def calc_place_fields(
        self,
        position_histogram: np.ndarray,
        spike_histograms: np.ndarray,
        posterior: bool = True,
    ) -> np.ndarray:
        place_fields = np.zeros(
            (self.data["n_cells"], self.params.n_bins_x, self.params.n_bins_y)
        )
        for i in range(self.data["n_cells"]):
            place_fields[i] = self.calc_one_place_field(
                position_histogram, spike_histograms[i], posterior=posterior
            )
        return place_fields

    def calc_one_place_field(
        self,
        position_hist_s: np.ndarray,
        spike_hist: np.ndarray,
        posterior: bool = True,
    ) -> np.ndarray:
        if posterior:
            spike_hist_with_prior = (
                spike_hist + self.params.place_field_prior_alpha_s - 1
            )
            pos_hist_with_prior_s = (
                position_hist_s + self.params.place_field_prior_beta_s
            )
            place_field_raw = spike_hist_with_prior / pos_hist_with_prior_s
        else:
            place_field_raw = spike_hist / position_hist_s
            place_field_raw = np.nan_to_num(place_field_raw)
        if self.params.rotate_placefields:
            place_field_raw = np.roll(place_field_raw, np.random.randint(50), axis=0)
            place_field_raw = np.roll(place_field_raw, np.random.randint(50), axis=1)
        pf_gaussian_sd_bins = utils.cm_to_bins(self.params.place_field_gaussian_sd_cm)
        place_field_smoothed = gaussian_filter(
            place_field_raw, sigma=pf_gaussian_sd_bins
        )
        return place_field_smoothed

    def get_spike_positions(
        self, cell_spike_times: np.ndarray, pos_xy: np.ndarray, pos_times: np.ndarray
    ) -> np.ndarray:
        cell_spike_pos_xy = np.array(
            [
                self.find_position_during_spike(pos_xy, pos_times, time)
                for time in cell_spike_times
            ]
        )
        return cell_spike_pos_xy

    def find_position_during_spike(
        self, pos_xy: np.ndarray, pos_times: np.ndarray, spike_time: float
    ) -> np.ndarray:
        abs_diff = np.abs(pos_times - spike_time)
        min_diff = np.min(abs_diff)
        if min_diff > self.params.position_recording_gap_threshold_s:
            print(
                "find_pos_ind_nearest_spike() returning value larger than gap "
                f"threshold: {min_diff, np.where(abs_diff == min_diff)}"
            )
        nearest_pos_xy = pos_xy[abs_diff == min_diff][0]
        if nearest_pos_xy.shape != (2,):
            nearest_pos_xy = nearest_pos_xy[0]
        return nearest_pos_xy

    def calc_run_mean_firing_rate(
        self, position_histogram: np.ndarray, spiking_histograms: np.ndarray
    ) -> np.ndarray:
        total_run_time = np.sum(position_histogram)
        total_spikes = np.sum(spiking_histograms, axis=(1, 2))
        mean_fr_array = total_spikes / total_run_time
        return mean_fr_array

    def calc_max_tuning_curve_array(self, place_fields: np.ndarray) -> np.ndarray:
        max_fr_array = np.max(place_fields, axis=(1, 2))
        return max_fr_array

    def check_excitatory_inhibitory_classification(
        self, mean_fr_array: np.ndarray
    ) -> tuple:
        excitatory_neurons = np.squeeze(
            np.argwhere(
                mean_fr_array
                < self.params.inhibitory_firing_rate_threshold_spikes_per_s
            )
        )
        inhibitory_neurons = np.squeeze(
            np.argwhere(
                mean_fr_array
                > self.params.inhibitory_firing_rate_threshold_spikes_per_s
            )
        )
        different_inhibitory_classification = np.any(
            inhibitory_neurons != self.data["inhibitory_neurons"]
        )
        different_excitatory_classification = np.any(
            excitatory_neurons != self.data["excitatory_neurons"]
        )
        if different_excitatory_classification > 0:
            print(
                f"Place field check: WARNING, "
                "excitatory neurons classified differently from original paper."
            )
        elif different_inhibitory_classification > 0:
            print(
                f"Place field check: WARNING, "
                "inhibitory neurons classified differently from original paper."
            )
        else:
            print(
                "Place field check: SUCCESSFUL, same classification of "
                "excitatory and inhibitory neurons as original paper."
            )
        return (excitatory_neurons, inhibitory_neurons)

    def classify_place_cells(
        self, excitatory_ids: np.ndarray, max_tuning_curve_array: np.ndarray
    ) -> tuple:
        max_tuning_curve_above_thresh = np.squeeze(
            np.argwhere(
                max_tuning_curve_array
                > self.params.place_field_minimum_tuning_curve_peak_spikes_per_s
            )
        )
        place_cell_ids = np.intersect1d(excitatory_ids, max_tuning_curve_above_thresh)
        return (place_cell_ids, len(place_cell_ids))
