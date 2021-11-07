import numpy as np

import replay_structure.utils as utils
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.config import Run_Snippet_Preprocessing_Parameters


class Run_Snippet_Preprocessing:
    """
    Extracts neural activity within run snippets.
    """

    def __init__(
        self,
        ratday: RatDay_Preprocessing,
        ripple_data: Ripple_Preprocessing,
        params: Run_Snippet_Preprocessing_Parameters,
    ):
        self.params = params
        self.pf_matrix = utils.get_pf_matrix(
            ratday.place_field_data["place_fields"],
            ratday.place_field_data["place_cell_ids"],
        )
        print("Getting spikemats")
        self.run_info = self.get_run_snippet_info(ratday, ripple_data)

    def get_run_snippet_info(
        self, ratday: RatDay_Preprocessing, ripple_data: Ripple_Preprocessing
    ) -> dict:
        np.random.seed(self.params.random_seed)
        run_snippet_info = dict()
        run_snippet_info["run_times_s"] = self.select_run_snippets(ratday, ripple_data)
        run_snippet_info["true_trajectories_cm"] = utils.get_trajectories(
            ratday, run_snippet_info["run_times_s"]
        )
        run_snippet_info["spikemats"] = self.get_spikemats(
            ratday, run_snippet_info["run_times_s"]
        )

        return run_snippet_info

    def select_run_snippets(
        self, ratday: RatDay_Preprocessing, ripple_data: Ripple_Preprocessing
    ) -> np.ndarray:
        run_snippet_durations_s = self.get_run_snippet_durations(ripple_data)
        run_period_times_s = self.get_run_period_times(ratday)
        run_snippet_times_s = self.get_run_snippet_times(
            run_period_times_s, run_snippet_durations_s
        )
        return run_snippet_times_s

    def get_run_period_times(self, ratday: RatDay_Preprocessing):
        run_lengths = (
            ratday.velocity_info["run_ends"] - ratday.velocity_info["run_starts"]
        )
        run_periods_use = run_lengths > self.params.run_period_threshold_s
        run_starts = ratday.velocity_info["run_starts"][run_periods_use]
        run_ends = ratday.velocity_info["run_ends"][run_periods_use]
        return np.vstack((run_starts, run_ends)).T

    def get_run_snippet_durations(self, ripple_data: Ripple_Preprocessing):
        ripple_spikemat_durations_s = (
            ripple_data.ripple_info["popburst_times_s"][:, 1]
            - ripple_data.ripple_info["popburst_times_s"][:, 0]
        )
        run_snippet_durations_s = (
            ripple_spikemat_durations_s * self.params.duration_scaling_factor
        )
        return run_snippet_durations_s

    def get_run_snippet_times(
        self, run_period_times_s: np.ndarray, run_snippet_durations_s: np.ndarray
    ) -> np.ndarray:
        run_snippet_times = np.zeros((len(run_snippet_durations_s), 2))
        run_period_times_sample_pool = run_period_times_s

        for i, snippet_duration in enumerate(run_snippet_durations_s):
            # sample run period that is larger than snippet length
            sample_pool_durations = (
                run_period_times_sample_pool[:, 1] - run_period_times_sample_pool[:, 0]
            )
            snippet_times_sample_pool = run_period_times_sample_pool[
                sample_pool_durations > snippet_duration
            ]
            if len(snippet_times_sample_pool) > 0:

                run_ind_sample = np.random.choice(len(snippet_times_sample_pool))
                run_sample_times = snippet_times_sample_pool[run_ind_sample]

                # sample snippet from within run period
                snippet_start = np.random.uniform(
                    run_sample_times[0], run_sample_times[1] - snippet_duration
                )
                snippet_end = snippet_start + snippet_duration
                run_snippet_times[i] = [snippet_start, snippet_end]
            else:
                run_snippet_times[i] = [np.nan, np.nan]
        run_snippet_times = run_snippet_times[
            ~np.any(np.isnan(run_snippet_times), axis=1)
        ]
        return run_snippet_times

    def get_spikemats(
        self, ratday: RatDay_Preprocessing, run_period_times: np.ndarray
    ) -> dict:
        spikemats = dict()
        for run_period_num in range(len(run_period_times)):
            run_period_start = run_period_times[run_period_num][0]
            run_period_end = run_period_times[run_period_num][1]
            spikemats[run_period_num] = utils.get_spikemat(
                ratday.data["spike_ids"],
                ratday.data["spike_times_s"],
                ratday.place_field_data["place_cell_ids"],
                run_period_start,
                run_period_end,
                self.params.time_window_s,
                self.params.time_window_advance_s,
            )
        return spikemats
