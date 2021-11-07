import numpy as np
import replay_structure.utils as utils
from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.config import Structure_Analysis_Input_Parameters


class PF_Analysis:
    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        decoding_type="map",
        save_only_trajectories=False,
    ):
        self.min_adjacent_distance = 50
        self.min_steps_in_trajectory = 10
        self.min_trajectory_distance = 80
        self.decoding_type = decoding_type
        self.results = self.run_pf_analysis(
            structure_data, save_only_trajectories=save_only_trajectories
        )

    # --------------------------------------

    def run_pf_analysis(
        self, structure_data: Structure_Analysis_Input, save_only_trajectories=False
    ):
        results = self.initialize_results()
        for ripple_num in range(len(structure_data.spikemats)):
            if (structure_data.spikemats[ripple_num] is not None) and (
                structure_data.spikemats[ripple_num].shape[0] != 0
            ):
                emission_probabilities_log = utils.calc_poisson_emission_probabilities_log(
                    structure_data.spikemats[ripple_num],
                    structure_data.pf_matrix,
                    structure_data.params.time_window_s,
                )
                if self.decoding_type == "map":
                    map_positions = self.get_map_position(
                        structure_data.params, emission_probabilities_log
                    )
                elif self.decoding_type == "mean":
                    map_positions = self.get_weighted_map_position(
                        structure_data.params, emission_probabilities_log
                    )
                summed_posterior = self.get_summed_posterior(
                    structure_data.params, emission_probabilities_log
                )
                (
                    trajectory_map_positions,
                    trajectory_distance,
                    trajectory_start_to_end_distance,
                    trajectory_num_steps,
                    trajectory_start_ind,
                    trajectory_end_ind,
                    sig,
                ) = self.identify_trajectory_events(map_positions.T)
            else:
                emission_probabilities_log = np.nan
                map_positions = np.nan
                summed_posterior = np.nan
                trajectory_map_positions = np.nan
                trajectory_distance = np.nan
                trajectory_start_to_end_distance = np.nan
                trajectory_num_steps = np.nan
                trajectory_start_ind = np.nan
                trajectory_end_ind = np.nan
                sig = np.nan
            results["posteriors"][ripple_num] = emission_probabilities_log
            results["map_positions"][ripple_num] = map_positions
            results["summed_posterior"][ripple_num] = summed_posterior
            results["trajectory_map_positions"][ripple_num] = trajectory_map_positions
            results["trajectory_distance"][ripple_num] = trajectory_distance
            results["trajectory_start_to_end_distance"][
                ripple_num
            ] = trajectory_start_to_end_distance
            results["trajectory_num_steps"][ripple_num] = trajectory_num_steps
            results["trajectory_start_ind"][ripple_num] = trajectory_start_ind
            results["trajectory_end_ind"][ripple_num] = trajectory_end_ind
            results["significant_trajectory"][ripple_num] = sig
        if save_only_trajectories:
            for key in results:
                if (key != "trajectory_map_positions") and (key != "map_positions"):
                    results[key] = None
        return results

    def initialize_results(self) -> dict:
        results: dict = dict()
        results["posteriors"] = dict()
        results["map_positions"] = dict()
        results["summed_posterior"] = dict()
        results["significant_trajectory"] = dict()
        results["trajectory_map_positions"] = dict()
        results["trajectory_distance"] = dict()
        results["trajectory_start_to_end_distance"] = dict()
        results["trajectory_start_ind"] = dict()
        results["trajectory_end_ind"] = dict()
        results["trajectory_num_steps"] = dict()
        return results

    def get_map_position(
        self,
        params: Structure_Analysis_Input_Parameters,
        emission_probabilities_log: np.ndarray,
    ) -> np.ndarray:
        map_ind = np.nanargmax(emission_probabilities_log, axis=0)
        map_y = (map_ind // params.n_bins_x) * params.bin_size_cm + (
            params.bin_size_cm / 2
        )
        map_x = (map_ind % params.n_bins_y) * params.bin_size_cm + (
            params.bin_size_cm / 2
        )
        return np.squeeze([map_x, map_y]).T

    def get_weighted_map_position(
        self,
        params: Structure_Analysis_Input_Parameters,
        emission_probabilities_log: np.ndarray,
    ) -> np.ndarray:

        x = np.arange(params.n_bins_x)
        y = np.arange(params.n_bins_y)
        xx, yy = np.meshgrid(x, y)
        # map_ind = np.nanargmax(emission_probabilities_log, axis=0)
        emission_probabilities = np.exp(emission_probabilities_log).reshape(
            params.n_bins_x, params.n_bins_y, -1
        )
        normalized_emission_probabilies = emission_probabilities / np.nansum(
            emission_probabilities, axis=(0, 1)
        )
        map_x = np.nansum(
            normalized_emission_probabilies.transpose(2, 0, 1) * xx, axis=(1, 2)
        )
        map_y = np.nansum(
            normalized_emission_probabilies.transpose(2, 0, 1) * yy, axis=(1, 2)
        )
        map_x_cm = map_x * params.bin_size_cm + (params.bin_size_cm / 2)
        map_y_cm = map_y * params.bin_size_cm + (params.bin_size_cm / 2)
        return np.squeeze([map_x_cm, map_y_cm]).T

    def get_summed_posterior(
        self,
        params: Structure_Analysis_Input_Parameters,
        emission_probabilities_log: np.ndarray,
    ) -> np.ndarray:
        n_timesteps = np.shape(emission_probabilities_log)[1]
        emission_exp = np.exp(emission_probabilities_log)
        emission_exp_norm = emission_exp / np.sum(emission_exp, axis=0)
        emission_2d = np.reshape(
            emission_exp_norm, (params.n_bins_x, params.n_bins_x, n_timesteps)
        )
        sum_emission_exp = np.sum(emission_2d, axis=2)
        return sum_emission_exp

    def identify_trajectory_events(self, map_positions: np.ndarray):

        map_distances = self.calc_map_distances(map_positions)
        (
            trajectory_start_ind,
            trajectory_end_ind,
            trajectory_num_steps,
        ) = self.find_longest_continuous_trajectory(map_distances)
        trajectory_distance = self.calc_trajectory_distance(
            map_positions, trajectory_start_ind, trajectory_end_ind
        )
        trajectory_start_to_end_distance = self.calc_start_to_end_trajectory_distance(
            map_positions, trajectory_start_ind, trajectory_end_ind
        )
        trajectory_map_positions = map_positions[
            :, trajectory_start_ind:trajectory_end_ind
        ]
        if (
            trajectory_start_to_end_distance > self.min_trajectory_distance
            and trajectory_num_steps > self.min_steps_in_trajectory
        ):
            sig = True
        else:
            sig = False
        return (
            trajectory_map_positions.T,
            trajectory_distance,
            trajectory_start_to_end_distance,
            trajectory_num_steps,
            trajectory_start_ind,
            trajectory_end_ind,
            sig,
        )

    def calc_map_distances(self, map_positions):
        x_pos = map_positions[0]
        y_pos = map_positions[1]
        x_diff = np.diff(x_pos)
        y_diff = np.diff(y_pos)
        distances = np.sqrt(x_diff ** 2 + y_diff ** 2)
        return distances

    def find_longest_continuous_trajectory(self, map_distance):
        below_threshold_list = map_distance < self.min_adjacent_distance
        max_counter = 0
        max_counter_ind = list()
        current_counter = 0
        current_counter_ind = list()
        for ind, below_threshold in enumerate(below_threshold_list):
            if below_threshold:
                current_counter += 1
                current_counter_ind.append(ind)
            else:
                current_counter = 0
                current_counter_ind = list()
            if current_counter > max_counter:
                max_counter = current_counter
                max_counter_ind = current_counter_ind
        if len(max_counter_ind) > 0:
            trajectory_start_ind = min(max_counter_ind)
            trajectory_end_ind = max(max_counter_ind)
            trajectory_num_steps = trajectory_end_ind - trajectory_start_ind
        else:
            trajectory_start_ind = 0
            trajectory_end_ind = 0
            trajectory_num_steps = 0
        return trajectory_start_ind, trajectory_end_ind, trajectory_num_steps

    def calc_trajectory_distance(
        self, map_positions, trajectory_start_ind, trajectory_end_ind
    ):
        start_x = map_positions[0][trajectory_start_ind]
        start_y = map_positions[1][trajectory_start_ind]
        distances = np.array([])
        for end_ind in range(trajectory_start_ind, trajectory_end_ind + 1):
            end_x = map_positions[0][end_ind]
            end_y = map_positions[1][end_ind]
            trajectory_distance = np.sqrt(
                (end_x - start_x) ** 2 + (end_y - start_y) ** 2
            )
            distances = np.append(distances, trajectory_distance)
        trajectory_distance = np.max(distances)
        return trajectory_distance

    def calc_start_to_end_trajectory_distance(
        self, map_positions, trajectory_start_ind, trajectory_end_ind
    ):
        start_x = map_positions[0][trajectory_start_ind]
        start_y = map_positions[1][trajectory_start_ind]
        end_x = map_positions[0][trajectory_end_ind]
        end_y = map_positions[1][trajectory_end_ind]
        start_to_end_trajectory_distance = np.sqrt(
            (end_x - start_x) ** 2 + (end_y - start_y) ** 2
        )

        return start_to_end_trajectory_distance
