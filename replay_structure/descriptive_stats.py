"""
Module for calculating the descriptive stats of trajectories decoded from SWRs.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from replay_structure.metadata import SESSION_RATDAY, N_SESSIONS
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.highsynchronyevents import HighSynchronyEvents_Preprocessing
from replay_structure.model_comparison import Model_Comparison
from replay_structure.structure_trajectory import Most_Likely_Trajectories


def get_descriptive_stats(
    ratday_data: Dict[Tuple[int, int], RatDay_Preprocessing],
    ripple_data: Dict[Tuple[int, int], Ripple_Preprocessing],
    model_comparison_results: Dict[Tuple[int, int], Model_Comparison],
    trajectory_results: Dict[Tuple[int, int], Most_Likely_Trajectories],
) -> pd.DataFrame:

    descriptive_stats_df = pd.DataFrame()

    # get descriptive stats for all SWRs across sessions.
    ripple_id = 0
    for session in range(N_SESSIONS):
        rat = SESSION_RATDAY[session]["rat"]
        day = SESSION_RATDAY[session]["day"]
        if (rat, day) in ratday_data:
            n_ripples = ratday_data[rat, day].data["n_ripples"]
            for ripple_num in range(n_ripples):
                ripple_series = pd.Series(dtype=np.float)
                ripple_series["rat"] = rat
                ripple_series["day"] = day
                ripple_series["ripple"] = ripple_num
                ripple_series["ripple_id"] = ripple_id
                (
                    ripple_series["start_time_s"],
                    ripple_series["end_time_s"],
                    ripple_series["duration_ms"],
                    ripple_series["map_classified_PF"],
                ) = get_ripple_information(ratday_data[(rat, day)], ripple_num)
                (
                    ripple_series["avg_fr"],
                    ripple_series["n_neurons_active"],
                ) = calc_neural_stats(ripple_data[(rat, day)], ripple_num)
                (
                    ripple_series["distance_cm"],
                    ripple_series["velocity_cm_s"],
                    ripple_series["direct_distance_cm"],
                    ripple_series["straightness"],
                    ripple_series["direction"],
                ) = calc_trajectory_stats(
                    trajectory_results[(rat, day)].most_likely_trajectories[ripple_num]
                )
                spikemat_times_s = ratday_data[(rat, day)].data["ripple_times_s"]
                (
                    ripple_series["trial_number"],
                    ripple_series["trial_type"],
                    ripple_series["trial_duration_s"],
                    ripple_series["home_well"],
                    ripple_series["goal_well"],
                    ripple_series["current_location_x"],
                    ripple_series["current_location_y"],
                    ripple_series["current_location_type"],
                ) = get_trial_information(
                    ratday_data[(rat, day)],
                    spikemat_times_s,
                    ripple_num,
                    threshold_cm=9,
                )
                (
                    ripple_series["best_fit_model"],
                    ripple_series["trajectory_model"],
                    ripple_series["diffusion_model_ev"],
                    ripple_series["momentum_model_ev"],
                    ripple_series["stationary_model_ev"],
                    ripple_series["stationary_gaussian_model_ev"],
                    ripple_series["random_model_ev"],
                ) = get_mc_info(model_comparison_results[(rat, day)], ripple_num)
                descriptive_stats_df = descriptive_stats_df.append(
                    ripple_series, ignore_index=True
                )

                ripple_id += 1

    # get replay within trial information
    trajectory_df = descriptive_stats_df[descriptive_stats_df["trajectory_model"] == 1]
    (ripple_num_in_trial_array, n_replay_in_trial_array) = get_replay_trial_information(
        ratday_data, trajectory_df
    )
    descriptive_stats_df.loc[
        trajectory_df.index, "replay_num_in_trial"
    ] = ripple_num_in_trial_array
    descriptive_stats_df.loc[
        trajectory_df.index, "n_replay_in_trial"
    ] = n_replay_in_trial_array
    return descriptive_stats_df


# -----------------------


def calc_trajectory_stats(
    trajectory_cm: np.ndarray, time_window_s: float = 0.003
) -> tuple:
    if trajectory_cm is not None:
        distance_cm = calc_distance(trajectory_cm)
        velocity_cm_s = calc_velocity(distance_cm, len(trajectory_cm) * time_window_s)
        direct_distance_cm = calc_direct_distance(trajectory_cm)
        straightness = calc_straightness(distance_cm, direct_distance_cm)
        direction = calc_direction(trajectory_cm)
    else:
        distance_cm, velocity_cm_s, direct_distance_cm, straightness, direction = (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        )
    return distance_cm, velocity_cm_s, direct_distance_cm, straightness, direction


def calc_distance(trajectory: np.ndarray) -> float:
    return np.sum(np.sqrt(np.sum((trajectory[1:] - trajectory[:-1]) ** 2, axis=1)))


def calc_velocity(distance: float, duration: float) -> float:
    return distance / duration


def calc_direct_distance(trajectory: np.ndarray) -> float:
    return np.sqrt(np.sum((trajectory[-1] - trajectory[0]) ** 2))


def calc_straightness(distance: float, direct_distance: float) -> float:
    if distance != 0:
        return direct_distance / distance
    else:
        return np.nan


def calc_direction(trajectory: np.ndarray) -> float:
    radian = np.arctan(
        (trajectory[0][0] - trajectory[-1][0]) / (trajectory[0][1] - trajectory[-1][1])
    )
    return np.degrees(radian)


# -----------------------


def get_ripple_information(ratday_data: RatDay_Preprocessing, ripple_num: int) -> tuple:
    start_time_s = ratday_data.data["ripple_times_s"][ripple_num, 0]
    end_time_s = ratday_data.data["ripple_times_s"][ripple_num, 1]
    duration_ms = (
        ratday_data.data["ripple_times_s"][ripple_num, 1]
        - ratday_data.data["ripple_times_s"][ripple_num, 0]
    ) * 1000

    map_classified = (
        True if ripple_num in ratday_data.data["significant_ripples"] else False
    )
    return (start_time_s, end_time_s, duration_ms, map_classified)


# -----------------------


def calc_neural_stats(ripple_data: Ripple_Preprocessing, ripple_num: int) -> tuple:
    if ripple_data.ripple_info["spikemats_popburst"][ripple_num] is not None:
        duration_s = (
            np.shape(ripple_data.ripple_info["spikemats_popburst"][ripple_num])[0]
            * ripple_data.params.time_window_s
        )
        n_cells = np.shape(ripple_data.ripple_info["spikemats_popburst"][ripple_num])[1]
        total_spikes = ripple_data.ripple_info["spikemats_popburst"][ripple_num].sum()
        avg_fr = total_spikes / duration_s / n_cells
        n_neurons_active = np.sum(
            np.any(ripple_data.ripple_info["spikemats_popburst"][ripple_num], axis=1)
        )
    else:
        avg_fr, n_neurons_active = np.nan, np.nan
    return avg_fr, n_neurons_active


# -----------------------


def get_trial_information(
    ratday_data: RatDay_Preprocessing,
    spikemat_times_s: np.ndarray,
    ripple_num: int,
    threshold_cm=7,
) -> tuple:
    trial_number = get_trial_number(ratday_data, spikemat_times_s, ripple_num)
    trial_type = get_trial_type(ratday_data, trial_number)
    trial_duration_s = get_trial_duration_s(ratday_data, trial_number)
    home_well = int(ratday_data.data["well_sequence"][0, 1] - 1)
    goal_well = get_goal_well(ratday_data, trial_number, trial_type)
    current_location_x, current_location_y = get_current_location(
        ratday_data, spikemat_times_s, ripple_num
    )
    current_location_type = get_current_location_type(
        ratday_data,
        home_well,
        goal_well,
        [current_location_x, current_location_y],
        threshold_cm,
    )
    return (
        trial_number,
        trial_type,
        trial_duration_s,
        home_well,
        goal_well,
        current_location_x,
        current_location_y,
        current_location_type,
    )


def get_trial_number(
    ratday: RatDay_Preprocessing, spikemat_times_s: np.ndarray, ripple_num: int
) -> int:
    time_bool = spikemat_times_s[ripple_num, 0] < ratday.data["well_sequence"][:, 0]
    if time_bool.sum() > 0:
        trial_number = np.argwhere(time_bool)[0][0] - 1
    else:
        trial_number = len(ratday.data["well_sequence"][:, 0]) - 1
    return trial_number


def get_trial_type(ratday: RatDay_Preprocessing, trial_number: int) -> str:
    home_well = ratday.data["well_sequence"][0, 1]
    if ratday.data["well_sequence"][int(trial_number), 1].astype(int) == home_well:
        trial_type = "home"
    else:
        trial_type = "away"
    return trial_type


def get_trial_duration_s(ratday: RatDay_Preprocessing, trial_number: int) -> float:
    if trial_number == 0:
        trial_duraration_s = (
            ratday.data["well_sequence"][0, 0] - ratday.data["pos_times_s"][0]
        )
    else:
        trial_duraration_s = (
            ratday.data["well_sequence"][trial_number, 0]
            - ratday.data["well_sequence"][trial_number - 1, 0]
        )
    return trial_duraration_s


def get_goal_well(
    ratday_data: RatDay_Preprocessing, trial_number: int, trial_type: str
) -> int:
    if trial_type == "home":
        goal_well = int(ratday_data.data["well_sequence"][trial_number - 1, 1]) - 1
    else:
        goal_well = int(ratday_data.data["well_sequence"][trial_number, 1]) - 1
    return goal_well


def get_current_location(
    ratday: RatDay_Preprocessing, spikemat_times_s: np.ndarray, ripple_num: int
) -> List[float]:
    if np.isnan(spikemat_times_s[ripple_num, 0]):
        return [np.nan, np.nan]
    else:
        diff = np.abs(ratday.data["pos_times_s"] - spikemat_times_s[ripple_num, 0])
        min_diff_ind = np.argwhere(diff == np.min(diff))[0][0]
        start_position = ratday.data["pos_xy_cm"][min_diff_ind]
        return [start_position[0], start_position[1]]


def get_current_location_type(
    ratday_data: RatDay_Preprocessing,
    home_well: int,
    goal_well: int,
    current_location: List[float],
    threshold_cm: float,
) -> str:
    home_location = ratday_data.data["well_locations"][home_well, :2]
    goal_location = ratday_data.data["well_locations"][goal_well, :2]
    dist_to_home = np.sqrt(np.sum((current_location - home_location) ** 2))
    dist_to_goal = np.sqrt(np.sum((current_location - goal_location) ** 2))
    if (dist_to_home < threshold_cm) & (dist_to_goal < threshold_cm):
        if home_well == goal_well:
            current_well_type = "home"
        else:
            print("ERROR")
    elif dist_to_home < threshold_cm:
        current_well_type = "home"
    elif dist_to_goal < threshold_cm:
        current_well_type = "goal"
    else:
        current_well_type = "other"
    return current_well_type


# -----------------------


def get_mc_info(mc_results: Model_Comparison, ripple_num: int) -> tuple:
    best_fit_model = mc_results.results_dataframe["mll_model"][ripple_num]
    if (best_fit_model == "diffusion") or (best_fit_model == "momentum"):
        trajectory_model = True
    else:
        trajectory_model = False
    diffusion_model_ev = mc_results.results_dataframe["diffusion"][ripple_num]
    momentum_model_ev = mc_results.results_dataframe["momentum"][ripple_num]
    stationary_model_ev = mc_results.results_dataframe["stationary"][ripple_num]
    stationary_gaussian_model_ev = mc_results.results_dataframe["stationary_gaussian"][
        ripple_num
    ]
    random_model_ev = mc_results.results_dataframe["random"][ripple_num]
    return (
        best_fit_model,
        trajectory_model,
        diffusion_model_ev,
        momentum_model_ev,
        stationary_model_ev,
        stationary_gaussian_model_ev,
        random_model_ev,
    )


# -----------------------


def get_replay_trial_information(
    ratday_data_all_sessions: Dict[Tuple[int, int], RatDay_Preprocessing],
    trajectory_df: pd.DataFrame,
) -> tuple:
    ripple_num_in_trial_array = get_replay_num_in_trial(
        trajectory_df["trial_number"].values
    )
    n_replay_in_trial_array = get_n_replay_in_trial(
        ratday_data_all_sessions, trajectory_df
    )
    return ripple_num_in_trial_array, n_replay_in_trial_array


def get_replay_num_in_trial(replay_trial_numbers: np.ndarray) -> int:
    ripple_num_in_trial_array = np.zeros(len(replay_trial_numbers))
    ripple_num_in_trial = 1
    for i in range(1, len(replay_trial_numbers)):
        if replay_trial_numbers[i] == replay_trial_numbers[i - 1]:
            ripple_num_in_trial_array[i] = ripple_num_in_trial
            ripple_num_in_trial += 1
        else:
            ripple_num_in_trial_array[i] = 0
            ripple_num_in_trial = 1
    return ripple_num_in_trial_array


def get_n_replay_in_trial(
    ratday_data_all_sessions: Dict[Tuple[int, int], RatDay_Preprocessing],
    trajectory_df: pd.DataFrame,
) -> int:
    n_replay_in_trial_array = np.zeros(len(trajectory_df))

    for i in range(N_SESSIONS):
        rat = SESSION_RATDAY[i]["rat"]
        day = SESSION_RATDAY[i]["day"]
        n_trials = ratday_data_all_sessions[rat, day].data["well_sequence"].shape[0]
        for trial_number in range(n_trials - 1):
            replay_in_trial = (
                (trajectory_df["rat"] == rat)
                & (trajectory_df["day"] == day)
                & (trajectory_df["trial_number"] == (trial_number + 1))
            )
            n_replay_in_trial = replay_in_trial.sum()
            n_replay_in_trial_array[replay_in_trial] = n_replay_in_trial
    return n_replay_in_trial_array


# -----------------------


def get_descriptive_stats_hse(
    ratday_data: Dict[Tuple[int, int], RatDay_Preprocessing],
    hse_data: Dict[Tuple[int, int], HighSynchronyEvents_Preprocessing],
    model_comparison_results: Dict[Tuple[int, int], Model_Comparison],
) -> pd.DataFrame:

    descriptive_stats_df = pd.DataFrame()

    ripple_id = 0

    for session in range(N_SESSIONS):
        rat = SESSION_RATDAY[session]["rat"]
        day = SESSION_RATDAY[session]["day"]
        if (rat, day) in ratday_data:
            print(rat, day)
            n_ripples = len(hse_data[rat, day].spikemat_info["spikemats_popburst"])
            for ripple_num in range(n_ripples):
                ripple_series = pd.Series(dtype=np.float)
                ripple_series["rat"] = rat
                ripple_series["day"] = day
                ripple_series["ripple"] = ripple_num
                ripple_series["ripple_id"] = ripple_id
                spikemat_times_s = hse_data[(rat, day)].spikemat_info[
                    "popburst_times_s"
                ]
                (
                    ripple_series["trial_number"],
                    ripple_series["trial_type"],
                    ripple_series["trial_duration_s"],
                    ripple_series["home_well"],
                    ripple_series["goal_well"],
                    ripple_series["current_location_x"],
                    ripple_series["current_location_y"],
                    ripple_series["current_location_type"],
                ) = get_trial_information(
                    ratday_data[(rat, day)],
                    spikemat_times_s,
                    ripple_num,
                    threshold_cm=9,
                )
                (
                    ripple_series["best_fit_model"],
                    ripple_series["trajectory_model"],
                    ripple_series["diffusion_model_ev"],
                    ripple_series["momentum_model_ev"],
                    ripple_series["stationary_model_ev"],
                    ripple_series["stationary_gaussian_model_ev"],
                    ripple_series["random_model_ev"],
                ) = get_mc_info(model_comparison_results[(rat, day)], ripple_num)
                descriptive_stats_df = descriptive_stats_df.append(
                    ripple_series, ignore_index=True
                )

                ripple_id += 1

    return descriptive_stats_df
