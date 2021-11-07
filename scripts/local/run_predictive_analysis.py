import click
import numpy as np
from typing import Dict, Tuple


import replay_structure.predictive_analysis as pred
from replay_structure.read_write import (
    load_ratday_data,
    load_spikemat_data,
    load_trajectory_results,
    load_pf_analysis,
    save_predictive_analysis,
)
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.highsynchronyevents import HighSynchronyEvents_Preprocessing
from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.metadata import (
    HighSynchronyEvents,
    HighSynchronyEvents_PF_Data,
    Ripples,
    Session_List,
    Ripples_PF_Data,
    Session_Name,
    Data_Type,
    string_to_data_type,
)


def load_ratday_data_all_sessions(
    bin_size_cm: int = 4
) -> Dict[Tuple[int, int], RatDay_Preprocessing]:
    ratday_data: Dict[Tuple[int, int], RatDay_Preprocessing] = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        ratday_data[session.rat, session.day] = load_ratday_data(
            session, bin_size_cm=bin_size_cm
        )
    return ratday_data


def load_spikemat_times_all_sessions(
    data_type: Data_Type, bin_size_cm: int = 4
) -> Dict[Tuple[int, int], np.ndarray]:
    spikemat_times_s = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        rd_ripple_data = load_spikemat_data(
            session,
            data_type.default_time_window_ms,
            data_type.name,
            bin_size_cm=bin_size_cm,
        )
        if isinstance(rd_ripple_data, Ripple_Preprocessing):
            # assert isinstance(session.rat, int)
            spikemat_times_s[session.rat, session.day] = rd_ripple_data.data[
                "ripple_times_s"
            ]
        elif isinstance(rd_ripple_data, HighSynchronyEvents_Preprocessing):
            spikemat_times_s[session.rat, session.day] = rd_ripple_data.spikemat_info[
                "popburst_times_s"
            ]
        else:
            raise Exception("Invalid data_type.")
    return spikemat_times_s


def load_trajectory_results_all_sessions(
    data_type: Data_Type, trajectory_type, bin_size_cm: int = 4, time_window_ms: int = 3
) -> Dict[Tuple[int, int], dict]:
    trajectory_results = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        if trajectory_type == "viterbi":
            viterbi_trajectories = load_trajectory_results(
                session,
                data_type.default_time_window_ms,
                data_type.name,
                data_type.default_likelihood_function,
                bin_size_cm=bin_size_cm,
            )
            trajectory_results[
                session.rat, session.day
            ] = viterbi_trajectories.most_likely_trajectories
        elif (trajectory_type == "map") or (trajectory_type == "mean"):
            if isinstance(data_type.name, Ripples):
                data_type_ = Ripples_PF_Data
            elif isinstance(data_type.name, HighSynchronyEvents):
                data_type_ = HighSynchronyEvents_PF_Data
            pf_trajectories = load_pf_analysis(
                session,
                data_type_.default_time_window_ms,
                data_type_.name,
                decoding_type=trajectory_type,
            )
            trajectory_results[session.rat, session.day] = pf_trajectories.results[
                "trajectory_map_positions"
            ]
        else:
            raise Exception("Invalid trajectory type")

    return trajectory_results


def get_behavior_paths(
    ratday_data: Dict[Tuple[int, int], RatDay_Preprocessing],
    spikemat_times_s: Dict[Tuple[int, int], np.ndarray],
    distance_threshold_cm: float = 75,
) -> dict:
    past_path: dict = dict()
    future_path: dict = dict()

    for session in Session_List:
        assert isinstance(session, Session_Name)
        rat = session.rat
        day = session.day
        past_path[(rat, day)] = dict()
        future_path[(rat, day)] = dict()
        n_ripples = spikemat_times_s[rat, day].shape[0]
        for ripple_num in range(n_ripples):
            ripple_start = spikemat_times_s[rat, day][ripple_num, 0]
            ripple_end = spikemat_times_s[rat, day][ripple_num, 1]
            past_path_end_time = ripple_start
            past_path_start_time = ripple_start - 10
            past_path[(rat, day)][ripple_num] = pred.get_behavior_path(
                past_path_start_time,
                past_path_end_time,
                ratday_data[(rat, day)].data["pos_times_s"],
                ratday_data[(rat, day)].data["pos_xy_cm"],
                dist_thresh=distance_threshold_cm,
                path_type="past",
            )
            future_path_start_time = ripple_end
            future_path_end_time = ripple_end + 10
            future_path[(rat, day)][ripple_num] = pred.get_behavior_path(
                future_path_start_time,
                future_path_end_time,
                ratday_data[(rat, day)].data["pos_times_s"],
                ratday_data[(rat, day)].data["pos_xy_cm"],
                dist_thresh=distance_threshold_cm,
                path_type="future",
            )

    return {"past": past_path, "future": future_path}


def get_angular_distances(
    data_type: Data_Type,
    behavior_paths: dict,
    trajectory_results: dict,
    radius_array: np.ndarray = np.arange(15, 75, 3),
):
    angular_distances: dict = dict()
    if isinstance(data_type.name, Ripples):
        n_ripples_total: int = 2980
    elif isinstance(data_type.name, HighSynchronyEvents):
        n_ripples_total = 4469
    else:
        raise Exception("Invalid data type.")
    angular_distances["past"] = np.full((n_ripples_total, len(radius_array)), np.nan)
    angular_distances["future"] = np.full((n_ripples_total, len(radius_array)), np.nan)

    angular_distances["behavior"] = np.full(
        (n_ripples_total, len(radius_array)), np.nan
    )

    angular_distances["control_past"] = np.full(
        (n_ripples_total, len(radius_array)), np.nan
    )
    angular_distances["control_future"] = np.full(
        (n_ripples_total, len(radius_array)), np.nan
    )

    ripple_id = 0
    for rat, day in trajectory_results:
        for ripple in trajectory_results[rat, day]:
            if trajectory_results[rat, day][ripple] is not None:
                if len(behavior_paths["future"][rat, day][ripple]) > 0:
                    ripple_angular_dist, _, _ = pred.get_angular_dist_array(
                        behavior_paths["past"][rat, day][ripple],
                        trajectory_results[rat, day][ripple],
                        radius_array=radius_array,
                    )
                    angular_distances["past"][ripple_id] = ripple_angular_dist
                    ripple_angular_dist, _, _ = pred.get_angular_dist_array(
                        behavior_paths["future"][rat, day][ripple],
                        trajectory_results[rat, day][ripple],
                        radius_array=radius_array,
                    )
                    angular_distances["future"][ripple_id] = ripple_angular_dist
                    behavior_angular_dist, _, _ = pred.get_angular_dist_array(
                        behavior_paths["past"][rat, day][ripple],
                        behavior_paths["future"][rat, day][ripple],
                        radius_array=radius_array,
                    )
                    angular_distances["behavior"][ripple_id] = behavior_angular_dist

                    # get control results
                    random_rat = np.random.choice([1, 3])
                    random_day = np.random.randint(1, 3)
                    random_ripple = np.random.randint(
                        len(trajectory_results[random_rat, random_day])
                    )
                    # reselect ripple if selected ripple with no popburst
                    while (
                        trajectory_results[random_rat, random_day][random_ripple]
                        is None
                    ) or (
                        len(trajectory_results[random_rat, random_day][random_ripple])
                        == 0
                    ):
                        random_ripple = np.random.randint(
                            len(trajectory_results[random_rat, random_day])
                        )

                    if len(behavior_paths["past"][rat, day][ripple]) > 0:
                        dist = (
                            trajectory_results[random_rat, random_day][random_ripple][0]
                            - behavior_paths["past"][rat, day][ripple][0]
                        )
                        random_trajectory_shifted = (
                            trajectory_results[random_rat, random_day][random_ripple]
                            - dist
                        )
                        if len(behavior_paths["future"][rat, day][ripple]) > 0:
                            ripple_angular_dist, _, _ = pred.get_angular_dist_array(
                                behavior_paths["past"][rat, day][ripple],
                                random_trajectory_shifted,
                                radius_array=radius_array,
                            )
                            angular_distances["control_past"][
                                ripple_id
                            ] = ripple_angular_dist
                            ripple_angular_dist, _, _ = pred.get_angular_dist_array(
                                behavior_paths["future"][rat, day][ripple],
                                random_trajectory_shifted,
                                radius_array=radius_array,
                            )
                            angular_distances["control_future"][
                                ripple_id
                            ] = ripple_angular_dist

            ripple_id += 1

    return angular_distances


@click.command()
@click.option(
    "--trajectory_type",
    type=click.Choice(["viterbi", "map", "mean"]),
    default="viterbi",
)
@click.option(
    "--data_type",
    type=click.Choice(["ripples", "high_synchrony_events"]),
    default="ripples",
)
def main(trajectory_type: str, data_type: str, bin_size_cm: int = 4):

    data_type_ = string_to_data_type(data_type)

    # load data
    ratday_data = load_ratday_data_all_sessions()
    spikemat_times_s = load_spikemat_times_all_sessions(data_type_)
    trajectory_results = load_trajectory_results_all_sessions(
        data_type_, trajectory_type
    )

    # run predictive analysis
    behavior_paths = get_behavior_paths(ratday_data, spikemat_times_s)
    angular_distances = get_angular_distances(
        data_type_, behavior_paths, trajectory_results
    )

    save_predictive_analysis(
        data_type_.default_time_window_ms,
        data_type_.name,
        data_type_.default_likelihood_function,
        trajectory_type,
        (behavior_paths, angular_distances),
        bin_size_cm=bin_size_cm,
    )


if __name__ == "__main__":
    main()
