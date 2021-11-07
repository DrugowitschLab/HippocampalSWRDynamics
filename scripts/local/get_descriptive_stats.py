import pandas as pd
import click
from typing import Dict, Tuple

from replay_structure.read_write import (
    load_ratday_data,
    load_spikemat_data,
    load_model_comparison_results,
    load_trajectory_results,
    save_descriptive_stats,
)

from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.model_comparison import Model_Comparison
from replay_structure.structure_trajectory import Most_Likely_Trajectories
from replay_structure.descriptive_stats import (
    get_descriptive_stats,
    get_descriptive_stats_hse,
)
from replay_structure.metadata import (
    HighSynchronyEvents,
    Ripples,
    Session_Name,
    Session_List,
    string_to_data_type,
    Data_Type,
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


def load_ripple_data_all_sessions(
    data_type: Data_Type, bin_size_cm: int = 4, time_window_ms: int = 3
) -> dict:
    ripple_data = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        rd_ripple_data = load_spikemat_data(
            session,
            data_type.default_time_window_ms,
            data_type.name,
            bin_size_cm=bin_size_cm,
        )
        ripple_data[session.rat, session.day] = rd_ripple_data
    return ripple_data


def load_model_comparison_results_all_sessions(
    data_type: Data_Type, bin_size_cm: int = 4, time_window_ms: int = 3
) -> Dict[Tuple[int, int], Model_Comparison]:
    model_comparison_results: Dict[Tuple[int, int], Model_Comparison] = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        model_comparison_results[
            session.rat, session.day
        ] = load_model_comparison_results(
            session,
            data_type.default_time_window_ms,
            data_type.name,
            data_type.default_likelihood_function,
            bin_size_cm=bin_size_cm,
        )

    return model_comparison_results


def load_trajectory_results_all_sessions(
    data_type: Data_Type, bin_size_cm: int = 4, time_window_ms: int = 3
) -> Dict[Tuple[int, int], Most_Likely_Trajectories]:
    trajectory_results: Dict[Tuple[int, int], Most_Likely_Trajectories] = dict()
    for session in Session_List:
        assert isinstance(session, Session_Name)
        trajectory_results[session.rat, session.day] = load_trajectory_results(
            session,
            data_type.default_time_window_ms,
            data_type.name,
            data_type.default_likelihood_function,
            bin_size_cm=bin_size_cm,
        )
    return trajectory_results


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(["ripples", "high_synchrony_events"]),
    default="ripples",
)
def main(data_type: str, bin_size_cm: int = 4):

    data_type_ = string_to_data_type(data_type)

    # load data
    ratday_data = load_ratday_data_all_sessions()
    ripple_data = load_ripple_data_all_sessions(data_type_)
    model_comparison_results = load_model_comparison_results_all_sessions(data_type_)
    trajectory_results = load_trajectory_results_all_sessions(data_type_)

    if isinstance(data_type_.name, Ripples):
        descriptive_stats: pd.DataFrame = get_descriptive_stats(
            ratday_data, ripple_data, model_comparison_results, trajectory_results
        )
    elif isinstance(data_type_.name, HighSynchronyEvents):
        descriptive_stats = get_descriptive_stats_hse(
            ratday_data, ripple_data, model_comparison_results
        )

    save_descriptive_stats(
        data_type_.default_time_window_ms,
        data_type_.name,
        data_type_.default_likelihood_function,
        descriptive_stats,
        bin_size_cm=bin_size_cm,
    )


if __name__ == "__main__":
    main()
