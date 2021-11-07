import click

# import scipy.stats as sp
from typing import Optional

# import replay_structure.maximum_likelihood_parameters as ml_params
from replay_structure.metadata import (
    string_to_data_type,
    string_to_session_indicator,
    Session_Indicator,
    Data_Type,
    Likelihood_Function,
    string_to_likelihood_function,
)
from replay_structure.structure_trajectory import Most_Likely_Trajectories
from replay_structure.read_write import load_structure_data, save_trajectory_results


def get_trajectories(
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    sd_meters: Optional[float],
) -> None:
    print(
        f"running viterbi algorithm on {data_type.name} data, "
        f"with {bin_size_cm}cm bins and {time_window_ms}ms time window"
    )
    structure_data = load_structure_data(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=bin_size_cm,
    )

    if sd_meters is None:
        raise AttributeError("Enter sd_meters")
    print(sd_meters)

    if structure_data is not None:
        trajectory_results = Most_Likely_Trajectories(structure_data, sd_meters)
    else:
        trajectory_results = None
    save_trajectory_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        trajectory_results,
        bin_size_cm=bin_size_cm,
    )


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(
        [
            "ripples",
            "run_snippets",
            "poisson_simulated_ripples",
            "negbinomial_simulated_ripples",
        ]
    ),
    required=True,
)
@click.option("--session", default=None)
@click.option("--likelihood_function", type=click.STRING, default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option("--sd_meters", default=None, type=click.FLOAT)
def main(
    data_type: str,
    session: str,
    likelihood_function: Optional[str],
    bin_size_cm: int,
    time_window_ms: Optional[int],
    sd_meters: Optional[float],
):

    data_type_ = string_to_data_type(data_type)
    if time_window_ms is None:
        time_window_ms = data_type_.default_time_window_ms

    if likelihood_function is None:
        likelihood_function_ = data_type_.default_likelihood_function
    else:
        likelihood_function_ = string_to_likelihood_function(likelihood_function)

    if session is not None:
        session_indicator: Session_Indicator = string_to_session_indicator(session)
        get_trajectories(
            data_type_,
            session_indicator,
            bin_size_cm,
            time_window_ms,
            likelihood_function_,
            sd_meters,
        )
    else:
        for session_indicator in data_type_.session_list[2:]:
            get_trajectories(
                data_type_,
                session_indicator,
                bin_size_cm,
                time_window_ms,
                likelihood_function_,
                sd_meters,
            )


if __name__ == "__main__":
    main()
