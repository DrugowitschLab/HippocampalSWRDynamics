import click
from typing import Optional

from replay_structure.diffusion_constant import Diffusion_Constant
from replay_structure.metadata import (
    Data_Type,
    string_to_data_type,
    Session_Indicator,
    Session_Name,
    Simulated_Session_Name,
    string_to_session_indicator,
)
from replay_structure.read_write import (
    load_trajectory_results,
    load_spikemat_data,
    load_model_recovery_simulated_trajectory_set,
    save_diffusion_constant_results,
)
from replay_structure.run_snippet_preprocessing import Run_Snippet_Preprocessing


def run_diffusion_constant_analysis(
    bin_size_cm: int,
    time_window_ms: int,
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    trajectory_type: str,
    bin_space: bool,
    filename_ext: str,
):

    if trajectory_type == "inferred":
        trajectories: dict = load_trajectory_results(
            session_indicator,
            time_window_ms,
            data_type.name,
            data_type.default_likelihood_function,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        ).most_likely_trajectories
    elif trajectory_type == "true":
        if isinstance(session_indicator, Session_Name):
            spikemat_data = load_spikemat_data(
                session_indicator,
                time_window_ms,
                data_type.name,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
            )
            assert isinstance(spikemat_data, Run_Snippet_Preprocessing)
            trajectories = spikemat_data.run_info["true_trajectories_cm"]
        elif isinstance(session_indicator, Simulated_Session_Name):
            mc_recovery_trajectories = load_model_recovery_simulated_trajectory_set(
                data_type.name, session_indicator, ext=filename_ext
            )
            trajectories = {
                i: mc_recovery_trajectories.trajectory_set[i].trajectory_cm
                for i in range(len(mc_recovery_trajectories.trajectory_set))
            }
    else:
        raise Exception("Invalid trajectory_type")
    diffusion_constant_results = Diffusion_Constant(
        trajectories, bin_size_cm=bin_size_cm
    )

    save_diffusion_constant_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        diffusion_constant_results,
        trajectory_type,
        bin_space=bin_space,
        bin_size_cm=bin_size_cm,
    )


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(["ripples", "run_snippets", "poisson_simulated_ripples"]),
    required=True,
)
@click.option("--session", default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option(
    "--trajectory_type", type=click.Choice(["true", "inferred"]), required=True
)
@click.option("--bin_space", is_flag=True)
@click.option("--filename_ext", type=click.STRING, default="")
def run(
    data_type: str,
    session: Optional[str],
    bin_size_cm: int,
    time_window_ms: int,
    trajectory_type: str,
    bin_space: bool,
    filename_ext: str,
):
    data_type_ = string_to_data_type(data_type)

    if time_window_ms is None:
        time_window_ms = data_type_.default_time_window_ms

    if session is not None:
        session_indicator: Session_Indicator = string_to_session_indicator(session)
        run_diffusion_constant_analysis(
            bin_size_cm,
            time_window_ms,
            data_type_,
            session_indicator,
            trajectory_type,
            bin_space,
            filename_ext,
        )
    else:
        for session_indicator in data_type_.session_list:
            run_diffusion_constant_analysis(
                bin_size_cm,
                time_window_ms,
                data_type_,
                session_indicator,
                trajectory_type,
                bin_space,
                filename_ext,
            )


if __name__ == "__main__":
    run()
