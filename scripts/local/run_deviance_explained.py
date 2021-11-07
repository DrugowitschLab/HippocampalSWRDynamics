from replay_structure.structure_analysis_input import Structure_Analysis_Input
import click
from typing import Optional

from replay_structure.deviance_models import Deviance_Explained
from replay_structure.metadata import (
    Data_Type,
    string_to_data_type,
    Session_Indicator,
    string_to_session_indicator,
    Likelihood_Function,
    string_to_likelihood_function,
    Session_List,
)
from replay_structure.read_write import (
    load_structure_data,
    load_model_comparison_results,
    save_deviance_explained_results,
)


def run_deviance_explained(
    data_type: Data_Type,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    session: Session_Indicator,
    filename_ext: str,
):

    # load structure data
    structure_data = load_structure_data(
        session,
        time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )

    structure_data_for_null: Optional[Structure_Analysis_Input]
    if data_type.simulated_data_name is not None:
        structure_data_for_null = load_structure_data(
            Session_List[0],
            time_window_ms,
            data_type.simulated_data_name,
            likelihood_function,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        )
    else:
        structure_data_for_null = None

    # load model comparison results
    model_comparison_results = load_model_comparison_results(
        session,
        time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )

    deviance_explained_results = Deviance_Explained(
        structure_data,
        model_comparison_results,
        structure_data_for_null=structure_data_for_null,
    )

    save_deviance_explained_results(
        session,
        time_window_ms,
        data_type.name,
        likelihood_function,
        deviance_explained_results,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
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
            "placefieldID_shuffle",
            "placefield_rotation",
            "high_synchrony_events",
        ]
    ),
    required=True,
)
@click.option("--session", default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option("--likelihood_function", type=click.STRING, default=None)
@click.option("--filename_ext", type=click.STRING, default="")
def run(
    data_type: str,
    session: Optional[str],
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Optional[str],
    filename_ext: str,
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
        run_deviance_explained(
            data_type_,
            bin_size_cm,
            time_window_ms,
            likelihood_function_,
            session_indicator,
            filename_ext,
        )
    else:
        for session_indicator in data_type_.session_list:
            run_deviance_explained(
                data_type_,
                bin_size_cm,
                time_window_ms,
                likelihood_function_,
                session_indicator,
                filename_ext,
            )


if __name__ == "__main__":
    run()
