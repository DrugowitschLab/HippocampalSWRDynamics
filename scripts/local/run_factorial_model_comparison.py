import click
import numpy as np
from typing import Optional, Dict

from replay_structure.model_comparison import Factorial_Model_Comparison
from replay_structure.read_write import (
    load_structure_model_results,
    load_marginalized_gridsearch_results,
    save_factorial_model_comparison_results,
)
from replay_structure.metadata import (
    MODELS,
    Data_Type,
    Poisson,
    Neg_Binomial,
    string_to_data_type,
    Session_Indicator,
    string_to_session_indicator,
    Likelihood_Function,
    Session_Name,
    Simulated_Session_Name,
)


def load_model_evidences(
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    filename_ext: str,
):

    model_evidences: Dict[str, np.ndarray] = dict()

    for model in MODELS:
        if model.n_params is not None:
            model_evidences[str(model.name)] = load_marginalized_gridsearch_results(
                session_indicator,
                time_window_ms,
                data_type.name,
                likelihood_function,
                model.name,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
            ).marginalized_model_evidences
        else:
            model_evidences[str(model.name)] = load_structure_model_results(
                session_indicator,
                time_window_ms,
                data_type.name,
                likelihood_function,
                model.name,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
            )
    return model_evidences


def run_factorial_model_comparison(
    data_type: Data_Type,
    bin_size_cm: int,
    time_window_ms: int,
    session_indicator: Session_Indicator,
    filename_ext: str,
):

    model_evidences: Dict[str, dict] = dict()
    for likelihood in [Poisson(), Neg_Binomial()]:
        model_evidences[str(likelihood)] = load_model_evidences(
            data_type,
            session_indicator,
            bin_size_cm,
            time_window_ms,
            likelihood,
            filename_ext,
        )

    if isinstance(session_indicator, Session_Name):
        random_effects_prior = 8
    elif isinstance(session_indicator, Simulated_Session_Name):
        random_effects_prior = 3
    else:
        raise Exception("Invalid session_indicator type.")

    mc_results = Factorial_Model_Comparison(
        model_evidences, random_effects_prior=random_effects_prior
    )

    save_factorial_model_comparison_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        mc_results,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(
        ["ripples", "poisson_simulated_ripples", "negbinomial_simulated_ripples"]
    ),
    default="ripples",
)
@click.option("--session", default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=3)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    data_type: str,
    session: Optional[str],
    bin_size_cm: int,
    time_window_ms: int,
    filename_ext: str,
):
    data_type_ = string_to_data_type(data_type)

    if session is not None:
        session_indicator: Session_Indicator = string_to_session_indicator(session)
        run_factorial_model_comparison(
            data_type_, bin_size_cm, time_window_ms, session_indicator, filename_ext
        )
    else:
        for session_indicator in data_type_.session_list:
            run_factorial_model_comparison(
                data_type_, bin_size_cm, time_window_ms, session_indicator, filename_ext
            )


if __name__ == "__main__":
    main()
