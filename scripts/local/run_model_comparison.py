import click
import numpy as np
from typing import Optional, Dict

from replay_structure.structure_models_gridsearch import Structure_Gridsearch
from replay_structure.model_comparison import (
    Gridsearch_Marginalization,
    Model_Comparison,
)
from replay_structure.read_write import (
    load_structure_model_results,
    load_gridsearch_results,
    save_marginalized_gridsearch_results,
    load_marginalized_gridsearch_results,
    save_model_comparison_results,
    aggregate_momentum_gridsearch,
)
from replay_structure.metadata import (
    MODELS,
    Data_Type,
    Momentum_Model,
    Poisson,
    string_to_data_type,
    Session_Indicator,
    string_to_session_indicator,
    Session_Name,
    Simulated_Session_Name,
    Likelihood_Function,
    string_to_likelihood_function,
    Session_List,
)


def run_gridsearch_marginalization(
    data_type: Data_Type,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    session_indicator: Session_Indicator,
    filename_ext: str,
):
    for model in MODELS:
        if model.n_params is not None:
            gridsearch_results = load_gridsearch_results(
                session_indicator,
                time_window_ms,
                data_type.name,
                likelihood_function,
                model.name,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
            )
            assert isinstance(gridsearch_results, Structure_Gridsearch)
            if isinstance(session_indicator, Session_Name):
                marginalized_gridsearch = Gridsearch_Marginalization(gridsearch_results)
            elif isinstance(session_indicator, Simulated_Session_Name):
                assert data_type.simulated_data_name is not None
                marginalization_info = load_marginalized_gridsearch_results(
                    Session_List[0],
                    time_window_ms,
                    data_type.simulated_data_name,
                    Poisson(),
                    model.name,
                    bin_size_cm=bin_size_cm,
                ).marginalization_info
                marginalized_gridsearch = Gridsearch_Marginalization(
                    gridsearch_results, marginalization_info=marginalization_info
                )
            save_marginalized_gridsearch_results(
                session_indicator,
                time_window_ms,
                data_type.name,
                likelihood_function,
                model.name,
                marginalized_gridsearch,
                bin_size_cm=bin_size_cm,
                ext=filename_ext,
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


def run_model_comparison(
    data_type: Data_Type,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    session_indicator: Session_Indicator,
    filename_ext: str,
):

    if Momentum_Model in MODELS:
        aggregate_momentum_gridsearch(
            session_indicator,
            time_window_ms,
            data_type.name,
            likelihood_function,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        )

    run_gridsearch_marginalization(
        data_type,
        bin_size_cm,
        time_window_ms,
        likelihood_function,
        session_indicator,
        filename_ext,
    )

    model_evidences: dict = load_model_evidences(
        data_type,
        session_indicator,
        bin_size_cm,
        time_window_ms,
        likelihood_function,
        filename_ext,
    )

    if isinstance(session_indicator, Session_Name):
        random_effects_prior = 10
    elif isinstance(session_indicator, Simulated_Session_Name):
        random_effects_prior = 2
    else:
        raise Exception("Invalid session_indicator type.")

    mc_results = Model_Comparison(
        model_evidences, random_effects_prior=random_effects_prior
    )

    save_model_comparison_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        mc_results,
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
def main(
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
        run_model_comparison(
            data_type_,
            bin_size_cm,
            time_window_ms,
            likelihood_function_,
            session_indicator,
            filename_ext,
        )
    else:
        for session_indicator in data_type_.session_list:
            run_model_comparison(
                data_type_,
                bin_size_cm,
                time_window_ms,
                likelihood_function_,
                session_indicator,
                filename_ext,
            )


if __name__ == "__main__":
    main()
