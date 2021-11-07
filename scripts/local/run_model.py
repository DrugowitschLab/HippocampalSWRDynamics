import click

from typing import Optional

from replay_structure.metadata import (
    Likelihood_Function,
    Model,
    Diffusion,
    Momentum,
    Stationary,
    Stationary_Gaussian,
    Random,
    Data_Type,
    string_to_data_type,
    string_to_model,
    Session_Indicator,
    string_to_session_indicator,
    string_to_likelihood_function,
)
import replay_structure.structure_models as models
from replay_structure.read_write import (
    load_structure_data,
    save_structure_model_results,
)


def run_model(
    model: Model,
    data_type: Data_Type,
    session_indicator: Session_Indicator,
    bin_size_cm: int,
    time_window_ms: int,
    likelihood_function: Likelihood_Function,
    filename_ext: str,
):
    print(
        f"running {model.name} model on {data_type.name} data, "
        f"with {bin_size_cm}cm bins and {time_window_ms}ms time window"
    )
    structure_data = structure_data = load_structure_data(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )

    if isinstance(model.name, Diffusion):
        pass
    if isinstance(model.name, Momentum):
        pass
    if isinstance(model.name, Stationary):
        model_results = models.Stationary(structure_data).get_model_evidences()
    if isinstance(model.name, Stationary_Gaussian):
        pass
    if isinstance(model.name, Random):
        model_results = models.Random(structure_data).get_model_evidences()

    save_structure_model_results(
        session_indicator,
        time_window_ms,
        data_type.name,
        likelihood_function,
        model.name,
        model_results,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


@click.command()
@click.option(
    "--model_name",
    type=click.Choice(
        ["diffusion", "momentum", "stationary", "stationary_gaussian", "random"]
    ),
    required=True,
)
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
@click.option("--session", type=click.STRING, default=None)
@click.option("--likelihood_function", type=click.STRING, default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    model_name: str,
    data_type: str,
    session: Optional[str],
    likelihood_function: Optional[str],
    bin_size_cm: int,
    time_window_ms: int,
    filename_ext: str,
):
    data_type_ = string_to_data_type(data_type)
    model = string_to_model(model_name)
    if time_window_ms is None:
        time_window_ms = data_type_.default_time_window_ms

    if likelihood_function is None:
        likelihood_function_ = data_type_.default_likelihood_function
    else:
        likelihood_function_ = string_to_likelihood_function(likelihood_function)

    if session is not None:
        session_indicator: Session_Indicator = string_to_session_indicator(session)
        run_model(
            model,
            data_type_,
            session_indicator,
            bin_size_cm,
            time_window_ms,
            likelihood_function_,
            filename_ext,
        )
    else:
        for session_indicator in data_type_.session_list:
            run_model(
                model,
                data_type_,
                session_indicator,
                bin_size_cm,
                time_window_ms,
                likelihood_function_,
                filename_ext,
            )


if __name__ == "__main__":
    main()
