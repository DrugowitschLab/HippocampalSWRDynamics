import click
import numpy as np
import scipy.stats as sp
from typing import Optional

from replay_structure.model_recovery import (
    Model_Recovery_Trajectory_Set_Parameters,
    Model_Recovery_Trajectory_Set,
    Model_Parameter_Distribution_Prior,
    Diffusion_Model_Parameter_Prior,
    Momentum_Model_Parameter_Prior,
    Gaussian_Model_Parameter_Prior,
    Stationary_Model_Parameter_Prior,
    Random_Model_Parameter_Prior,
)

from replay_structure.simulated_neural_data import (
    Simulated_Data_Preprocessing,
    Simulated_Spikes_Parameters,
)
from replay_structure.read_write import (
    load_structure_data,
    load_marginalized_gridsearch_results,
    save_model_recovery_simulated_trajectory_set,
    save_spikemat_data,
)
from replay_structure.metadata import (
    MODELS_AS_STR,
    Data_Type,
    Poisson,
    string_to_data_type,
    Diffusion,
    Momentum,
    Stationary,
    Stationary_Gaussian,
    Random,
    Session_List,
    Simulated_Session_Name,
    string_to_session_indicator,
)
from replay_structure.utils import LogNorm_Distribution


def get_duration_distribution(
    spikemats: dict, time_window_s: float
) -> LogNorm_Distribution:
    spikemat_durations_s = (
        np.array(
            [
                spikemats[i].shape[0]
                for i in range(len(spikemats))
                if spikemats[i] is not None
            ]
        )
        * time_window_s
    )
    s, loc, scale = sp.lognorm.fit(spikemat_durations_s, floc=0)
    return LogNorm_Distribution(s=s, loc=loc, scale=scale)


def get_model_param_dist(
    session_indicator: Simulated_Session_Name,
    data_type,
    time_window_ms,
    bin_size_cm,
    filename_ext="",
) -> Model_Parameter_Distribution_Prior:
    model_param_dist: Model_Parameter_Distribution_Prior
    print(session_indicator.model.name)
    if isinstance(session_indicator.model.name, Diffusion):
        # load gridsearch results for distribution of params for simulated trajectories
        gridsearch_best_fit_distibution = load_marginalized_gridsearch_results(
            Session_List[0],
            time_window_ms,
            data_type.simulated_data_name,
            Poisson(),
            session_indicator.model.name,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        ).marginalization_info["fit_prior_params"]
        model_param_dist = Diffusion_Model_Parameter_Prior(
            gridsearch_best_fit_distibution["sd_meters"]
        )
    elif isinstance(session_indicator.model.name, Momentum):
        gridsearch_best_fit_distibution = load_marginalized_gridsearch_results(
            Session_List[0],
            time_window_ms,
            data_type.simulated_data_name,
            Poisson(),
            session_indicator.model.name,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        ).marginalization_info["fit_prior_params"]
        model_param_dist = Momentum_Model_Parameter_Prior(
            gridsearch_best_fit_distibution["2d_normal"]
        )
    elif isinstance(session_indicator.model.name, Stationary):
        model_param_dist = Stationary_Model_Parameter_Prior()
    elif isinstance(session_indicator.model.name, Stationary_Gaussian):
        gridsearch_best_fit_distibution = load_marginalized_gridsearch_results(
            Session_List[0],
            time_window_ms,
            data_type.simulated_data_name,
            Poisson(),
            session_indicator.model.name,
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        ).marginalization_info["fit_prior_params"]
        model_param_dist = Gaussian_Model_Parameter_Prior(
            gridsearch_best_fit_distibution["sd_meters"]
        )
    elif isinstance(session_indicator.model.name, Random):
        model_param_dist = Random_Model_Parameter_Prior()
    else:
        raise Exception("Invalid model.")
    return model_param_dist


def generate_data(
    bin_size_cm: int,
    time_window_ms: int,
    data_type: Data_Type,
    session_indicator: Simulated_Session_Name,
    filename_ext: str,
):
    print(
        "generating simulated data under {} model dynamics with {}cm bins".format(
            session_indicator.model.name, bin_size_cm
        )
    )

    # load structure data to get params for simulated trajectories and neural data
    assert data_type.simulated_data_name is not None
    structure_data = load_structure_data(
        Session_List[0],
        time_window_ms,
        data_type.simulated_data_name,
        data_type.default_likelihood_function,
        bin_size_cm=bin_size_cm,
    )

    # generate and save simulated trajectory set
    duration_s_dist = get_duration_distribution(
        structure_data.spikemats, structure_data.params.time_window_s
    )
    model_param_dist = get_model_param_dist(
        session_indicator, data_type, time_window_ms, bin_size_cm
    )
    trajectory_set_params = Model_Recovery_Trajectory_Set_Parameters(
        model_param_dist, duration_s_dist
    )
    trajectory_set = Model_Recovery_Trajectory_Set(trajectory_set_params)

    print("DONE")
    save_model_recovery_simulated_trajectory_set(
        trajectory_set, session_indicator, data_type.name, ext=filename_ext
    )

    pf_matrix_posterior = structure_data.pf_matrix

    # generate and save simulated neural data
    simulated_spikes_params = Simulated_Spikes_Parameters(
        pf_matrix_posterior,
        structure_data.params.time_window_ms,
        structure_data.params.likelihood_function_params,
    )

    simulated_data = Simulated_Data_Preprocessing(
        trajectory_set.trajectory_set, simulated_spikes_params
    )
    save_spikemat_data(
        simulated_data,
        session_indicator,
        time_window_ms,
        data_type.name,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


@click.command()
@click.option("--model", type=click.Choice(MODELS_AS_STR), default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option(
    "--data_type",
    type=click.Choice(["poisson_simulated_ripples", "negbinomial_simulated_ripples"]),
    default="simulated_ripples",
)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    model: Optional[str],
    bin_size_cm: int,
    time_window_ms: int,
    data_type: str,
    filename_ext: str,
):
    data_type_: Data_Type = string_to_data_type(data_type)
    if time_window_ms is None:
        time_window_ms = data_type_.default_time_window_ms

    if model is not None:
        session_indicator = string_to_session_indicator(model)
        assert isinstance(session_indicator, Simulated_Session_Name)
        generate_data(
            bin_size_cm, time_window_ms, data_type_, session_indicator, filename_ext
        )
    else:
        for session_indicator in data_type_.session_list:
            assert isinstance(session_indicator, Simulated_Session_Name)
            generate_data(
                bin_size_cm, time_window_ms, data_type_, session_indicator, filename_ext
            )


if __name__ == "__main__":
    main()
