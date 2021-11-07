import pickle
import compress_pickle
import os
import numpy as np
import pandas as pd
from typing import Optional, Union

from replay_structure.metadata import (
    DATA_PATH,
    Likelihood_Function,
    RESULTS_PATH,
    DATA_PATH_O2,
    RESULTS_PATH_O2,
    Model_Name,
    Momentum,
    Data_Type_Name,
    Session_Name,
    SessionSpikemat_Name,
    Simulated_Session_Name,
    Session_Indicator,
)
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.run_snippet_preprocessing import Run_Snippet_Preprocessing
from replay_structure.highsynchronyevents import HighSynchronyEvents_Preprocessing
from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.structure_models_gridsearch import Structure_Gridsearch
from replay_structure.structure_trajectory import Most_Likely_Trajectories
from replay_structure.marginals import All_Models_Marginals
from replay_structure.model_comparison import (
    Gridsearch_Marginalization,
    Model_Comparison,
    Factorial_Model_Comparison,
)
from replay_structure.deviance_models import Deviance_Explained
from replay_structure.diffusion_constant import Diffusion_Constant

from replay_structure.model_recovery import Model_Recovery_Trajectory_Set
from replay_structure.simulated_neural_data import Simulated_Data_Preprocessing
from replay_structure.pf_analysis import PF_Analysis


def load_data(filename, print_filename=True):
    if print_filename:
        print("loading ", filename)
    with open(filename, "rb") as file_object:
        raw_data = file_object.read()
        deserialized = pickle.loads(raw_data)
    return deserialized


def load_compressed_data(filename, print_filename=True):
    if print_filename:
        print("loading ", filename)
    with open(filename, "rb") as file_object:
        raw_data = file_object.read()
        deserialized = compress_pickle.loads(raw_data, "gzip")
    return deserialized


def save_data(data, filename, print_filename=True):
    if print_filename:
        print("saving ", filename)
    serialized = pickle.dumps(data)
    with open(filename, "wb") as file_object:
        file_object.write(serialized)


def save_compressed_data(data, filename, print_filename=True):
    if print_filename:
        print("saving ", filename)
    serialized = compress_pickle.dumps(data, "gzip")
    with open(filename, "wb") as file_object:
        file_object.write(serialized)


# ----


def save_ratday_data(
    ratday: RatDay_Preprocessing,
    session_indicator: Session_Name,
    bin_size_cm: int = 4,
    placefields_rotated: bool = False,
    ext="",
) -> None:
    if placefields_rotated:
        filename = os.path.join(
            DATA_PATH,
            "ratday",
            f"{session_indicator}_{bin_size_cm}cm_placefields_rotated{ext}.obj",
        )
    else:
        filename = os.path.join(
            DATA_PATH, "ratday", f"{session_indicator}_{bin_size_cm}cm{ext}.obj"
        )
    save_data(ratday, filename)


def load_ratday_data(
    session_indicator: Session_Name,
    bin_size_cm: int = 4,
    placefields_rotated: bool = False,
    ext="",
) -> RatDay_Preprocessing:
    if placefields_rotated:
        filename = os.path.join(
            DATA_PATH,
            "ratday",
            f"{session_indicator}_{bin_size_cm}cm_placefields_rotated{ext}.obj",
        )
    else:
        filename = os.path.join(
            DATA_PATH, "ratday", f"{session_indicator}_{bin_size_cm}cm{ext}.obj"
        )
    ratday = load_data(filename)
    return ratday


# ------


def save_spikemat_data(
    spikemat_data: Union[
        Ripple_Preprocessing,
        Run_Snippet_Preprocessing,
        Simulated_Data_Preprocessing,
        HighSynchronyEvents_Preprocessing,
    ],
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    bin_size_cm: int = 4,
    ext="",
) -> None:
    filename = os.path.join(
        DATA_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms{ext}.obj",
    )
    save_data(spikemat_data, filename)


def load_spikemat_data(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    bin_size_cm: int = 4,
    ext="",
    print_filename: bool = True,
) -> Union[
    Ripple_Preprocessing,
    Run_Snippet_Preprocessing,
    Simulated_Data_Preprocessing,
    HighSynchronyEvents_Preprocessing,
]:
    filename = os.path.join(
        DATA_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms{ext}.obj",
    )
    spikemat_data = load_data(filename, print_filename=print_filename)
    return spikemat_data


# ----


def save_structure_data(
    structure_data: Structure_Analysis_Input,
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    ext="",
) -> None:
    if isinstance(session_indicator, SessionSpikemat_Name):
        folder = "spikemat_structure_analysis_input"
    else:
        folder = "structure_analysis_input"

    filename = os.path.join(
        DATA_PATH,
        folder,
        f"{session_indicator}_{data_type}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}{ext}.obj",
    )
    save_data(structure_data, filename)


def load_structure_data(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    o2: bool = False,
    ext="",
    print_filename: bool = True,
) -> Structure_Analysis_Input:
    if isinstance(session_indicator, SessionSpikemat_Name):
        folder = "spikemat_structure_analysis_input"
    else:
        folder = "structure_analysis_input"
    filename = os.path.join(
        f"{DATA_PATH_O2 if o2 else DATA_PATH}",
        folder,
        f"{session_indicator}_{data_type}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}{ext}.obj",
    )
    structure_data = load_data(filename, print_filename)
    return structure_data


# ----


def save_structure_model_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    model: Model_Name,
    model_evidences: np.ndarray,
    bin_size_cm: int = 4,
    ext="",
) -> None:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_{model}{ext}.obj",
    )
    save_data(model_evidences, filename)


def load_structure_model_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    model: Model_Name,
    bin_size_cm: int = 4,
    ext="",
) -> np.ndarray:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_{model}{ext}.obj",
    )
    model_evidences = load_data(filename)
    return model_evidences


# ----


def save_gridsearch_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    model: Model_Name,
    gridsearch_results: Structure_Gridsearch,
    bin_size_cm: int = 4,
    spikemat_ind: Optional[int] = None,
    o2: bool = False,
    ext="",
) -> None:
    if spikemat_ind is not None:
        filename = os.path.join(
            f"{RESULTS_PATH_O2 if o2 else RESULTS_PATH}",
            str(data_type),
            f"{session_indicator}_spikemat{spikemat_ind}_{bin_size_cm}cm_"
            f"{time_window_ms}ms_{likelihood_function}_{model}_gridsearch{ext}.obj",
        )
    else:
        filename = os.path.join(
            f"{RESULTS_PATH_O2 if o2 else RESULTS_PATH}",
            str(data_type),
            f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
            f"{likelihood_function}_{model}_gridsearch{ext}.obj",
        )
    save_data(gridsearch_results, filename)


def load_gridsearch_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    model: Model_Name,
    bin_size_cm: int = 4,
    spikemat_ind: Optional[int] = None,
    # o2: bool = False,
    print_filename=True,
    ext="",
) -> Optional[Structure_Gridsearch]:
    if spikemat_ind is not None:
        filename = os.path.join(
            RESULTS_PATH,
            str(data_type),
            f"{session_indicator}_spikemat{spikemat_ind}_{bin_size_cm}cm_"
            f"{time_window_ms}ms_{likelihood_function}_{model}_gridsearch{ext}.obj",
        )
    else:
        filename = os.path.join(
            RESULTS_PATH,
            str(data_type),
            f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
            f"{likelihood_function}_{model}_gridsearch{ext}.obj",
        )
    if os.path.isfile(filename):
        # if datetime.fromtimestamp(os.path.getmtime(filename)).month == 2:
        gridsearch_results = load_data(filename, print_filename=print_filename)
        return gridsearch_results
    else:
        print(f"No file: {filename}")
        return None


def aggregate_momentum_gridsearch(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    ext="",
):
    # get n_ripples
    structure_data = load_structure_data(
        session_indicator,
        time_window_ms,
        data_type,
        likelihood_function,
        bin_size_cm=bin_size_cm,
    )
    n_spikemats = len(structure_data.spikemats)

    # load first ripple and get gridsearch parameters
    momentum_gridsearch_0 = load_gridsearch_results(
        session_indicator,
        time_window_ms,
        data_type,
        likelihood_function,
        Momentum(),
        spikemat_ind=0,
        bin_size_cm=bin_size_cm,
        print_filename=True,
        ext=ext,
    )
    if momentum_gridsearch_0 is None:
        momentum_gridsearch_0 = load_gridsearch_results(
            session_indicator,
            time_window_ms,
            data_type,
            likelihood_function,
            Momentum(),
            spikemat_ind=2,
            bin_size_cm=bin_size_cm,
            print_filename=False,
            ext=ext,
        )
    assert isinstance(momentum_gridsearch_0, Structure_Gridsearch)
    n_sd = len(momentum_gridsearch_0.gridsearch_params["sd_array_meters"])
    n_decay = len(momentum_gridsearch_0.gridsearch_params["decay_array"])

    # fill out gridsearch_results
    gridsearch_results = np.full((n_spikemats, n_sd, n_decay), np.nan)
    to_run_on_o2_medium = np.array([])
    for ripple in range(n_spikemats):
        ripple_gridsearch = load_gridsearch_results(
            session_indicator,
            time_window_ms,
            data_type,
            likelihood_function,
            Momentum(),
            spikemat_ind=ripple,
            bin_size_cm=bin_size_cm,
            print_filename=False,
            ext=ext,
        )
        if isinstance(ripple_gridsearch, Structure_Gridsearch):
            gridsearch_results[ripple] = ripple_gridsearch.gridsearch_results
        else:
            to_run_on_o2_medium = np.append(to_run_on_o2_medium, ripple)
    print(
        f"Session: {session_indicator}, run on o2 medium: {to_run_on_o2_medium}, "
        f"{len(to_run_on_o2_medium)} ripples total"
    )

    # replace gridsearch_results and save
    momentum_gridsearch_aggregated = momentum_gridsearch_0
    momentum_gridsearch_aggregated.gridsearch_results = gridsearch_results
    save_gridsearch_results(
        session_indicator,
        time_window_ms,
        data_type,
        likelihood_function,
        Momentum(),
        momentum_gridsearch_aggregated,
        bin_size_cm=bin_size_cm,
        ext=ext,
    )


# -----


def save_marginalized_gridsearch_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    model: Model_Name,
    marginalized_gridsearch: Gridsearch_Marginalization,
    bin_size_cm: int = 4,
    ext="",
) -> None:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_{model}_gridsearch_marginalization{ext}.obj",
    )
    save_data(marginalized_gridsearch, filename)


def load_marginalized_gridsearch_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    model: Model_Name,
    bin_size_cm: int = 4,
    ext="",
) -> Gridsearch_Marginalization:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_{model}_gridsearch_marginalization{ext}.obj",
    )
    marginalized_gridsearch = load_data(filename)
    return marginalized_gridsearch


# -----


def save_model_comparison_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    mc_results: Model_Comparison,
    bin_size_cm: int = 4,
    ext="",
):
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_model_comparison{ext}.obj",
    )
    save_data(mc_results, filename)


def load_model_comparison_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    ext="",
) -> Model_Comparison:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_model_comparison{ext}.obj",
    )
    mc_results = load_data(filename)
    return mc_results


# -----


def save_factorial_model_comparison_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    # likelihood_function: Likelihood_Function,
    mc_results: Factorial_Model_Comparison,
    bin_size_cm: int = 4,
    ext="",
):
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"factorial_model_comparison{ext}.obj",
    )
    save_data(mc_results, filename)


def load_factorial_model_comparison_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    # likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    ext="",
) -> Factorial_Model_Comparison:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"factorial_model_comparison{ext}.obj",
    )
    mc_results = load_data(filename)
    return mc_results


# -----


def save_deviance_explained_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    deviance_results: Deviance_Explained,
    bin_size_cm: int = 4,
    ext="",
):
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_deviance_explained{ext}.obj",
    )
    save_data(deviance_results, filename)


def load_deviance_explained_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    ext="",
) -> Deviance_Explained:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_deviance_explained{ext}.obj",
    )
    deviance_results = load_data(filename)
    return deviance_results


# -----


def save_trajectory_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    trajectory_results: Most_Likely_Trajectories,
    bin_size_cm: int = 4,
    o2=False,
    ext="",
) -> None:
    # if data_type == "ripple":
    filename = os.path.join(
        f"{RESULTS_PATH_O2 if o2 else RESULTS_PATH}",
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_trajectories{ext}.obj",
    )
    save_data(trajectory_results, filename)


def load_trajectory_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    ext="",
) -> Most_Likely_Trajectories:
    # if data_type == "ripple":
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_trajectories{ext}.obj",
    )
    trajectory_results = load_data(filename)
    return trajectory_results


# -----


def save_marginals(
    session_indicator: Session_Indicator,
    spikemat_ind: int,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    marginals: All_Models_Marginals,
    bin_size_cm: int = 4,
    ext="",
) -> None:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_spikemat{spikemat_ind}_{bin_size_cm}cm_"
        f"{time_window_ms}ms_{likelihood_function}_marginals{ext}.obj",
    )
    save_data(marginals, filename)


def load_marginals(
    session_indicator: Session_Indicator,
    spikemat_ind: int,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    ext="",
) -> All_Models_Marginals:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_spikemat{spikemat_ind}_{bin_size_cm}cm_"
        f"{time_window_ms}ms_{likelihood_function}_marginals{ext}.obj",
    )
    marginals = load_data(filename, print_filename=False)
    return marginals


def save_diffusion_marginals(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    marginals: dict,
    bin_size_cm: int = 4,
    ext="",
) -> None:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_"
        f"{time_window_ms}ms_{likelihood_function}_diffusion_marginals{ext}.obj",
    )
    save_data(marginals, filename)


def load_diffusion_marginals(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    ext="",
) -> dict:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_"
        f"{time_window_ms}ms_{likelihood_function}_diffusion_marginals{ext}.obj",
    )
    marginals = load_data(filename)
    return marginals


# -----


def save_diffusion_constant_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    diffusion_constant_results: Diffusion_Constant,
    trajectory_type: str,
    bin_size_cm: int = 4,
    bin_space: bool = False,
    ext="",
) -> None:
    if bin_space:
        filename = os.path.join(
            RESULTS_PATH,
            str(data_type),
            f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
            f"{likelihood_function}_{trajectory_type}_binned_trajectories_"
            f"diffusion_constant{ext}.obj",
        )
    else:
        filename = os.path.join(
            RESULTS_PATH,
            str(data_type),
            f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
            f"{likelihood_function}_{trajectory_type}_trajectories_"
            f"diffusion_constant{ext}.obj",
        )
    save_data(diffusion_constant_results, filename)


def load_diffusion_constant_results(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    trajectory_type: str,
    bin_size_cm: int = 4,
    bin_space: bool = False,
    ext="",
) -> Diffusion_Constant:
    if bin_space:
        filename = os.path.join(
            RESULTS_PATH,
            str(data_type),
            f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
            f"{likelihood_function}_{trajectory_type}_binned_trajectories_"
            f"diffusion_constant{ext}.obj",
        )
    else:
        filename = os.path.join(
            RESULTS_PATH,
            str(data_type),
            f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_"
            f"{likelihood_function}_{trajectory_type}_trajectories_"
            f"diffusion_constant{ext}.obj",
        )
    diffusion_constant_results = load_data(filename)
    return diffusion_constant_results


# -----


def save_descriptive_stats(
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    descriptive_stats: pd.DataFrame,
    bin_size_cm: int = 4,
    ext="",
) -> None:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"descriptive_stats_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}{ext}.csv",
    )
    descriptive_stats.to_csv(filename)


def load_descriptive_stats(
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    bin_size_cm: int = 4,
    ext="",
) -> pd.DataFrame:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"descriptive_stats_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}{ext}.csv",
    )
    descriptive_stats = pd.read_csv(filename)
    return descriptive_stats


# -----


def save_predictive_analysis(
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    trajectory_type: str,
    predictive_analysis,
    ext="",
    bin_size_cm: int = 4,
) -> None:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"predictive_analysis_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_{trajectory_type}trajectories{ext}.obj",
    )
    save_data(predictive_analysis, filename)


def load_predictive_analysis(
    time_window_ms: int,
    data_type: Data_Type_Name,
    likelihood_function: Likelihood_Function,
    trajectory_type: str,
    bin_size_cm: int = 4,
    ext="",
):
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"predictive_analysis_{bin_size_cm}cm_{time_window_ms}ms_"
        f"{likelihood_function}_{trajectory_type}trajectories{ext}.obj",
    )
    predictive_analysis = load_data(filename)
    return predictive_analysis


# -----


def save_model_recovery_simulated_trajectory_set(
    trajectory_set: Model_Recovery_Trajectory_Set,
    session_indicator: Simulated_Session_Name,
    data_type: Data_Type_Name,
    ext="",
):
    filename = os.path.join(
        DATA_PATH,
        str(data_type),
        f"{session_indicator}_simulated_trajectories{ext}.obj",
    )
    save_data(trajectory_set, filename)


def load_model_recovery_simulated_trajectory_set(
    data_type: Data_Type_Name, session_indicator: Simulated_Session_Name, ext=""
) -> Model_Recovery_Trajectory_Set:
    filename = os.path.join(
        DATA_PATH,
        str(data_type),
        f"{session_indicator}_simulated_trajectories{ext}.obj",
    )
    trajectory_set = load_data(filename)
    return trajectory_set


# ------


def save_pf_analysis(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    map_analysis: PF_Analysis,
    decoding_type: str,
    bin_size_cm: int = 4,
    ext="",
):
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_pf_analysis_"
        f"{decoding_type}{ext}.obj",
    )
    save_data(map_analysis, filename)


def load_pf_analysis(
    session_indicator: Session_Indicator,
    time_window_ms: int,
    data_type: Data_Type_Name,
    decoding_type: str,
    bin_size_cm: int = 4,
    ext="",
    print_filename: bool = True,
) -> PF_Analysis:
    filename = os.path.join(
        RESULTS_PATH,
        str(data_type),
        f"{session_indicator}_{bin_size_cm}cm_{time_window_ms}ms_pf_analysis_"
        f"{decoding_type}{ext}.obj",
    )
    map_analysis = load_data(filename, print_filename=print_filename)
    return map_analysis
