"""
This file defines the parameters used at each step in the analysis pipeline.
"""

import numpy as np
from typing import Optional, Dict

from replay_structure.metadata import Likelihood_Function_Params  # Poisson_Params

# These constants are calculated across sessions in jupyter notebook (FigureS3.ipynb)
PF_SCALING_FACTOR = 2.9
VELOCITY_SCALING_FACTOR = 6.75
MAX_LIKELIHOOD_SD_METERS_RIPPLES = (
    0.74
)  # mode from diffusion gridsearch on all sessions
MAX_LIKELIHOOD_SD_METERS_RUN_SNIPPETS = (
    0.14
)  # mode from diffusion gridsearch on all sessions


class RatDay_Preprocessing_Parameters:
    """Defines parameters for preprocessing data from Pfeiffer & Foster (2013,2015).

    Parameters that are properties of the experiment are defined as class variables,
    while parameters that chosen for analysis are defined as instance variables.
    """

    ENVIRONMENT_WIDTH_CM: float = 200
    ENVIRONMENT_LENGTH_CM: float = 200
    POSITION_RECORDING_RESOLUTION_FRAMES_PER_S: float = 1 / 30

    def __init__(
        self,
        bin_size_cm: int = 4,
        position_recording_gap_threshold_s: float = 0.25,
        velocity_run_threshold_cm_per_s: float = 5,
        place_field_gaussian_sd_cm: float = 4,
        place_field_prior_mean_firing_rate_spikes_per_s: float = 1.0,
        place_field_prior_beta_s: float = 0.01,
        inhibitory_firing_rate_threshold_spikes_per_s: float = 10,
        place_field_minimum_tuning_curve_peak_spikes_per_s: float = 2,
        rotate_placefields: bool = False,
    ):
        self.bin_size_cm = bin_size_cm
        self.position_recording_gap_threshold_s = position_recording_gap_threshold_s
        self.velocity_run_threshold_cm_per_s = velocity_run_threshold_cm_per_s
        self.place_field_gaussian_sd_cm = place_field_gaussian_sd_cm
        self.place_field_prior_mean_firing_rate_spikes_per_s = (
            place_field_prior_mean_firing_rate_spikes_per_s
        )
        self.place_field_prior_beta_s = place_field_prior_beta_s
        self.place_field_prior_alpha_s: float = (
            place_field_prior_beta_s * place_field_prior_mean_firing_rate_spikes_per_s
            + 1
        )
        self.inhibitory_firing_rate_threshold_spikes_per_s = (
            inhibitory_firing_rate_threshold_spikes_per_s
        )
        self.place_field_minimum_tuning_curve_peak_spikes_per_s = (
            place_field_minimum_tuning_curve_peak_spikes_per_s
        )
        self.n_bins_x: int = int(self.ENVIRONMENT_WIDTH_CM / bin_size_cm)
        self.n_bins_y: int = int(self.ENVIRONMENT_LENGTH_CM / bin_size_cm)
        self.rotate_placefields = rotate_placefields


class Ripple_Preprocessing_Parameters:
    """Defines parameters for preprocessing neural data during SWRs for spatio-temporal
    structure analysis.
    """

    def __init__(
        self,
        ratday_preprocessing_params: RatDay_Preprocessing_Parameters,
        time_window_ms: int = 3,
        time_window_advance_ms: Optional[int] = None,
        select_population_burst: bool = True,
        avg_fr_smoothing_convolution: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25]),
        popburst_avg_spikes_per_s_threshold: float = 2,
        min_popburst_duration_ms: int = 30,
        shuffle_placefieldIDs: bool = False,
    ):
        self.ratday_preprocessing_params = ratday_preprocessing_params
        self.time_window_ms = time_window_ms
        if time_window_advance_ms is None:
            self.time_window_advance_ms = time_window_ms
        else:
            self.time_window_advance_ms = time_window_advance_ms
        self.time_window_s: float = self.time_window_ms / 1000
        self.time_window_advance_s: float = self.time_window_advance_ms / 1000
        self.select_population_burst = select_population_burst
        self.avg_fr_smoothing_convolution = avg_fr_smoothing_convolution
        self.popburst_avg_spikes_per_s_threshold = popburst_avg_spikes_per_s_threshold
        self.min_popburst_duration_ms = min_popburst_duration_ms
        self.min_popburst_n_time_windows = np.ceil(
            self.min_popburst_duration_ms / self.time_window_ms
        ).astype(int)
        # extract common parameters from ratday for easier use
        self.n_bins_x = self.ratday_preprocessing_params.n_bins_x
        self.n_bins_y = self.ratday_preprocessing_params.n_bins_y
        self.bin_size_cm = self.ratday_preprocessing_params.bin_size_cm
        self.shuffle_placefieldIDs = shuffle_placefieldIDs


class HighSynchronyEvents_Preprocessing_Parameters:
    """Defines parameters for preprocessing neural data during HSEs for structure
    analysis.
    """

    def __init__(
        self,
        ratday_preprocessing_params: RatDay_Preprocessing_Parameters,
        time_window_ms: int = 3,
        time_window_advance_ms: Optional[int] = None,
        select_population_burst: bool = True,
        avg_fr_smoothing_convolution: np.ndarray = np.array([0.25, 0.25, 0.25, 0.25]),
        popburst_avg_spikes_per_s_threshold: float = 2,
        min_popburst_duration_ms: int = 30,
        min_hse_duration_s: float = 0.05,
    ):
        self.ratday_preprocessing_params = ratday_preprocessing_params
        self.time_window_ms = time_window_ms
        if time_window_advance_ms is None:
            self.time_window_advance_ms = time_window_ms
        else:
            self.time_window_advance_ms = time_window_advance_ms
        self.time_window_s: float = self.time_window_ms / 1000
        self.time_window_advance_s: float = self.time_window_advance_ms / 1000
        self.select_population_burst = select_population_burst
        self.avg_fr_smoothing_convolution = avg_fr_smoothing_convolution
        self.popburst_avg_spikes_per_s_threshold = popburst_avg_spikes_per_s_threshold
        self.min_popburst_duration_ms = min_popburst_duration_ms
        self.min_popburst_n_time_windows = np.ceil(
            self.min_popburst_duration_ms / self.time_window_ms
        ).astype(int)
        # extract common parameters for easier use
        self.n_bins_x = self.ratday_preprocessing_params.n_bins_x
        self.n_bins_y = self.ratday_preprocessing_params.n_bins_y
        self.bin_size_cm = self.ratday_preprocessing_params.bin_size_cm
        self.min_hse_duration_s = min_hse_duration_s


class Run_Snippet_Preprocessing_Parameters:
    """Defines parameters for preprocessing neural data during run snippets for
    structure analysis.
    """

    def __init__(
        self,
        ratday_preprocessing_params: RatDay_Preprocessing_Parameters,
        time_window_ms: int = 60,
        time_window_advance_ms: Optional[int] = None,
        run_period_threshold_s: float = 2,
        random_seed=0,
        place_field_scaling_factor=PF_SCALING_FACTOR,
        velocity_scaling_factor=VELOCITY_SCALING_FACTOR,
    ):
        self.ratday_preprocessing_params = ratday_preprocessing_params
        self.time_window_ms = time_window_ms
        if time_window_advance_ms is None:
            self.time_window_advance_ms = time_window_ms
        else:
            self.time_window_advance_ms = time_window_advance_ms
        self.time_window_s: float = self.time_window_ms / 1000
        self.time_window_advance_s: float = self.time_window_advance_ms / 1000
        self.run_period_threshold_s = run_period_threshold_s
        self.random_seed = random_seed
        self.place_field_scaling_factor = place_field_scaling_factor
        self.velocity_scaling_factor = velocity_scaling_factor
        self.duration_scaling_factor = (
            self.velocity_scaling_factor * self.place_field_scaling_factor
        )
        # extract common parameters for easier use
        self.n_bins_x = self.ratday_preprocessing_params.n_bins_x
        self.n_bins_y = self.ratday_preprocessing_params.n_bins_y
        self.bin_size_cm = self.ratday_preprocessing_params.bin_size_cm


class Structure_Analysis_Input_Parameters:
    """Defines parameters for bringing preprocessed SWR, HSE, run snippets, and
    simulated SWR data into a consistent format for running structure models.
    """

    def __init__(
        self,
        likelihood_function_params: Likelihood_Function_Params,
        time_window_ms: int,
        bin_size_cm: int = 4,
        n_bins_x: int = 50,
        n_bins_y: int = 50,
        time_window_advance_ms: Optional[int] = None,
    ):
        self.bin_size_cm = bin_size_cm
        self.n_bins_x = n_bins_x
        self.n_bins_y = n_bins_y
        self.n_grid = n_bins_x * n_bins_y
        self.time_window_ms = time_window_ms
        self.time_window_s = time_window_ms / 1000
        self.time_window_advance_ms = time_window_advance_ms
        self.time_window_advance_s = (
            time_window_advance_ms / 1000
            if time_window_advance_ms is not None
            else None
        )
        self.likelihood_function_params = likelihood_function_params


class Structure_Model_Gridsearch_Parameters:
    """Defines the gridsearch parameters for running models with parameters."""

    def __init__(self, params: Dict[str, np.ndarray]):
        self.params = params

    @classmethod
    def ripple_diffusion_params(cls, sd_array_meters=np.logspace(-1, 0.8, 30).round(2)):
        params = {"sd_array_meters": sd_array_meters}
        return cls(params)

    @classmethod
    def ripple_momentum_params(
        cls,
        sd_array_meters=np.logspace(1.6, 2.6, 30).round(2),
        decay_array=np.array([1, 25, 50, 75, 100, 200, 300, 400, 500, 800]),
        initial_sd_m_per_s=10,
    ):
        params = {
            "sd_array_meters": sd_array_meters,
            "decay_array": decay_array,
            "initial_sd_m_per_s": initial_sd_m_per_s,
        }
        return cls(params)

    @classmethod
    def ripple_stationary_gaussian_params(
        cls, sd_array_meters=np.logspace(-2, 0.3, 30).round(2)
    ):
        params = {"sd_array_meters": sd_array_meters}
        return cls(params)

    @classmethod
    def run_diffusion_params(cls, sd_array_meters=np.logspace(-1.8, 0, 30)):
        params = {"sd_array_meters": sd_array_meters}
        return cls(params)

    @classmethod
    def run_momentum_params(
        cls,
        sd_array_meters=np.logspace(-0.3, 2.4, 25),
        decay_array=np.array([1, 10, 20, 40, 80, 120, 200, 400, 800, 1200, 2000, 4000]),
        initial_sd_m_per_s=5,
    ):
        params = {
            "sd_array_meters": sd_array_meters,
            "decay_array": decay_array,
            "initial_sd_m_per_s": initial_sd_m_per_s,
        }
        return cls(params)

    @classmethod
    def run_stationary_gaussian_params(cls, sd_array_meters=np.logspace(-2, 0.3, 30)):
        params = {"sd_array_meters": sd_array_meters}
        return cls(params)
