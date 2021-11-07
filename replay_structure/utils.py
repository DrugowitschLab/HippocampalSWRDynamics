"""
This module contains helper functions used in multiple files throughout the codebase.
"""

import numpy as np
import scipy.stats as sp
from scipy.special import factorial
from scipy.special import gamma
from typing import Optional, Tuple


def calc_poisson_emission_probabilities_log(
    spikemat: np.ndarray, place_fields: np.ndarray, time_window_s: float
) -> np.ndarray:
    """This function calculates the emission probabilities p(x_t|z_t)
    input: spikemat (T x Ncells), place_fields (Ncells x Ngrid), time_window_s
    output: (Ngrid x T matrix) of emission probabilities over time."""
    (n_timesteps, n_cells) = np.shape(spikemat)
    (n_cells, n_grid) = np.shape(place_fields)
    log_pfs = np.log(place_fields).T
    # compute pf_spikes sum: sum_n[x_t*ln[f_n(z_t)*delta_t]]
    pf_spikes_sum = np.zeros((n_timesteps, n_grid))
    for t in range(n_timesteps):
        x = spikemat[t]
        pf_spikes_sum[t] = np.sum(log_pfs * x, axis=1)
    time_window_spikes_sum = np.sum(spikemat * np.log(time_window_s), axis=1)
    # calcualte pf_sum: sum_n[f_n(z_t)*delta_t]]
    pf_sum = time_window_s * np.sum(place_fields, axis=0)
    spikes_sum = np.sum(np.log(factorial(spikemat)), axis=1)
    # calculate emission probs
    pf_sum_norm = (pf_spikes_sum.T + time_window_spikes_sum).T - pf_sum
    emission_probabilities_log = pf_sum_norm.T - spikes_sum
    return emission_probabilities_log


def calc_poisson_emission_probabilities(
    spikemat: np.ndarray, place_fields: np.ndarray, time_window_s: float
) -> np.ndarray:
    emission_probabilities_log = calc_poisson_emission_probabilities_log(
        spikemat, place_fields, time_window_s
    )
    emission_probabilities = np.exp(emission_probabilities_log)
    return emission_probabilities


def calc_poisson_emission_probability_log(
    spikemat: np.ndarray, place_fields: np.ndarray, time_window_s: float
) -> np.ndarray:
    (n_timesteps, n_cells) = np.shape(spikemat)
    (n_cells, n_grid) = np.shape(place_fields)
    x = np.sum(spikemat, axis=0)
    log_pfs = np.log(place_fields).T
    pf_spikes_sum = np.sum(log_pfs * x, axis=1)
    time_window_sum = np.sum(spikemat) * np.log(time_window_s)
    pf_sum = time_window_s * n_timesteps * np.sum(place_fields, axis=0)
    spikes_sum = np.sum(np.log(factorial(spikemat)))
    pf_sum_norm = pf_spikes_sum + time_window_sum - pf_sum
    emission_probability_log = pf_sum_norm - spikes_sum
    return emission_probability_log


def calc_neg_binomial_emission_probabilities_log(
    spikemat: np.ndarray,
    place_fields: np.ndarray,
    time_window_s: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    (n_timesteps, n_cells) = np.shape(spikemat)
    (n_cells, n_grid) = np.shape(place_fields)
    gamma_spikes_sum = np.sum(np.log(gamma(spikemat + alpha)), axis=1)
    spikes_sum = np.sum(np.log(factorial(spikemat)), axis=1)
    gamma_alpha_sum = n_cells * np.log(gamma(alpha))
    p = (place_fields.T * time_window_s) / (place_fields.T * time_window_s + beta)
    alpha_sum = alpha * np.sum(np.log(1 - p), axis=1)
    pf_spikes_sum = np.zeros((n_timesteps, n_grid))
    for t in range(n_timesteps):
        x = spikemat[t]
        pf_spikes_sum[t] = np.sum(np.log(p) * x, axis=1)
    emission_probabilities_log = (
        gamma_spikes_sum - spikes_sum - gamma_alpha_sum + pf_spikes_sum.T
    ).T + alpha_sum.T
    return emission_probabilities_log.T


def calc_neg_binomial_emission_probabilities(
    spikemat: np.ndarray,
    place_fields: np.ndarray,
    time_window_s: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    emission_probabilities_log = calc_neg_binomial_emission_probabilities_log(
        spikemat, place_fields, time_window_s, alpha, beta
    )
    emission_probabilities = np.exp(emission_probabilities_log)
    return emission_probabilities


def calc_neg_binomial_emission_probability_log(
    spikemat: np.ndarray,
    place_fields: np.ndarray,
    time_window_s: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    (n_timesteps, n_cells) = np.shape(spikemat)
    (n_cells, n_grid) = np.shape(place_fields)
    gamma_spikes_sum = np.sum(np.log(gamma(spikemat + alpha)))
    spikes_sum = np.sum(np.log(factorial(spikemat)))
    gamma_alpha_sum = n_timesteps * n_cells * np.log(gamma(alpha))
    p = (place_fields.T * time_window_s) / (place_fields.T * time_window_s + beta)
    alpha_sum = n_timesteps * alpha * np.sum(np.log(1 - p), axis=1)
    pf_spikes_sum = np.zeros((n_timesteps, n_grid))
    x = np.sum(spikemat, axis=0)
    pf_spikes_sum = np.sum(np.log(p) * x, axis=1)
    emission_probability_log = (
        gamma_spikes_sum - spikes_sum - gamma_alpha_sum + pf_spikes_sum.T
    ).T + alpha_sum.T
    return emission_probability_log


def meters_to_bins(array_in_meters, bin_size_cm: int = 4):
    array_in_cm = array_in_meters * 100  # m to cm
    array_in_bins = array_in_cm / bin_size_cm  # cm to bins
    return array_in_bins


def cm_to_bins(array_in_cm, bin_size_cm: int = 4):
    array_in_bins = np.floor(array_in_cm / bin_size_cm)  # cm to bins
    return array_in_bins


def get_pf_matrix(place_fields: np.ndarray, place_cell_ids: np.ndarray) -> np.ndarray:
    place_cell_tuning_curves = place_fields[place_cell_ids]
    place_cell_tuning_curves_flattened = place_cell_tuning_curves.reshape(
        (
            len(place_cell_ids),
            place_cell_tuning_curves.shape[1] * place_cell_tuning_curves.shape[2],
        )
    )
    return place_cell_tuning_curves_flattened


def get_spikemat(
    spike_ids: np.ndarray,
    spike_times: np.ndarray,
    place_cell_ids: np.ndarray,
    start_time: float,
    end_time: float,
    time_window_s: float,
    time_window_advance_s: float,
) -> np.ndarray:
    spikemat = np.empty(shape=(0, len(place_cell_ids)), dtype=int)
    timebin_start_time = start_time
    timebin_end_time = start_time + time_window_s
    while timebin_end_time < end_time:
        spikes_after_start = spike_times >= timebin_start_time
        spikes_before_end = spike_times < timebin_end_time
        timebin_bool = spikes_after_start == spikes_before_end
        spike_ids_in_window = spike_ids[timebin_bool]
        spikevector = np.array(
            [[sum(spike_ids_in_window == cell_id) for cell_id in place_cell_ids]]
        )
        spikemat = np.append(spikemat, spikevector, axis=0)
        timebin_start_time = timebin_start_time + time_window_advance_s
        timebin_end_time = timebin_end_time + time_window_advance_s
    return np.array(spikemat).astype(int)


def get_trajectories(ratday, run_times):
    trajectories = dict()
    for i in range(run_times.shape[0]):
        pos_start_ind = np.argwhere(ratday.data["pos_times_s"] > run_times[i][0])[0][0]
        pos_end_ind = np.argwhere(ratday.data["pos_times_s"] < run_times[i][1])[-1][0]
        trajectories[i] = np.array(
            [
                ratday.data["pos_xy_cm"][pos_start_ind:pos_end_ind, 0],
                ratday.data["pos_xy_cm"][pos_start_ind:pos_end_ind, 1],
            ]
        ).T
    return trajectories


def get_adjusted_parameters(theta, sigma, delta_t):
    n = np.power(10, 10)
    theta_adjusted = np.log(delta_t * theta + 1) / delta_t
    delta = n * delta_t
    continuous_function = (
        sigma ** 2
        / (theta)
        * (
            (2 * theta * delta)
            - np.exp(-2 * theta * delta)
            + 4 * np.exp(-theta * delta)
            - 3
        )
        / (2 * theta ** 2)
    )
    prefactor = delta_t ** 2 / (2 * theta_adjusted)
    numerator = (
        (delta / delta_t) * (-np.exp(2 * theta_adjusted * delta_t))
        - 2 * np.exp(-theta_adjusted * (delta - delta_t))
        - 2 * np.exp(-theta_adjusted * delta)
        + np.exp(-2 * theta_adjusted * delta)
        + 2 * np.exp(theta_adjusted * delta_t)
        + (delta / delta_t)
        + 1
    )
    denominator = (np.exp(theta_adjusted * delta_t) - 1) ** 2
    discrete_function = prefactor * -(numerator / denominator)
    sigma_adjusted = np.sqrt(continuous_function / discrete_function)
    return theta_adjusted, sigma_adjusted


def get_reverse_adjusted_parameters(theta_adjusted, sigma_adjusted, delta_t):
    n = np.power(10, 10)
    theta = (np.exp(delta_t * theta_adjusted) - 1) / delta_t
    delta = n * delta_t
    continuous_function = (
        1
        / (theta)
        * (
            (2 * theta * delta)
            - np.exp(-2 * theta * delta)
            + 4 * np.exp(-theta * delta)
            - 3
        )
        / (2 * theta ** 2)
    )
    prefactor = delta_t ** 2 * sigma_adjusted ** 2 / (2 * theta_adjusted)
    numerator = (
        (delta / delta_t) * (-np.exp(2 * theta_adjusted * delta_t))
        - 2 * np.exp(-theta_adjusted * (delta - delta_t))
        - 2 * np.exp(-theta_adjusted * delta)
        + np.exp(-2 * theta_adjusted * delta)
        + 2 * np.exp(theta_adjusted * delta_t)
        + (delta / delta_t)
        + 1
    )
    denominator = (np.exp(theta_adjusted * delta_t) - 1) ** 2
    discrete_function = prefactor * -(numerator / denominator)
    sigma = np.sqrt(discrete_function / continuous_function)
    return theta, sigma


def boolean_to_times(boolean_array, times):
    """Given a boolean array and an array of times, extract start and end times of where
    boolean_array = True
    """
    start_times = []
    end_times = []
    previous_val = boolean_array[0]
    if previous_val:
        start_times.append(times[0])
    for count, val in enumerate(boolean_array[1:]):
        i = count + 1
        if val != previous_val:
            if val:
                start_times.append(times[i])
            else:
                end_times.append(times[i])
        previous_val = val
    if val:
        end_times.append(times[-1])
    return np.array(start_times), np.array(end_times)


def times_to_bool(data_times, start_time, end_time):
    times_after_start = data_times >= start_time
    times_before_end = data_times <= end_time
    window_ind = times_after_start & times_before_end
    return window_ind


def get_marginal_sum(marginals, n_bins=50):
    if len(marginals.shape) > 1:
        marginal_norm = marginals / np.sum(marginals, axis=0)
        marginal_2d = np.reshape(marginal_norm, (n_bins, n_bins, marginals.shape[1]))
        marginal_plot = np.log(np.nansum(marginal_2d, axis=2))
    else:
        marginal_2d = marginals.reshape(n_bins, n_bins)
        marginal_plot = np.nan_to_num(np.log(marginal_2d))
    return np.nan_to_num(marginal_plot.T)


def get_p_models(log_likelihoods):
    likelihoods_unnormalized = np.exp(log_likelihoods - np.max(log_likelihoods))
    p_models = likelihoods_unnormalized / np.sum(likelihoods_unnormalized)
    return p_models


class LogNorm_Distribution:
    def __init__(self, s: float, loc: float, scale: float):
        self.s = s
        self.loc = loc
        self.scale = scale

    def draw_param_sample(self, param_limits=Optional[Tuple[float, float]]) -> float:
        param_sample = sp.lognorm.rvs(s=self.s, loc=self.loc, scale=self.scale)
        if param_limits is not None:
            while (param_sample < param_limits[0]) or (param_sample > param_limits[1]):
                param_sample = sp.lognorm.rvs(s=self.s, loc=self.loc, scale=self.scale)
        return param_sample


class Norm_Distribution:
    def __init__(self, mean: np.ndarray, sd: np.ndarray):
        self.mean = mean
        self.sd = sd

    def draw_param_sample(self, param_limits=Optional[Tuple[float, float]]) -> float:
        param_sample = sp.norm.rvs(loc=self.mean, scale=self.sd)
        if param_limits is not None:
            while (param_sample < param_limits[0]) or (param_sample > param_limits[1]):
                param_sample = sp.norm.rvs(loc=self.mean, scale=self.sd)
        return param_sample


class InvGamma_Distribution:
    def __init__(self, a: float, loc: float, scale: float):
        self.a = a
        self.loc = loc
        self.scale = scale

    def draw_param_sample(self, param_limits=Optional[Tuple[float, float]]) -> float:
        param_sample = sp.invgamma.rvs(a=self.a, loc=self.loc, scale=self.scale)
        if param_limits is not None:
            while (param_sample < param_limits[0]) or (param_sample > param_limits[1]):
                param_sample = sp.invgamma.rvs(a=self.a, loc=self.loc, scale=self.scale)
        return param_sample


class LogNorm2D_Distribution:
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = mean
        self.cov = cov

    def draw_param_sample(
        self, param_limits=Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> Tuple[float, float]:
        param0_sample, param1_sample = sp.multivariate_normal.rvs(
            mean=self.mean, cov=self.cov
        )

        if param_limits is not None:
            while (
                (param0_sample < param_limits[0][0])
                or (param0_sample > param_limits[0][1])
                or (param1_sample < param_limits[1][0])
                or (param1_sample > param_limits[1][1])
            ):
                param0_sample, param1_sample = sp.multivariate_normal.rvs(
                    mean=self.mean, cov=self.cov
                )
        return np.exp(param0_sample), np.exp(param1_sample)


class Discrete_Distribution_Parameters:
    def __init__(self, param_array: np.ndarray, param_dist: np.ndarray):
        self.param_array = param_array
        self.param_dist = param_dist

    def draw_param_sample(self) -> float:
        param_sample = np.random.choice(self.param_array, p=self.param_dist)
        return param_sample

    def draw_2D_param_sample(self) -> Tuple[float, float]:
        param_sample_ind = np.random.choice(
            np.arange(len(self.param_dist.flatten())), p=self.param_dist.flatten()
        )
        flattened_param_array = self.param_array.reshape((-1, 2))
        return (
            flattened_param_array[param_sample_ind],
            flattened_param_array[param_sample_ind],
        )
