import numpy as np
import scipy.stats as sp


class Diffusion_Constant:
    """Implementation of the method from Stella et al. (2019) for calculating the
    diffusion constant from decoded spatial trajectories."""

    def __init__(self, trajectories: dict, bin_size_cm: int = 4, n_time_windows=30):
        self.bin_size_cm = bin_size_cm
        self.n_time_windows = n_time_windows
        self.diffusion_constant_info = self.run_analysis(trajectories)
        self.alpha = self.diffusion_constant_info["fit_params"]["slope"]
        self.bootstrap_alpha_dist = self.run_bootstraps(trajectories)

    def run_analysis(self, trajectories: dict):
        diffusion_constant_info = dict()
        diffusion_constant_info["distance_data"] = self.get_distance_data(trajectories)
        diffusion_constant_info["distance_by_time"] = self.calc_distance_by_time(
            diffusion_constant_info["distance_data"]
        )
        diffusion_constant_info["fit_params"] = self.fit_linear_regression(
            diffusion_constant_info["distance_by_time"]
        )
        return diffusion_constant_info

    def get_distance_data(self, trajectories: dict):
        """Calculate distance between points for each delta t."""
        distance_data = dict()
        for t in range(1, self.n_time_windows + 1):
            distance_data[t] = np.array([])
            for ripple_num in trajectories:
                if trajectories[ripple_num] is not None:
                    start_point_set = trajectories[ripple_num][:-t]
                    end_point_set = trajectories[ripple_num][t:]
                    distance_squared = ((start_point_set - end_point_set) ** 2).sum(
                        axis=1
                    )
                    distance_data[t] = np.append(distance_data[t], distance_squared)
        return distance_data

    def calc_distance_by_time(self, distance_data: dict):
        distance_by_time = np.zeros(self.n_time_windows)
        for t in range(1, self.n_time_windows + 1):
            distance_by_time[t - 1] = np.sqrt(np.mean(distance_data[t]))
        return distance_by_time

    def fit_linear_regression(self, distance_by_time: np.ndarray) -> dict:
        have_value = ~np.isnan(distance_by_time)
        slope, intercept, r_value, p_value, std_err = sp.stats.linregress(
            np.log(np.arange(1, have_value.sum() + 1)),
            np.log(distance_by_time[have_value]),
        )
        return {"slope": slope, "intercept": intercept}

    def run_bootstraps(self, trajectories: dict, n_bootstraps=1000):
        np.random.seed(0)
        alpha_dist = np.zeros(n_bootstraps)
        n_trajectories = len(trajectories)
        for i in range(n_bootstraps):
            trajectory_sample_inds = np.random.choice(
                n_trajectories, size=n_trajectories, replace=True
            )
            trajectory_samples = {i: trajectories[i] for i in trajectory_sample_inds}
            sample_results = self.run_analysis(trajectory_samples)
            alpha_dist[i] = sample_results["fit_params"]["slope"]
        print((alpha_dist > 0.5).sum() / 1000)
        return alpha_dist
