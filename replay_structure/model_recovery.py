"""
Module for generating simulated spatial trajectories according to the dynamics of each
model to be used for the model recovery analysis.
"""

from typing import Union, List, Tuple
import numpy as np

from replay_structure.simulated_trajectories import (
    Simulated_Trajectory,
    Simulated_Trajectory_Parameters,
    Simulated_Trajectory_Model_Parameters,
    Diffusion_Model_Parameters,
    Momentum_Model_Parameters,
    Gaussian_Model_Parameters,
    Stationary_Model_Parameters,
    Random_Model_Parameters,
)
from replay_structure.utils import (
    LogNorm_Distribution,
    InvGamma_Distribution,
    LogNorm2D_Distribution,
)


class Diffusion_Model_Parameter_Prior:
    """Class for drawing parameters for the diffusion model from a prior distribution
    fit to the real data."""

    def __init__(self, sd_meters_dist: InvGamma_Distribution):
        self.sd_meters_dist = sd_meters_dist
        self.sd_meters_limit: Tuple[float, float] = (0.01, 5.5)

    def draw_params(self) -> Diffusion_Model_Parameters:
        sd_meters = self.sd_meters_dist.draw_param_sample(
            param_limits=self.sd_meters_limit
        )
        return Diffusion_Model_Parameters(sd_meters=sd_meters)


class Momentum_Model_Parameter_Prior:
    """Class for drawing parameters for the momentum model from a prior distribution
    fit to the real data."""

    def __init__(self, sd_decay_meters_dist: LogNorm2D_Distribution):
        self.sd_decay_meters_dist = sd_decay_meters_dist
        self.sd_meters_limit: Tuple[float, float] = (0.001, 600)
        self.decay_limit: Tuple[float, float] = (0, 700)

    def draw_params(self) -> Momentum_Model_Parameters:
        sd_meters, decay = self.sd_decay_meters_dist.draw_param_sample(
            param_limits=(self.sd_meters_limit, self.decay_limit)
        )
        return Momentum_Model_Parameters(sd_meters=sd_meters, decay=decay)


class Gaussian_Model_Parameter_Prior:
    """Class for drawing parameters for the Gaussian model from a prior distribution
    fit to the real data."""

    def __init__(self, sd_meters_dist: InvGamma_Distribution):
        self.sd_meters_dist = sd_meters_dist
        self.sd_meters_limit: Tuple[float, float] = (0.01, 1)

    def draw_params(self) -> Gaussian_Model_Parameters:
        sd_meters = self.sd_meters_dist.draw_param_sample(
            param_limits=self.sd_meters_limit
        )
        return Gaussian_Model_Parameters(sd_meters=sd_meters)


class Stationary_Model_Parameter_Prior:
    def __init__(self):
        pass

    def draw_params(self) -> Stationary_Model_Parameters:
        return Stationary_Model_Parameters()


class Random_Model_Parameter_Prior:
    def __init__(self):
        pass

    def draw_params(self) -> Random_Model_Parameters:
        return Random_Model_Parameters()


Model_Parameter_Distribution_Prior = Union[
    Diffusion_Model_Parameter_Prior,
    Momentum_Model_Parameter_Prior,
    Gaussian_Model_Parameter_Prior,
    Stationary_Model_Parameter_Prior,
    Random_Model_Parameter_Prior,
]


class Model_Recovery_Trajectory_Set_Parameters:
    """Trajectory set paramters for model recovery."""

    def __init__(
        self,
        model_parameter_dist: Model_Parameter_Distribution_Prior,
        duration_s_dist: LogNorm_Distribution,
        n_trajectories: int = 100,
        time_window_s: float = 0.001,
        arena_length_cm: float = 200,
        random_seed=0,  # 35
    ):
        self.model_params_dist = model_parameter_dist
        self.duration_s_dist = duration_s_dist
        self.duration_s_dist_limit: Tuple[float, float] = (0, 1)
        self.n_trajectories = n_trajectories
        self.time_window_s = time_window_s
        self.arena_length_cm = arena_length_cm
        self.random_seed = random_seed

    def get_trajectory_params(self) -> Simulated_Trajectory_Parameters:
        duration_s = self.duration_s_dist.draw_param_sample(
            param_limits=self.duration_s_dist_limit
        )
        return Simulated_Trajectory_Parameters(
            duration_s=duration_s,
            arena_length_cm=self.arena_length_cm,
            time_window_s=self.time_window_s,
        )

    def get_model_params(self) -> Simulated_Trajectory_Model_Parameters:
        model_params = self.model_params_dist.draw_params()
        return model_params


class Model_Recovery_Trajectory_Set:
    """Class for generating a simulated trajectory set to be used for model recovery."""

    def __init__(self, trajectory_set_params: Model_Recovery_Trajectory_Set_Parameters):
        self.trajectory_set_params = trajectory_set_params
        self.trajectory_set: List[Simulated_Trajectory] = self.generate_trajectory_set(
            self.trajectory_set_params
        )

    @staticmethod
    def generate_trajectory_set(
        trajectory_set_params: Model_Recovery_Trajectory_Set_Parameters,
    ) -> List[Simulated_Trajectory]:
        trajectory_set: List[Simulated_Trajectory] = list()
        np.random.seed(trajectory_set_params.random_seed)
        for i in range(trajectory_set_params.n_trajectories):
            trajectory_params = trajectory_set_params.get_trajectory_params()
            model_params = trajectory_set_params.get_model_params()
            trajectory_attempt: Simulated_Trajectory = Simulated_Trajectory(
                trajectory_params, model_params
            )
            counter = 0
            while trajectory_attempt.trajectory_cm is None:
                if counter > 1000:
                    model_params = trajectory_set_params.get_model_params()
                trajectory_attempt = Simulated_Trajectory(
                    trajectory_params, model_params
                )
                counter += 1

            trajectory_set.append(trajectory_attempt)
        return trajectory_set
