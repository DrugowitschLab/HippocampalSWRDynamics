import numpy as np
import scipy.stats as sp

from typing import Optional, NamedTuple, Union


class Diffusion_Model_Parameters(NamedTuple):
    sd_meters: float


class Momentum_Model_Parameters(NamedTuple):
    sd_meters: float
    decay: float
    SD_0_METERS: float = 1


class Gaussian_Model_Parameters(NamedTuple):
    sd_meters: float


class Stationary_Model_Parameters(NamedTuple):
    pass


class Random_Model_Parameters(NamedTuple):
    pass


Simulated_Trajectory_Model_Parameters = Union[
    Diffusion_Model_Parameters,
    Momentum_Model_Parameters,
    Gaussian_Model_Parameters,
    Stationary_Model_Parameters,
    Random_Model_Parameters,
]


class Simulated_Trajectory_Parameters(NamedTuple):
    duration_s: float
    time_window_s: float
    arena_length_cm: float


class Simulated_Trajectory:
    """Class for generating a simulated trajectory. Methods implement generating
    a trajectory according to the dynamcis of each model.
    """

    def __init__(
        self,
        trajectory_parameters: Simulated_Trajectory_Parameters,
        model_parameters: Simulated_Trajectory_Model_Parameters,
    ):
        self.trajectory_params = trajectory_parameters
        self.model_parameters = model_parameters
        self.trajectory_cm = self.generate_trajectory(
            self.trajectory_params, self.model_parameters
        )

    @staticmethod
    def generate_trajectory(
        trajectory_parameters: Simulated_Trajectory_Parameters,
        model_parameters: Simulated_Trajectory_Model_Parameters,
    ) -> np.ndarray:
        if isinstance(model_parameters, Diffusion_Model_Parameters):
            trajectory = Simulated_Trajectory.generate_diffusion_trajectory(
                trajectory_parameters, model_parameters
            )
        elif isinstance(model_parameters, Momentum_Model_Parameters):
            trajectory = Simulated_Trajectory.generate_momentum_trajectory(
                trajectory_parameters, model_parameters
            )
        elif isinstance(model_parameters, Stationary_Model_Parameters):
            trajectory = Simulated_Trajectory.generate_stationary_trajectory(
                trajectory_parameters
            )
        elif isinstance(model_parameters, Gaussian_Model_Parameters):
            trajectory = Simulated_Trajectory.generate_gaussian_trajectory(
                trajectory_parameters, model_parameters
            )
        elif isinstance(model_parameters, Random_Model_Parameters):
            trajectory = Simulated_Trajectory.generate_random_trajectory(
                trajectory_parameters
            )
        else:
            raise Exception("Invalid model parameters.")
        return trajectory

    @staticmethod
    def generate_diffusion_trajectory(
        trajectory_parameters: Simulated_Trajectory_Parameters,
        model_parameters: Diffusion_Model_Parameters,
    ) -> Optional[np.ndarray]:
        sd_cm = model_parameters.sd_meters * 100
        n_timesteps = int(
            trajectory_parameters.duration_s / trajectory_parameters.time_window_s
        )
        trajectory = np.zeros((n_timesteps, 2))
        start_position = np.random.randint(
            trajectory_parameters.arena_length_cm, size=2
        )
        trajectory[0] = start_position
        for i in range(1, n_timesteps):
            next_position = sp.multivariate_normal.rvs(
                trajectory[i - 1], sd_cm ** 2 * trajectory_parameters.time_window_s
            )
            counter = 0
            while np.any(
                next_position > trajectory_parameters.arena_length_cm
            ) or np.any(next_position < 0):
                next_position = sp.multivariate_normal.rvs(
                    trajectory[i - 1], sd_cm ** 2 * trajectory_parameters.time_window_s
                )
                counter += 1
                if counter > 20:
                    print()
                    return None
            trajectory[i] = next_position
        return trajectory

    @staticmethod
    def generate_momentum_trajectory(
        trajectory_parameters: Simulated_Trajectory_Parameters,
        model_parameters: Momentum_Model_Parameters,
    ) -> Optional[np.ndarray]:
        sd_0_cm = model_parameters.SD_0_METERS * 100
        sd_cm = model_parameters.sd_meters * 100
        decay = model_parameters.decay
        n_timesteps = int(
            trajectory_parameters.duration_s / trajectory_parameters.time_window_s
        )
        trajectory = np.zeros((n_timesteps, 2))
        position_0 = np.random.randint(trajectory_parameters.arena_length_cm, size=2)
        trajectory[0] = position_0
        position_1 = sp.multivariate_normal.rvs(
            position_0, sd_0_cm ** 2 * trajectory_parameters.time_window_s
        )
        while np.any(position_1 > trajectory_parameters.arena_length_cm) or np.any(
            position_1 < 0
        ):
            position_1 = sp.multivariate_normal.rvs(
                position_0, sd_0_cm ** 2 * trajectory_parameters.time_window_s
            )
        trajectory[1] = position_1
        for i in range(2, n_timesteps):
            mean = (
                (1 + np.exp(-trajectory_parameters.time_window_s * decay))
                * trajectory[i - 1]
                - np.exp(-trajectory_parameters.time_window_s * decay)
                * trajectory[i - 2]
            )
            mean = (2 - trajectory_parameters.time_window_s * decay) * trajectory[
                i - 1
            ] - (1 - trajectory_parameters.time_window_s * decay) * trajectory[i - 2]
            var = sd_cm ** 2 * (trajectory_parameters.time_window_s ** 3)
            next_position = sp.multivariate_normal.rvs(mean, var)
            counter = 0
            while np.any(
                next_position > trajectory_parameters.arena_length_cm
            ) or np.any(next_position < 0):
                next_position = sp.multivariate_normal.rvs(mean, var)
                counter += 1
                if counter > 200:
                    return None
            trajectory[i] = next_position
        return trajectory

    @staticmethod
    def generate_stationary_trajectory(
        trajectory_parameters: Simulated_Trajectory_Parameters,
    ) -> np.ndarray:
        n_timesteps = int(
            trajectory_parameters.duration_s / trajectory_parameters.time_window_s
        )
        position = np.random.randint(trajectory_parameters.arena_length_cm, size=2)
        trajectory = np.tile(position, (n_timesteps, 1))
        return trajectory

    @staticmethod
    def generate_gaussian_trajectory(
        trajectory_parameters: Simulated_Trajectory_Parameters,
        model_parameters: Gaussian_Model_Parameters,
    ) -> Optional[np.ndarray]:
        sd_cm = model_parameters.sd_meters * 100
        cov = [[sd_cm ** 2, 0], [0, sd_cm ** 2]]
        n_timesteps = int(
            trajectory_parameters.duration_s / trajectory_parameters.time_window_s
        )
        mean = np.random.randint(trajectory_parameters.arena_length_cm, size=2)
        trajectory = sp.multivariate_normal.rvs(mean, cov, size=(n_timesteps))
        out_of_bounds = np.where(
            (trajectory > trajectory_parameters.arena_length_cm) + (trajectory < 0)
        )[0]
        if len(out_of_bounds) > 0:
            for i in out_of_bounds:
                new_point = sp.multivariate_normal.rvs(mean, cov, size=1)
                counter = 0
                while np.any(
                    new_point > trajectory_parameters.arena_length_cm
                ) or np.any(new_point < 0):
                    new_point = sp.multivariate_normal.rvs(mean, cov, size=1)
                    counter += 1
                    if counter > 20:
                        return None
                trajectory[i] = new_point
        return trajectory

    @staticmethod
    def generate_random_trajectory(
        trajectory_parameters: Simulated_Trajectory_Parameters,
    ) -> np.ndarray:
        n_timesteps = int(
            trajectory_parameters.duration_s / trajectory_parameters.time_window_s
        )
        trajectory = np.random.randint(
            trajectory_parameters.arena_length_cm, size=(n_timesteps, 2)
        )
        return trajectory
