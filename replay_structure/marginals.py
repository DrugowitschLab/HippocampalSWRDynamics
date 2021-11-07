import replay_structure.structure_models as models

from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.metadata import PLOTTING_FOLDER


class All_Models_Marginals:
    """Calculates the marginals p(z_t|x_1:T) for a single SWR. For models with
    parameters, the parameter settings to use must be provided.
    """

    def __init__(
        self,
        structure_data: Structure_Analysis_Input,
        spikemat_ind: int,
        stationary_gaussian_params: dict = {"sd_meters": 0.01},
        diffusion_params: dict = {"sd_meters": 0.8},
        momentum_params: dict = {"sd_0_meters": 0.03, "sd_meters": 100, "decay": 100},
        plotting_folder=PLOTTING_FOLDER,
    ):
        self.params = {
            "stationary_gaussian": stationary_gaussian_params,
            "diffusion": diffusion_params,
            "momentum": momentum_params,
        }
        self.marginals = self.run_all_models(
            structure_data, spikemat_ind, plotting_folder=plotting_folder
        )

    def run_all_models(
        self,
        structure_data: Structure_Analysis_Input,
        spikemat_ind: int,
        plotting_folder=PLOTTING_FOLDER,
    ) -> dict:
        marginals = dict()
        marginals["diffusion"] = models.Diffusion(
            structure_data, self.params["diffusion"]["sd_meters"]
        ).get_spikemat_marginals(spikemat_ind)

        marginals["momentum"] = models.Momentum(
            structure_data,
            self.params["momentum"]["sd_0_meters"],
            self.params["momentum"]["sd_meters"],
            self.params["momentum"]["decay"],
            plotting=True,
            plotting_folder=plotting_folder,
        ).get_spikemat_marginals(spikemat_ind)

        marginals["stationary"] = models.Stationary(
            structure_data
        ).get_spikemat_marginals(spikemat_ind)

        marginals["stationary_gaussian"] = models.Stationary_Gaussian(
            structure_data, self.params["stationary_gaussian"]["sd_meters"]
        ).get_spikemat_marginals(spikemat_ind)

        marginals["random"] = models.Random(structure_data).get_spikemat_marginals(
            spikemat_ind
        )

        return marginals
