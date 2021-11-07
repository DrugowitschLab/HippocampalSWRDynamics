import os
import shutil
import click
from typing import Optional

from replay_structure.structure_analysis_input import Structure_Analysis_Input
from replay_structure.marginals import All_Models_Marginals
from replay_structure.structure_models import Diffusion
from replay_structure.metadata import (
    PLOTTING_FOLDER,
    string_to_data_type,
    string_to_session_indicator,
    Session_Indicator,
    Data_Type,
)
from replay_structure.read_write import (
    load_structure_data,
    save_marginals,
    save_diffusion_marginals,
)

SG_PARAMS = dict()
DIFFUSION_PARAMS = dict()
MOMENTUM_PARAMS = dict()

# for ripples
SG_PARAMS["ripples"] = {"sd_meters": 0.06}
DIFFUSION_PARAMS["ripples"] = {"sd_meters": 0.98}
MOMENTUM_PARAMS["ripples"] = {"sd_meters": 130, "decay": 100, "sd_0_meters": 0.03}

# for run snippets
SG_PARAMS["run_snippets"] = {"sd_meters": 0.1}
DIFFUSION_PARAMS["run_snippets"] = {"sd_meters": 0.14}
MOMENTUM_PARAMS["run_snippets"] = {"sd_meters": 2.4, "decay": 20, "sd_0_meters": 0.03}


def get_marginals(
    session: Session_Indicator,
    structure_data: Structure_Analysis_Input,
    spikemat_ind: int,
    bin_size_cm: int,
    time_window_ms,
    data_type: Data_Type,
    filename_ext: str,
):
    print(f"running spikemat {spikemat_ind}")
    plotting_folder = os.path.join(PLOTTING_FOLDER, f"{session}spikemat{spikemat_ind}")
    if not os.path.exists(plotting_folder):
        os.mkdir(plotting_folder)
    marginals = All_Models_Marginals(
        structure_data,
        spikemat_ind,
        stationary_gaussian_params=SG_PARAMS[str(data_type.name)],
        diffusion_params=DIFFUSION_PARAMS[str(data_type.name)],
        momentum_params=MOMENTUM_PARAMS[str(data_type.name)],
        plotting_folder=plotting_folder,
    )
    print(marginals.marginals["momentum"].sum())
    shutil.rmtree(plotting_folder)
    save_marginals(
        session,
        spikemat_ind,
        time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        marginals,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


def get_diffusion_marginals(
    session: Session_Indicator,
    structure_data: Structure_Analysis_Input,
    bin_size_cm: int,
    time_window_ms,
    data_type: Data_Type,
    filename_ext: str,
):
    sd_meters = DIFFUSION_PARAMS[str(data_type.name)]["sd_meters"]
    print(sd_meters)
    marginals = Diffusion(structure_data, sd_meters).get_marginals()
    save_diffusion_marginals(
        session,
        time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        marginals,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


@click.command()
@click.option(
    "--data_type", type=click.Choice(["ripples", "run_snippets"]), default="ripples"
)
@click.option("--session", default="0")
@click.option("--spikemat_ind", type=click.INT, default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option("--diffusion_only", is_flag=True)
@click.option("--filename_ext", type=click.STRING, default="")
def main(
    data_type: str,
    session: str,
    spikemat_ind: Optional[int],
    bin_size_cm: int,
    time_window_ms: int,
    diffusion_only: bool,
    filename_ext: str,
):
    data_type_ = string_to_data_type(data_type)
    session_indicator: Session_Indicator = string_to_session_indicator(session)
    structure_data = load_structure_data(
        session_indicator,
        data_type_.default_time_window_ms,
        data_type_.name,
        data_type_.default_likelihood_function,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )
    if time_window_ms is None:
        time_window_ms = data_type_.default_time_window_ms

    if not diffusion_only:
        if spikemat_ind is not None:
            get_marginals(
                session_indicator,
                structure_data,
                spikemat_ind,
                bin_size_cm,
                time_window_ms,
                data_type_,
                filename_ext,
            )
        else:
            for spikemat_ind in range(len(structure_data.spikemats)):
                get_marginals(
                    session_indicator,
                    structure_data,
                    spikemat_ind,
                    bin_size_cm,
                    time_window_ms,
                    data_type_,
                    filename_ext,
                )

    else:
        get_diffusion_marginals(
            session_indicator,
            structure_data,
            bin_size_cm,
            time_window_ms,
            data_type_,
            filename_ext,
        )


if __name__ == "__main__":
    main()
