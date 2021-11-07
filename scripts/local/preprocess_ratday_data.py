import os
import click
import scipy.io as spio
from typing import Optional

from replay_structure.read_write import save_ratday_data
from replay_structure.config import RatDay_Preprocessing_Parameters
from replay_structure.ratday_preprocessing import RatDay_Preprocessing
from replay_structure.metadata import (
    DATA_PATH,
    string_to_session_indicator,
    Session_List,
    Session_Name,
)


def load_matlab_struct(file_path: str):
    matlab_struct = spio.loadmat(file_path, squeeze_me=True, struct_as_record=False)
    return matlab_struct


def get_session_data(matlab_struct, session_indicator: Session_Name) -> dict:
    matlab_struct_dict = {
        (1, 1): matlab_struct["Data"].Rat1.Day1,
        (1, 2): matlab_struct["Data"].Rat1.Day2,
        (2, 1): matlab_struct["Data"].Rat2.Day1,
        (2, 2): matlab_struct["Data"].Rat2.Day2,
        (3, 1): matlab_struct["Data"].Rat3.Day1,
        (3, 2): matlab_struct["Data"].Rat3.Day2,
        (4, 1): matlab_struct["Data"].Rat4.Day1,
        (4, 2): matlab_struct["Data"].Rat4.Day2,
    }
    return matlab_struct_dict[(session_indicator.rat, session_indicator.day)]


def run_preprocessing(
    matlab_struct,
    session_indicator: Session_Name,
    bin_size_cm: int,
    rotate_placefields: bool,
    filename_ext: str,
) -> None:
    print(f"Running session {session_indicator} with {bin_size_cm}cm bins")
    session_data = get_session_data(matlab_struct, session_indicator)
    params = RatDay_Preprocessing_Parameters(
        bin_size_cm=bin_size_cm, rotate_placefields=rotate_placefields
    )
    ratday = RatDay_Preprocessing(session_data, params)
    save_ratday_data(
        ratday,
        session_indicator,
        bin_size_cm,
        placefields_rotated=rotate_placefields,
        ext=filename_ext,
    )


@click.command()
@click.option("--session", type=click.INT, default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--filename_ext", default="")
@click.option("--rotate_placefields", type=click.BOOL, default=False)
def main(
    session: Optional[int],
    bin_size_cm: int,
    filename_ext: str,
    rotate_placefields: bool,
):

    # load data
    print("loading data")
    file_path = os.path.join(DATA_PATH, "OpenFieldData.mat")
    matlab_struct = load_matlab_struct(file_path)

    if session is not None:
        session_indicator = string_to_session_indicator(session)
        assert isinstance(session_indicator, Session_Name)
        run_preprocessing(
            matlab_struct,
            session_indicator,
            bin_size_cm,
            rotate_placefields,
            filename_ext,
        )
    else:
        for session_indicator in Session_List:
            assert isinstance(session_indicator, Session_Name)
            run_preprocessing(
                matlab_struct,
                session_indicator,
                bin_size_cm,
                rotate_placefields,
                filename_ext,
            )


if __name__ == "__main__":
    main()
