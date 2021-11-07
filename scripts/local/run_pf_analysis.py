import click
from typing import Optional

from replay_structure.pf_analysis import PF_Analysis
from replay_structure.metadata import (
    Data_Type,
    string_to_data_type,
    string_to_session_indicator,
    Session_Indicator,
)
from replay_structure.read_write import load_structure_data, save_pf_analysis


def run_map_preprocessing(
    session: Session_Indicator,
    bin_size_cm: int,
    time_window_ms: int,
    data_type: Data_Type,
    decoding_type: str,
    filename_ext: str,
):
    print(f"running session {session} with {bin_size_cm}cm bins")

    pf_data = load_structure_data(
        session,
        time_window_ms,
        data_type.name,
        data_type.default_likelihood_function,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )

    if pf_data is None:
        print(f"no data for: {session}")
        map_results = None
    else:
        map_results = PF_Analysis(
            pf_data, decoding_type=decoding_type, save_only_trajectories=False
        )
    save_pf_analysis(
        session,
        time_window_ms,
        data_type.name,
        map_results,
        decoding_type,
        bin_size_cm=bin_size_cm,
        ext=filename_ext,
    )


@click.command()
@click.option("--session", type=click.INT, default=None)
@click.option(
    "--data_type",
    type=click.Choice(["ripples_pf", "high_synchrony_events_pf"]),
    required=True,
)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--filename_ext", type=click.STRING, default="")
@click.option("--decoding_type", type=click.Choice(["map", "mean"]), default="map")
def main(
    session: Optional[int],
    data_type: str,
    bin_size_cm: int,
    filename_ext: str,
    decoding_type: str,
):
    data_type_ = string_to_data_type(data_type)
    time_window_ms = data_type_.default_time_window_ms

    if session is not None:
        session_indicator: Session_Indicator = string_to_session_indicator(session)
        run_map_preprocessing(
            session_indicator,
            bin_size_cm,
            time_window_ms,
            data_type_,
            decoding_type,
            filename_ext,
        )
    else:
        for session_indicator in data_type_.session_list:
            run_map_preprocessing(
                session_indicator,
                bin_size_cm,
                time_window_ms,
                data_type_,
                decoding_type,
                filename_ext,
            )


if __name__ == "__main__":
    main()
