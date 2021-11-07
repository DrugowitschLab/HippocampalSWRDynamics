import click
from typing import Optional, Union

from replay_structure.read_write import (
    load_ratday_data,
    load_spikemat_data,
    save_spikemat_data,
)
from replay_structure.ripple_preprocessing import Ripple_Preprocessing
from replay_structure.highsynchronyevents import HighSynchronyEvents_Preprocessing
from replay_structure.run_snippet_preprocessing import Run_Snippet_Preprocessing
from replay_structure.config import (
    Ripple_Preprocessing_Parameters,
    Run_Snippet_Preprocessing_Parameters,
    HighSynchronyEvents_Preprocessing_Parameters,
)
from replay_structure.metadata import (
    string_to_data_type,
    string_to_session_indicator,
    Session_Name,
    Data_Type,
    Ripples,
    Run_Snippets,
    Ripples_PF,
    PlaceFieldID_Shuffle,
    PlaceField_Rotation,
    HighSynchronyEvents,
    HighSynchronyEvents_PF,
)


def run_spikemat_preprocessing(
    data_type: Data_Type,
    session_indicator: Session_Name,
    bin_size_cm: int,
    time_window_ms: int,
    filename_ext: str,
) -> None:

    print(
        f"running session {session_indicator} with {bin_size_cm}cm bins"
        f"and {time_window_ms}ms time window."
    )
    ratday_data = load_ratday_data(session_indicator, bin_size_cm)

    spikemat_data: Union[
        Ripple_Preprocessing,
        Run_Snippet_Preprocessing,
        HighSynchronyEvents_Preprocessing,
    ]
    if isinstance(data_type.name, Ripples):
        ripple_params = Ripple_Preprocessing_Parameters(
            ratday_data.params, time_window_ms=time_window_ms
        )
        spikemat_data = Ripple_Preprocessing(ratday_data, ripple_params)
    elif isinstance(data_type.name, Run_Snippets):
        run_snippet_params = Run_Snippet_Preprocessing_Parameters(
            ratday_data.params, time_window_ms=time_window_ms
        )
        ripple_time_window_ms_for_selecing_behavior_snippets = 3
        ripple_data = load_spikemat_data(
            session_indicator,
            ripple_time_window_ms_for_selecing_behavior_snippets,
            Ripples(),
            bin_size_cm=bin_size_cm,
            ext="",
        )
        assert isinstance(ripple_data, Ripple_Preprocessing)
        spikemat_data = Run_Snippet_Preprocessing(
            ratday_data, ripple_data, run_snippet_params
        )
    elif isinstance(data_type.name, Ripples_PF):
        ripple_time_window_ms = 3
        ripple_data = load_spikemat_data(
            session_indicator,
            ripple_time_window_ms,
            Ripples(),
            bin_size_cm=bin_size_cm,
            ext=filename_ext,
        )
        assert isinstance(ripple_data, Ripple_Preprocessing)
        popburst_times_s = ripple_data.ripple_info["popburst_times_s"]
        ripple_params = Ripple_Preprocessing_Parameters(
            ratday_data.params, time_window_ms=time_window_ms, time_window_advance_ms=5
        )
        spikemat_data = Ripple_Preprocessing(
            ratday_data, ripple_params, popburst_times_s=popburst_times_s
        )
    elif isinstance(data_type.name, HighSynchronyEvents):
        params = HighSynchronyEvents_Preprocessing_Parameters(
            ratday_data.params, time_window_ms=time_window_ms
        )
        spikemat_data = HighSynchronyEvents_Preprocessing(ratday_data, params)
    elif isinstance(data_type.name, HighSynchronyEvents_PF):
        params = HighSynchronyEvents_Preprocessing_Parameters(
            ratday_data.params, time_window_ms=time_window_ms, time_window_advance_ms=5
        )
        spikemat_data = HighSynchronyEvents_Preprocessing(ratday_data, params)
    elif isinstance(data_type.name, PlaceFieldID_Shuffle):
        ripple_params = Ripple_Preprocessing_Parameters(
            ratday_data.params,
            time_window_ms=time_window_ms,
            shuffle_placefieldIDs=True,
        )
        spikemat_data = Ripple_Preprocessing(ratday_data, ripple_params)
    elif isinstance(data_type.name, PlaceField_Rotation):
        # overwrite regular ratday with place field rotation
        ratday_data = load_ratday_data(
            session_indicator, bin_size_cm, placefields_rotated=True, ext=""
        )
        ripple_params = Ripple_Preprocessing_Parameters(
            ratday_data.params, time_window_ms=time_window_ms
        )
        spikemat_data = Ripple_Preprocessing(ratday_data, ripple_params)

    else:
        raise AttributeError("Invalid data_type.")

    save_spikemat_data(
        spikemat_data,
        session_indicator,
        time_window_ms,
        data_type.name,
        bin_size_cm,
        ext=filename_ext,
    )


@click.command()
@click.option(
    "--data_type",
    type=click.Choice(
        [
            "ripples",
            "run_snippets",
            "ripples_pf",
            "high_synchrony_events_pf",
            "placefieldID_shuffle",
            "placefield_rotation",
            "high_synchrony_events",
        ]
    ),
    required=True,
)
@click.option("--session", type=click.INT, default=None)
@click.option("--bin_size_cm", type=click.INT, default=4)
@click.option("--time_window_ms", type=click.INT, default=None)
@click.option("--filename_ext", default="")
def main(
    data_type: str,
    session: Optional[int],
    bin_size_cm: int,
    time_window_ms: Optional[int],
    filename_ext: str,
):
    data_type_ = string_to_data_type(data_type)
    if time_window_ms is None:
        time_window_ms = data_type_.default_time_window_ms

    if session is not None:
        session_indicator = string_to_session_indicator(session)
        assert isinstance(session_indicator, Session_Name)
        run_spikemat_preprocessing(
            data_type_, session_indicator, bin_size_cm, time_window_ms, filename_ext
        )
    else:
        for session_indicator in data_type_.session_list:
            assert isinstance(session_indicator, Session_Name)
            run_spikemat_preprocessing(
                data_type_, session_indicator, bin_size_cm, time_window_ms, filename_ext
            )


if __name__ == "__main__":
    main()
