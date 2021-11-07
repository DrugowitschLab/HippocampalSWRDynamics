import sys
from replay_structure.metadata import (
    string_to_likelihood_function,
    string_to_session_indicator,
    string_to_data_type,
    MODELS_AS_STR,
)
from scripts.o2.o2_lib import submit_diffusion_gridsearch

session = str(sys.argv[1])
session_type = str(sys.argv[2])
data_type = str(sys.argv[3])
time_window_ms = int(sys.argv[4])
likelihood_function = str(sys.argv[5])

if session_type == "simulated":
    session = MODELS_AS_STR[int(session)]

session_indicator = string_to_session_indicator(session)
data_type_ = string_to_data_type(data_type)
likelihood_function_ = string_to_likelihood_function(likelihood_function)

print("running diffusion gridsearch for {session_indicator}")

submit_diffusion_gridsearch(
    session_indicator, time_window_ms, data_type_, likelihood_function_
)

print("done")
