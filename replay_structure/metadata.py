"""
Module for defining project metadata.
- Dynamics model types (e.g. stationary, momentum)
- Data types (e.g. ripples, run_snippets)
- Session types (e.g. real recording sessions, simulated sessions)
- Likelihood types (e.g. poisson, neg_binomial)
"""


from typing import Union, NamedTuple, Optional, List, Tuple

# --------------------------------------------------------------------------------------
# DEFINE PATHS FOR LOADING/SAVING DATA


DATA_PATH = "/Users/emmakrause/Documents/PhD/Lab/replay_structure/data"
RESULTS_PATH = "/Users/emmakrause/Documents/PhD/Lab/replay_structure/results"
FIGURES_PATH = "/Users/emmakrause/Documents/PhD/Lab/replay_structure/figures"
PLOTTING_FOLDER = (
    "/Users/emmakrause/Documents/PhD/Lab/replay_structure/results/plotting_temp"
)
DATA_PATH_O2 = "/home/ek195/replay_structure/data"
RESULTS_PATH_O2 = "/home/ek195/replay_structure/results"

# --------------------------------------------------------------------------------------
# DEFINE MODEL TYPES


class Diffusion(NamedTuple):
    def __str__(self):
        return "diffusion"


class Momentum(NamedTuple):
    def __str__(self):
        return "momentum"


class Stationary(NamedTuple):
    def __str__(self):
        return "stationary"


class Stationary_Gaussian(NamedTuple):
    def __str__(self):
        return "stationary_gaussian"


class Random(NamedTuple):
    def __str__(self):
        return "random"


Model_Name = Union[Diffusion, Momentum, Stationary, Stationary_Gaussian, Random]


class Model(NamedTuple):
    name: Model_Name
    trajectory: bool
    n_params: Optional[int]


Diffusion_Model = Model(name=Diffusion(), trajectory=True, n_params=1)
Momentum_Model = Model(name=Momentum(), trajectory=True, n_params=2)
Stationary_Model = Model(name=Stationary(), trajectory=False, n_params=None)
Stationary_Gaussian_Model = Model(
    name=Stationary_Gaussian(), trajectory=False, n_params=1
)
Random_Model = Model(name=Random(), trajectory=False, n_params=None)

MODELS = [
    Diffusion_Model,
    Momentum_Model,
    Stationary_Model,
    Stationary_Gaussian_Model,
    Random_Model,
]
N_MODELS = len(MODELS)


def model_to_string(model: Model) -> str:
    return str(model.name)


MODELS_AS_STR = list(map(model_to_string, MODELS))

MODEL_NAMES_FOR_PLOTTING_DICT = {
    "diffusion": "Diffusion",
    "momentum": "Momentum",
    "stationary": "Stationary",
    "stationary_gaussian": "Gaussian",
    "random": "Random",
}
MODEL_NAMES_FOR_PLOTTING_LIST = [
    MODEL_NAMES_FOR_PLOTTING_DICT[model] for model in MODELS_AS_STR
]

# --------------------------------------------------------------------------------------
# DEFINE SESSION TYPES


SESSION_RATDAY = {
    0: dict(rat=1, day=1, n_SWRs=322),
    1: dict(rat=1, day=2, n_SWRs=527),
    2: dict(rat=2, day=1, n_SWRs=222),
    3: dict(rat=2, day=2, n_SWRs=257),
    4: dict(rat=3, day=1, n_SWRs=406),
    5: dict(rat=3, day=2, n_SWRs=296),
    6: dict(rat=4, day=1, n_SWRs=594),
    7: dict(rat=4, day=2, n_SWRs=356),
}
SESSIONS_AS_STR = ["0", "1", "2", "3", "4", "5", "6", "7"]
N_SESSIONS = len(SESSION_RATDAY)


class Session_Name(NamedTuple):
    rat: int
    day: int

    def __str__(self):
        return f"rat{self.rat}day{self.day}"


class Simulated_Session_Name(NamedTuple):
    model: Model

    def __str__(self):
        return str(self.model.name)


class SessionSpikemat_Name(NamedTuple):
    rat: int
    day: int
    spikemat: int
    session_name: Session_Name

    def __str__(self):
        return f"rat{self.rat}day{self.day}spikemat{self.spikemat}"


Session_Indicator = Union[Session_Name, Simulated_Session_Name, SessionSpikemat_Name]


Session_List: List[Session_Indicator] = [
    Session_Name(rat=SESSION_RATDAY[session]["rat"], day=SESSION_RATDAY[session]["day"])
    for session in range(N_SESSIONS)
]
Simulated_Session_List: List[Session_Indicator] = [
    Simulated_Session_Name(model) for model in MODELS
]
SessionSpikemat_List: List[Session_Indicator] = [
    SessionSpikemat_Name(
        rat=SESSION_RATDAY[session]["rat"],
        day=SESSION_RATDAY[session]["day"],
        spikemat=spikemat,
        session_name=Session_Name(
            rat=SESSION_RATDAY[session]["rat"], day=SESSION_RATDAY[session]["day"]
        ),
    )
    for session in range(8)
    for spikemat in range(SESSION_RATDAY[session]["n_SWRs"])
]

# --------------------------------------------------------------------------------------
# DEFINE LIKELIHOOD TYPES


class Neg_Binomial(NamedTuple):
    def __str__(self):
        return "negbinomial"


class Poisson(NamedTuple):
    def __str__(self):
        return "poisson"


Likelihood_Function = Union[Neg_Binomial, Poisson]


class Neg_Binomial_Params(NamedTuple):
    alpha: float
    beta: float
    name: Likelihood_Function = Neg_Binomial()


class Poisson_Params(NamedTuple):
    rate_scaling: Optional[float]
    name: Likelihood_Function = Poisson()


Likelihood_Function_Params = Union[Neg_Binomial_Params, Poisson_Params]

# --------------------------------------------------------------------------------------
# DEFINE DATA TYPES


class Ripples(NamedTuple):
    def __str__(self):
        return "ripples"


class Run_Snippets(NamedTuple):
    def __str__(self):
        return "run_snippets"


class Poisson_Simulated_Ripples(NamedTuple):
    def __str__(self):
        return "poisson_simulated_ripples"


class NegBinomial_Simulated_Ripples(NamedTuple):
    def __str__(self):
        return "negbinomial_simulated_ripples"


class Ripples_PF(NamedTuple):
    def __str__(self):
        return "ripples_pf"


class PlaceFieldID_Shuffle(NamedTuple):
    def __str__(self):
        return "placefieldID_shuffle"


class PlaceField_Rotation(NamedTuple):
    def __str__(self):
        return "placefield_rotation"


class HighSynchronyEvents(NamedTuple):
    def __str__(self):
        return "high_synchrony_events"


class HighSynchronyEvents_PF(NamedTuple):
    def __str__(self):
        return "high_synchrony_events_pf"


Data_Type_Name = Union[
    Ripples,
    Run_Snippets,
    Poisson_Simulated_Ripples,
    NegBinomial_Simulated_Ripples,
    Ripples_PF,
    HighSynchronyEvents_PF,
    PlaceFieldID_Shuffle,
    PlaceField_Rotation,
    HighSynchronyEvents,
]


class Data_Type(NamedTuple):
    name: Data_Type_Name
    simulated_data_name: Optional[Ripples]
    session_list: List[Session_Indicator]
    default_time_window_ms: int
    default_likelihood_function: Likelihood_Function


Ripple_Data = Data_Type(
    name=Ripples(),
    simulated_data_name=None,
    default_time_window_ms=3,
    session_list=Session_List,
    default_likelihood_function=Poisson(),
)
Run_Snippet_Data = Data_Type(
    name=Run_Snippets(),
    simulated_data_name=None,
    default_time_window_ms=60,
    session_list=Session_List,
    default_likelihood_function=Poisson(),
)
Poisson_Simulated_Ripple_Data = Data_Type(
    name=Poisson_Simulated_Ripples(),
    simulated_data_name=Ripples(),
    default_time_window_ms=3,
    session_list=Simulated_Session_List,
    default_likelihood_function=Poisson(),
)
NegBinomial_Simulated_Ripple_Data = Data_Type(
    name=NegBinomial_Simulated_Ripples(),
    simulated_data_name=Ripples(),
    default_time_window_ms=3,
    session_list=Simulated_Session_List,
    default_likelihood_function=Neg_Binomial(),
)
Ripples_PF_Data = Data_Type(
    name=Ripples_PF(),
    simulated_data_name=None,
    default_time_window_ms=20,
    session_list=Session_List,
    default_likelihood_function=Poisson(),
)
PlaceFieldID_Shuffle_Data = Data_Type(
    name=PlaceFieldID_Shuffle(),
    simulated_data_name=None,
    default_time_window_ms=3,
    session_list=Session_List,
    default_likelihood_function=Poisson(),
)
PlaceField_Rotation_Data = Data_Type(
    name=PlaceField_Rotation(),
    simulated_data_name=None,
    default_time_window_ms=3,
    session_list=Session_List,
    default_likelihood_function=Poisson(),
)
HighSynchronyEvents_Data = Data_Type(
    name=HighSynchronyEvents(),
    simulated_data_name=None,
    default_time_window_ms=3,
    session_list=Session_List,
    default_likelihood_function=Poisson(),
)
HighSynchronyEvents_PF_Data = Data_Type(
    name=HighSynchronyEvents_PF(),
    simulated_data_name=None,
    default_time_window_ms=20,
    session_list=Session_List,
    default_likelihood_function=Poisson(),
)


# --------------------------------------------------------------------------------------
# FUNCTIONS FOR CONVERTING CLI INPUT


def string_to_session_indicator(
    session_indicator: Union[str, int, Tuple[int, int]]
) -> Session_Indicator:
    if isinstance(session_indicator, int):
        return Session_Name(
            rat=SESSION_RATDAY[session_indicator]["rat"],
            day=SESSION_RATDAY[session_indicator]["day"],
        )
    elif isinstance(session_indicator, str):
        if session_indicator in SESSIONS_AS_STR:
            return Session_Name(
                rat=SESSION_RATDAY[int(session_indicator)]["rat"],
                day=SESSION_RATDAY[int(session_indicator)]["day"],
            )
        elif session_indicator in MODELS_AS_STR:
            return Simulated_Session_Name(string_to_model(session_indicator))
        else:
            raise Exception("Input cannot be mapped to session or simulated session")
    elif isinstance(session_indicator, tuple):
        return SessionSpikemat_Name(
            rat=SESSION_RATDAY[session_indicator[0]]["rat"],
            day=SESSION_RATDAY[session_indicator[0]]["day"],
            spikemat=session_indicator[1],
            session_name=Session_Name(
                rat=SESSION_RATDAY[session_indicator[0]]["rat"],
                day=SESSION_RATDAY[session_indicator[0]]["day"],
            ),
        )
    else:
        raise Exception("Input error")


def string_to_data_type(data_type: str) -> Data_Type:
    if data_type == "ripples":
        return Ripple_Data
    elif data_type == "run_snippets":
        return Run_Snippet_Data
    elif data_type == "poisson_simulated_ripples":
        return Poisson_Simulated_Ripple_Data
    elif data_type == "negbinomial_simulated_ripples":
        return NegBinomial_Simulated_Ripple_Data
    elif data_type == "ripples_pf":
        return Ripples_PF_Data
    elif data_type == "placefieldID_shuffle":
        return PlaceFieldID_Shuffle_Data
    elif data_type == "placefield_rotation":
        return PlaceField_Rotation_Data
    elif data_type == "high_synchrony_events":
        return HighSynchronyEvents_Data
    elif data_type == "high_synchrony_events_pf":
        return HighSynchronyEvents_PF_Data
    else:
        raise Exception("Invalid data_type")


def string_to_model(model_name: str) -> Model:
    if model_name == "diffusion":
        return Diffusion_Model
    elif model_name == "momentum":
        return Momentum_Model
    elif model_name == "stationary":
        return Stationary_Model
    elif model_name == "stationary_gaussian":
        return Stationary_Gaussian_Model
    elif model_name == "random":
        return Random_Model
    else:
        raise Exception("Invalid model_name")


def string_to_likelihood_function(
    likelihood_function_string: str
) -> Likelihood_Function:
    if likelihood_function_string == "poisson":
        return Poisson()
    elif likelihood_function_string == "negbinomial":
        return Neg_Binomial()
    else:
        raise Exception("Invalid likelihood_function_string")
