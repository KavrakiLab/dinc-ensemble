from enum import Enum, StrEnum

import multiprocessing
from dataclasses import dataclass
from .utils_decorate_docstring import docstring_parameter, ParameterDataclasses

#-------------------DINC CORE PARAMETERS--------------------------#
# Type of docking job: crossdocking, redocking, scoring
class DINC_JOB_TYPE(str, Enum):
    CROSSDOCK = "CROSSDOCK"
    REDOCK = "REDOCK"
    SCORING = "SCORING"
DEFAULT_DINC_JOB_TYPE = DINC_JOB_TYPE.REDOCK

# Type of docking style: incremental, nonincremental, randomize
class DINC_DOCK_TYPE(str, Enum):
    INCREMENTAL = "INCREMENTAL"
    CLASSIC = "CLASSIC"
    EXHAUSTIVE = "EXHAUSTIVE"
DEFAULT_DINC_DOCK_TYPE = DINC_DOCK_TYPE.CLASSIC


# Tool that DINC uses for sampling during the incremental docking: AD4, Vina
class DINC_DOCK_ENGINE(str, Enum):
    VINA = "VINA"
    # TODO: - add others?
DEFAULT_DOCK_ENGINE = DINC_DOCK_ENGINE.VINA

# Number of threads allocated for the parallelized docking protocol.
# 4  for all
DEFAULT_DINC_NUM_REPLICAS: int = 24

# Directory for all outputs (default: the place where t is runnning)
DEFAULT_OUTPUT_DIR:str = "."

# Number of output ligands (default: 5)
DEFAULT_N_OUT:int = 5

@dataclass
@docstring_parameter(DEFAULT_DINC_JOB_TYPE,
                     DEFAULT_DINC_DOCK_TYPE,
                     DEFAULT_DOCK_ENGINE, 
                     DEFAULT_DINC_NUM_REPLICAS,
                     DEFAULT_OUTPUT_DIR,
                     DEFAULT_N_OUT)
class DincCoreParams(ParameterDataclasses):
    """
    Core parameters for DINC-Ensemble.

    Attributes
    ----------
    job_type : DINC_JOB_TYPE
        The type of job to be performed. 
        Defaults to {0}
    dock_type : DINC_DOCK_TYPE
        The type of docking to be performed. 
        Defaults to {1}
    dock_engine : DINC_DOCK_ENGINE
        The docking method to be used. 
        Defaults to {2}
    replica_num : int
        Number of replicas to run of each job
        Defaults to {3}
    output_dir : str
        Directory for writing outputs.
        Defaults to {4}
    n_out : int
        Number of output ligand files.
        Defaults to {5}
    """
  
    # core parameters
    job_type: DINC_JOB_TYPE = DEFAULT_DINC_JOB_TYPE
    dock_engine : DINC_DOCK_ENGINE = DEFAULT_DOCK_ENGINE
    replica_num : int = DEFAULT_DINC_NUM_REPLICAS
    dock_type: DINC_DOCK_TYPE = DEFAULT_DINC_DOCK_TYPE
    output_dir: str = DEFAULT_OUTPUT_DIR
    n_out: str = DEFAULT_N_OUT