from enum import Enum, StrEnum
from dataclasses import dataclass
from .utils_decorate_docstring import docstring_parameter, ParameterDataclasses

#-------------------DINC VINA ENGINE PARAMETERS--------------------------#

# Type of scoring function to optimize: autodock4, vina, vinardo
class SCORE_F(str, Enum):
    AD4 = "AD4"
    VINA = "VINA"
    VINARDO = "VINARDO"
DEFAULT_SCORE_F = SCORE_F.VINA

# Vina exhaustiveness parameter
DEFAULT_VINA_EXHAUSTIVE: int = 8

# Number of outputs to produce in each round
DEFAULT_VINA_N_POSES: int = 100

# Number of cpus to use by vina (0 - all)
DEFAULT_VINA_CPU_CNT: int = 4

# Vina seed
DEFAULT_VINA_SEED: int = 0

# Minimum RMSD difference between poses 
DEFAULT_VINA_MIN_RMSD: float = 1

# Maximum number of evaluation (0 - heuristics)
DEFAULT_VINA_MAX_EVALS: int = 0

# Number of randomized steps
DEFAULT_RAND_STEPS: int = 1000


@dataclass
@docstring_parameter(DEFAULT_SCORE_F,\
                    DEFAULT_VINA_EXHAUSTIVE,\
                    DEFAULT_VINA_N_POSES,\
                    DEFAULT_VINA_CPU_CNT,\
                    DEFAULT_VINA_SEED,\
                    DEFAULT_VINA_MIN_RMSD,\
                    DEFAULT_VINA_MAX_EVALS,\
                    DEFAULT_RAND_STEPS)
class VinaEngineParams(ParameterDataclasses):
    '''
    Parameters for DINC-Ensemble analysis.

    Attributes
    ----------
    score_f: SCORE_F
        Type of scoring function for Vina to optimize: autodock4, vina, vinardo
        Defaults to {0}
    exhaustive: int
        Vina exhaustiveness parameter
        Defaults to {1}
    n_poses: int
        Number of outputs for Vina to produce in each round
        Defaults to {2}
    cpu_count: int
        Number of cpus for Vina to use (0 - all)
        Defaults to {3}
    seed: int
        Vina seed (0 - random)
        Defaults to {4}
    min_rmsd: float
        Minimum rmsd difference between poses for Vina
        Defaults to {5}
    max_evals: int
        Maximum number of Vina evaluations (0 - heuristics)
        Defaults to {6}
    rand_steps: int
        Number of randomized steps for Vina randomization
        Defaults to {7}
   '''
   
    score_f: SCORE_F = DEFAULT_SCORE_F
    exhaustive: int = DEFAULT_VINA_EXHAUSTIVE
    n_poses: int = DEFAULT_VINA_N_POSES
    cpu_count: int = DEFAULT_VINA_CPU_CNT
    seed: int = DEFAULT_VINA_SEED
    min_rmsd: float = DEFAULT_VINA_MIN_RMSD
    max_evals: int = DEFAULT_VINA_MAX_EVALS
    rand_steps: int = DEFAULT_RAND_STEPS