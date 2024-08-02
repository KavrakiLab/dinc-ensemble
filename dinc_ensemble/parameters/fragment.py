from enum import Enum
from dataclasses import dataclass
from .utils_decorate_docstring import docstring_parameter, ParameterDataclasses
from typing import Optional

# Protocol defining the parameters driving the decomposition of the ligand into fragments:
# - automatic: values for frag_size and frag_new will be based on the ligand's size
# - manual: manually set values for frag_size and frag_new (see below)
class DINC_FRAGMENT_MODE(str, Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"
DEFAULT_DINC_FRAGMENT_MODE = DINC_FRAGMENT_MODE.MANUAL

# Total number of active degrees-of-freedom (i.e., torsion angles) in a fragment.
DEFAULT_DINC_FRAG_SIZE: int = 4

# Maximum number of "new" degrees-of-freedom in a fragment (other than the initial fragment).
# This number must be less than frag_size.
DEFAULT_DINC_FRAG_NEW: int = 3

# Protocol for selecting the root atom of the ligand's torsion tree:
# - random: select a heavy atom of the ligand at random
# - automatic: automatically select a heavy atom from the ligand (see below)
# - user: the root atom is user-specified (see below).
class DINC_ROOT_TYPE(str, Enum):
    AUTO = "AUTO"
    RANDOM = "RANDOM"
    USER = "USER"
DEFAULT_DINC_ROOT_TYPE = DINC_ROOT_TYPE.AUTO

# If root_type is "automatic", root_auto defines in which way the root atom is selected:
# - first: select the first heavy atom of the ligand
# - last: select the last heavy atom of the ligand
# - largest: select the atom maximizing the number of heavy atoms in the initial fragment
# - H_bonds: select the atom maximizing the potential for hydrogen bonds in the initial fragment
class DINC_ROOT_AUTO(str, Enum):
    FIRST = "FIRST"
    LAST = "LAST"
    LARGEST = "LARGEST"
    H_BONDS = "H_BONDS"
DEFAULT_DINC_ROOT_AUTO = DINC_ROOT_AUTO.H_BONDS


@dataclass
@docstring_parameter(DEFAULT_DINC_FRAGMENT_MODE,
                 DEFAULT_DINC_FRAG_SIZE,
                 DEFAULT_DINC_FRAG_NEW,
                 DEFAULT_DINC_ROOT_TYPE,\
                 DEFAULT_DINC_ROOT_AUTO)
class DincFragParams(ParameterDataclasses):
    """
    Fragmentation parameters for DINC.

    Attributes
    ----------
    frag_mode : DINC_FRAGMENT_MODE
        The fragmentation mode. 
        Defaults to {0}
    frag_size : int
        The total number of active degrees-of-freedom (i.e., torsion angles) in a fragment. 
        Defaults to {1}
    frag_new : int
        The maximum number of "new" degrees-of-freedom in a fragment (other than the initial fragment). 
        Defaults to {2}
    root_type : DINC_ROOT_TYPE
        The protocol for selecting the root atom of the ligand's torsion tree. 
        Defaults to {3}
    root_auto : DINC_ROOT_AUTO
        The protocol for selecting the root atom if root_type is "automatic". 
        Defaults to {4}
    root_name : str
        Name of the root atom (will be used if root_type is USER)
        Defaults to None
    """
    
    frag_mode: DINC_FRAGMENT_MODE = DEFAULT_DINC_FRAGMENT_MODE
    frag_size: int = DEFAULT_DINC_FRAG_SIZE
    frag_new: int = DEFAULT_DINC_FRAG_NEW
    root_type: DINC_ROOT_TYPE = DEFAULT_DINC_ROOT_TYPE
    root_auto: DINC_ROOT_AUTO = DEFAULT_DINC_ROOT_AUTO
    root_name: Optional[str] = None
    
