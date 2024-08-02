from enum import Enum, StrEnum
import psutil
from dataclasses import dataclass
from .utils_decorate_docstring import docstring_parameter, ParameterDataclasses

# Box center type: ligc, protc, user
# Set the center of the box to be the ligand's center, the protein's center, or other (see below).
class BBOX_CENTER_TYPE(str, Enum):
    LIGC = "LIGC"
    PROTC = "PROTC"
    USER = "USER"
DEFAULT_BBOX_CENTER_TYPE = BBOX_CENTER_TYPE.LIGC

# Coordinates of the box center (if user defined)
DEFAULT_BBOX_CENTER_X:float = 0
DEFAULT_BBOX_CENTER_Y:float = 0
DEFAULT_BBOX_CENTER_Z:float = 0

# Box center type: lig, user
# Set the dimension of the box to be size of the ligand with padding, the protein's center, or other (see below).
class BBOX_DIM_TYPE(str, Enum):
    LIG = "LIG"
    USER = "USER"
DEFAULT_BBOX_DIM_TYPE = BBOX_DIM_TYPE.LIG

# Box padding (if box dimensions are ligc user defined, A)
DEFAULT_BBOX_PADDING:float = 0

# Box dimentsions (if user defined, A)
DEFAULT_BBOX_DIM_X:float = 0
DEFAULT_BBOX_DIM_Y:float = 0
DEFAULT_BBOX_DIM_Z:float = 0


# If receptors should be aligned before docking
DEFAULT_ALIGN_RECEPTORS:bool = True
# Which receptor should be used as reference for alignment 
# (based on input order)
DEFAULT_REF_RECEPTOR_IND:int = 0


@dataclass
@docstring_parameter(DEFAULT_BBOX_CENTER_TYPE, DEFAULT_BBOX_CENTER_X, 
                     DEFAULT_BBOX_CENTER_Y, DEFAULT_BBOX_CENTER_Z, 
                     DEFAULT_BBOX_DIM_TYPE, DEFAULT_BBOX_PADDING,
                     DEFAULT_BBOX_DIM_X, DEFAULT_BBOX_DIM_Y,
                     DEFAULT_BBOX_DIM_Z, DEFAULT_ALIGN_RECEPTORS,
                     DEFAULT_REF_RECEPTOR_IND)
class DincReceptorParams(ParameterDataclasses):
    """
    Core parameters for DINC-Ensemble.

    Attributes
    ----------
    bbox_center_type : BBOX_CENTER_TYPE
        The type of bounding box center to be used. 
        Defaults to {0}
    bbox_center_x : float
        The x-coordinate of the bounding box center. 
        Defaults to {1}
    bbox_center_y : float
        The y-coordinate of the bounding box center. 
        Defaults to {2}
    bbox_center_z : float
        The z-coordinate of the bounding box center. 
        Defaults to {3}
    bbox_dim_type : DEFAULT_BBOX_DIM_TYPE
        The type of bounding box dimension to be used. 
        Defaults to {4}
    bbox_dim_x : float
        The x dimension of the bounding box (A). 
        Defaults to {5}
    bbox_dim_y : float
        The y dimension of the bounding box (A). 
        Defaults to {6}
    bbox_dim_z : float
        The z dimension of the bounding box (A). 
        Defaults to {7}
    align_receptors : bool
        If receptors should be aligned to a reference before docking. 
        Defaults to {8}
    ref_receptor : int
        Receptor to use as reference for alignment and bbox estimation.
        Based on input order
        Defaults to {9}
    """
  
    # core parameters
    bbox_center_type: BBOX_CENTER_TYPE = DEFAULT_BBOX_CENTER_TYPE
    bbox_center_x: float = DEFAULT_BBOX_CENTER_X
    bbox_center_y: float = DEFAULT_BBOX_CENTER_Y
    bbox_center_z: float = DEFAULT_BBOX_CENTER_Z
    bbox_dim_type: BBOX_DIM_TYPE = DEFAULT_BBOX_DIM_TYPE
    bbox_padding: float = DEFAULT_BBOX_PADDING
    bbox_dim_x: float = DEFAULT_BBOX_DIM_X
    bbox_dim_y: float = DEFAULT_BBOX_DIM_Y
    bbox_dim_z: float = DEFAULT_BBOX_DIM_Z
    align_receptors: bool = DEFAULT_ALIGN_RECEPTORS
    ref_receptor: int = DEFAULT_REF_RECEPTOR_IND
