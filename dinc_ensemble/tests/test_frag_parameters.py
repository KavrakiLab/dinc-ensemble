from dinc_ensemble import DINC_FRAG_PARAMS
from dinc_ensemble.parameters.fragment \
    import DINC_FRAGMENT_MODE, \
DEFAULT_DINC_FRAGMENT_MODE, DEFAULT_DINC_FRAG_SIZE, \
    DEFAULT_DINC_FRAG_NEW, \
    DINC_ROOT_TYPE, DEFAULT_DINC_ROOT_TYPE, \
    DINC_ROOT_AUTO, DEFAULT_DINC_ROOT_AUTO

    
import pytest

def test_frag_params():
    #check if all parameters are equal to default at the beginning
    assert DINC_FRAG_PARAMS.frag_mode == DEFAULT_DINC_FRAGMENT_MODE
    #frag params will depend on ligand size if mode is automatic!
    #assert DINC_FRAG_PARAMS.frag_size == DEFAULT_DINC_FRAG_SIZE
    #assert DINC_FRAG_PARAMS.frag_new == DEFAULT_DINC_FRAG_NEW
    assert DINC_FRAG_PARAMS.root_type == DEFAULT_DINC_ROOT_TYPE
    assert DINC_FRAG_PARAMS.root_auto == DEFAULT_DINC_ROOT_AUTO
    
    # make sure that the things are changed
    DINC_FRAG_PARAMS.frag_mode = DINC_FRAGMENT_MODE.MANUAL
    DINC_FRAG_PARAMS.frag_size = 1
    DINC_FRAG_PARAMS.frag_new = 2
    DINC_FRAG_PARAMS.root_type = DINC_ROOT_TYPE.RANDOM
    DINC_FRAG_PARAMS.root_auto = DINC_ROOT_AUTO.H_BONDS
    
    assert DINC_FRAG_PARAMS.frag_mode == DINC_FRAGMENT_MODE.MANUAL
    assert DINC_FRAG_PARAMS.frag_size == 1
    assert DINC_FRAG_PARAMS.frag_new == 2
    assert DINC_FRAG_PARAMS.root_type == DINC_ROOT_TYPE.RANDOM
    assert DINC_FRAG_PARAMS.root_auto == DINC_ROOT_AUTO.H_BONDS

        