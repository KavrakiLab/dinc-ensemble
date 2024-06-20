from dinc_ensemble import DINC_CORE_PARAMS
from dinc_ensemble.parameters.core import *
import pytest

def test_sample_score_rescore():
    #check if all parameters are equal to default at the beginning
    assert DINC_CORE_PARAMS.job_type == DEFAULT_DINC_JOB_TYPE
    assert DINC_CORE_PARAMS.dock_engine == DEFAULT_DOCK_ENGINE
    assert DINC_CORE_PARAMS.replica_num == DEFAULT_DINC_NUM_REPLICAS
    assert DINC_CORE_PARAMS.output_dir == DEFAULT_OUTPUT_DIR
    assert DINC_CORE_PARAMS.dock_type == DEFAULT_DINC_DOCK_TYPE
    # make sure that the things are changed
    DINC_CORE_PARAMS.job_type = DINC_JOB_TYPE.CROSSDOCK
    DINC_CORE_PARAMS.dock_engine = DINC_DOCK_ENGINE.VINA
    DINC_CORE_PARAMS.replica_num = 2
    DINC_CORE_PARAMS.output_dir = "random"
    DINC_CORE_PARAMS.dock_type = DINC_DOCK_TYPE.EXHAUSTIVE
    
    assert DINC_CORE_PARAMS.job_type == DINC_JOB_TYPE.CROSSDOCK
    assert DINC_CORE_PARAMS.dock_engine == DINC_DOCK_ENGINE.VINA
    assert DINC_CORE_PARAMS.replica_num == 2
    assert DINC_CORE_PARAMS.output_dir == "random"
    assert DINC_CORE_PARAMS.dock_type == DINC_DOCK_TYPE.EXHAUSTIVE

        