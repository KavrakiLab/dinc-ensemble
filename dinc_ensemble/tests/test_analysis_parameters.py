from dinc_ensemble import DINC_ANALYSIS_PARAMS
from dinc_ensemble.parameters.analysis \
    import DEFAULT_DINC_N_OUT, \
DINC_PLOT_SCORE_RMSD, DEFAULT_DINC_PLOT_SCORE_RMSD, \
    DINC_RMSD, DEFAULT_DINC_RMSD

    
import pytest

def test_frag_params():
    #check if all parameters are equal to default at the beginning
    assert DINC_ANALYSIS_PARAMS.n_out == DEFAULT_DINC_N_OUT
    assert DINC_ANALYSIS_PARAMS.dinc_rmsd == DEFAULT_DINC_RMSD
    assert DINC_ANALYSIS_PARAMS.plot_score_vs_rmsd == DEFAULT_DINC_PLOT_SCORE_RMSD
    
    # make sure that the things are changed
    DINC_ANALYSIS_PARAMS.n_out = 12
    DINC_ANALYSIS_PARAMS.dinc_rmsd = DINC_RMSD.SYMMETRIC
    DINC_ANALYSIS_PARAMS.plot_score_vs_rmsd = DINC_PLOT_SCORE_RMSD.NO
    
    assert DINC_ANALYSIS_PARAMS.n_out == 12
    assert DINC_ANALYSIS_PARAMS.dinc_rmsd == DINC_RMSD.SYMMETRIC
    assert DINC_ANALYSIS_PARAMS.plot_score_vs_rmsd == DINC_PLOT_SCORE_RMSD.NO
    

        