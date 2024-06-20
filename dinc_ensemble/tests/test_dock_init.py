"""
Unit tests for docking module.
"""
import pytest
from pathlib import Path

from dinc_ensemble import init_dinc_ensemble_threads
from dinc_ensemble.parameters import DincCoreParams

def test_dock_init_1_receptor(data_directory, test_directory):

    # Test if initializing fragments works
    pdb_code = "1a3e"
    dof = 39
    ligand_fname = Path(data_directory) / "pdbbind_test_ligands/{}_ligand_dof_{}.mol2".format(pdb_code, dof)
    receptor_fname = Path(data_directory) / "pdbbind_test_ligands/{}_receptor_dof_{}.pdb".format(pdb_code, dof)
    output_fname = Path(test_directory) / "{}_tmp_test".format(pdb_code)
    DincCoreParams.output_dir = str(output_fname)
    locations_df, dince_job_elems, dinc_run_info = init_dinc_ensemble_threads(str(ligand_fname), [str(receptor_fname)])

    

