"""
Unit tests for the ligand fragment module.
"""
import pytest

from dinc_ensemble.ligand.core import DINCMolecule
from dinc_ensemble.ligand import DINCFragment
from dinc_ensemble.parameters.fragment import *
from .frag_test_utils import *

def test_write_pdbqt_fragments(
                        ligands_10dof,
                        ligands_20dof):
    # Test if initializing fragments works
    init_fragments_multi_ligand([
                        ligands_10dof,
                        ligands_20dof
                        ], write_pdbqt=True)
    
def test_write_svg_fragments(
                    ligands_10dof):
    # Test if initializing fragments works
    init_fragments_multi_ligand([
                        ligands_10dof
                        ], write_svg=True)
    
    
def test_df_fragments(
                    ligands_10dof):
    # Test if initializing fragments works
    init_fragments_multi_ligand([
                        ligands_10dof
                        ], get_df=True)

    
