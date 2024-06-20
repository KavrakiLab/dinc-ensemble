"""
Unit tests for the ligand fragment module.
"""
import pytest

from dinc_ensemble.ligand.core import DINCMolecule
from dinc_ensemble.ligand import DINCFragment
from dinc_ensemble.parameters.fragment import *
from .frag_test_utils import *

def test_init_fragments(ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof):
    # Test if initializing fragments works
    init_fragments_multi_ligand([
                        ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof
                        ])

def test_init_fragments_different_rand_root(ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof):
    
    init_fragments_multi_ligand([
                        ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof
                        ],
                        root_type=DINC_ROOT_TYPE.RANDOM)

def test_init_fragments_different_user_root(ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof):
    
    init_fragments_multi_ligand([
                        ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof
                        ],
                        root_type=DINC_ROOT_TYPE.USER)
    
def test_init_fragments_different_first_root(
                        ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof):
    
    init_fragments_multi_ligand([
                        ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof
                        ],
                        root_type=DINC_ROOT_TYPE.AUTO,
                        root_auto=DINC_ROOT_AUTO.FIRST)
    
def test_init_fragments_different_last_root(ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof):
    
    init_fragments_multi_ligand([
                        ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof
                        ],
                        root_type=DINC_ROOT_TYPE.AUTO,
                        root_auto=DINC_ROOT_AUTO.LAST)

def test_init_fragments_different_hbond_root(ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof):
    
    init_fragments_multi_ligand([
                        ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof
                        ],
                        root_type=DINC_ROOT_TYPE.AUTO,
                        root_auto=DINC_ROOT_AUTO.H_BONDS)
    
def test_init_fragments_different_largest_root(ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof):
    
    init_fragments_multi_ligand([
                        ligands_0dof,
                        ligands_1dof,
                        ligands_5dof,
                        ligands_10dof,
                        ligands_20dof
                        ],
                        root_type=DINC_ROOT_TYPE.AUTO,
                        root_auto=DINC_ROOT_AUTO.LARGEST)

    
