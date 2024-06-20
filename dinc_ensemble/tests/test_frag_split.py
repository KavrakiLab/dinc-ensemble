import pytest

from dinc_ensemble.ligand import DINCFragment
from dinc_ensemble.ligand.core import DINCMolecule
from dinc_ensemble.parameters import *
from copy import deepcopy

def test_simple_ligand0(ligands_0dof):
    for lig in ligands_0dof:
        lig_copy = deepcopy(lig)
        frag = DINCFragment(lig_copy)
        frag._split_to_fragments_()
        assert len(frag.split_frags) == 1

def test_simple_ligand1(ligands_1dof):
    for lig in ligands_1dof:
        lig_copy = deepcopy(lig)
        frag = DINCFragment(lig_copy)
        frag._split_to_fragments_()
        assert len(frag.split_frags) == 1
        params = DincFragParams(frag_mode= DINC_FRAGMENT_MODE.MANUAL,
                                frag_size=0,
                            frag_new=1)
        frag = DINCFragment(lig_copy, params)
        frag._split_to_fragments_()
        assert len(frag.split_frags) == 2
        params = DincFragParams(frag_mode= DINC_FRAGMENT_MODE.MANUAL,
                                frag_size=1)
        frag = DINCFragment(lig_copy, params)
        frag._split_to_fragments_()
        assert len(frag.split_frags) == 1

def test_simple_ligand5(ligands_5dof):
    for lig in ligands_5dof:
        lig_copy = deepcopy(lig)
        params = DincFragParams(frag_mode= DINC_FRAGMENT_MODE.MANUAL,
                                frag_size=0,
                            frag_new=1,
                            root_type=DINC_ROOT_TYPE.AUTO,
                            root_auto=DINC_ROOT_AUTO.FIRST)
        frag = DINCFragment(lig_copy, params)
        frag._split_to_fragments_()
        assert len(frag.split_frags) == 6

def test_simple_ligand10(ligands_10dof:
                         list[DINCMolecule]):
    for lig in ligands_10dof:
        lig_copy = deepcopy(lig)
        print(lig._mol_name)
        params = DincFragParams(frag_mode= DINC_FRAGMENT_MODE.MANUAL,
                                frag_size=0,
                            frag_new=1)
        frag = DINCFragment(lig_copy, params)
        frag._split_to_fragments_()
        assert len(frag.split_frags) == 11

def test_simple_ligand20(ligands_20dof):
    for lig in ligands_20dof:
        lig_copy = deepcopy(lig)
        params = DincFragParams(frag_mode= DINC_FRAGMENT_MODE.MANUAL,
                                frag_size=0,
                            frag_new=1,
                            root_type=DINC_ROOT_TYPE.AUTO,
                            root_auto=DINC_ROOT_AUTO.FIRST)
        frag = DINCFragment(lig_copy, params)
        frag._split_to_fragments_()
        assert len(frag.split_frags) == 21
        params = DincFragParams(root_type=DINC_ROOT_TYPE.AUTO,
                            root_auto=DINC_ROOT_AUTO.FIRST)
        frag = DINCFragment(lig_copy,
                            params)
        frag._split_to_fragments_()
        assert len(frag.split_frags) == 2