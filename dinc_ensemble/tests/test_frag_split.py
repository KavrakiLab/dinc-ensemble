import pytest

from dinc_ensemble.ligand import DINCFragment
from dinc_ensemble.ligand.core import DINCMolecule
from dinc_ensemble.parameters import *
from copy import deepcopy

def test_simple_ligand0(ligands_0dof):
    for lig in ligands_0dof:
        lig_copy = deepcopy(lig)
        frag = DINCFragment(lig_copy)
        assert len(frag.fragments) == 1

def test_simple_ligand1(ligands_1dof):
    for lig in ligands_1dof:
        lig_copy = deepcopy(lig)
        frag = DINCFragment(lig_copy)
        assert len(frag.fragments) == 1
        assert len(frag.fragments_dincmol) == 1
        params = DincFragParams(frag_mode= DINC_FRAGMENT_MODE.MANUAL,
                                frag_size=0,
                            frag_new=1)
        frag = DINCFragment(lig_copy, params)
        assert len(frag.fragments) == 2
        assert len(frag.fragments_dincmol) == 2
        params = DincFragParams(frag_mode= DINC_FRAGMENT_MODE.MANUAL,
                                frag_size=1)
        frag = DINCFragment(lig_copy, params)
        assert len(frag.fragments) == 1
        assert len(frag.fragments_dincmol) == 1

def test_simple_ligand5(ligands_5dof):
    for lig in ligands_5dof:
        lig_copy = deepcopy(lig)
        params = DincFragParams(frag_mode= DINC_FRAGMENT_MODE.MANUAL,
                                frag_size=0,
                            frag_new=1,
                            root_type=DINC_ROOT_TYPE.AUTO,
                            root_auto=DINC_ROOT_AUTO.FIRST)
        frag = DINCFragment(lig_copy, params)
        assert len(frag.fragments) > 0
        assert len(frag.fragments_dincmol) > 0

def test_simple_ligand10(ligands_10dof:
                         list[DINCMolecule]):
    for lig in ligands_10dof:
        lig_copy = deepcopy(lig)
        print(lig._mol_name)
        params = DincFragParams(frag_mode= DINC_FRAGMENT_MODE.MANUAL,
                                frag_size=0,
                            frag_new=1)
        frag = DINCFragment(lig_copy, params)
        assert (len(frag.fragments) == 10 or len(frag.fragments) == 11)
        assert (len(frag.fragments_dincmol) == 10 or len(frag.fragments_dincmol) == 11)

def test_simple_ligand20(ligands_20dof):
    for lig in ligands_20dof:
        lig_copy = deepcopy(lig)
        params = DincFragParams(frag_mode= DINC_FRAGMENT_MODE.MANUAL,
                                frag_size=0,
                            frag_new=1,
                            root_type=DINC_ROOT_TYPE.AUTO,
                            root_auto=DINC_ROOT_AUTO.FIRST)
        frag = DINCFragment(lig_copy, params)
        assert len(frag.fragments) >= 18
        assert len(frag.fragments_dincmol) >= 18
        params = DincFragParams(root_type=DINC_ROOT_TYPE.AUTO,
                            root_auto=DINC_ROOT_AUTO.FIRST)
        frag = DINCFragment(lig_copy,
                            params)
        assert (len(frag.fragments) == 7 or len(frag.fragments) == 6)