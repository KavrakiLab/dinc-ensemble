import pytest

from dinc_ensemble.ligand import DINCFragment
from dinc_ensemble.ligand.core import DINCMolecule
from dinc_ensemble.parameters import *
from copy import deepcopy

def test_simple_ligand0_update_conf_translation(ligands_0dof):
    for lig in ligands_0dof:
        lig_copy = deepcopy(lig)
        frag = DINCFragment(lig_copy)
        frag0 = frag.fragments[0]
        init_x = frag0.allAtoms[0].coords[0]
        init_pdbqt = frag0.pdbqt_str
        frag.init_conformation(0, translation=[10, 0, 0])
        atom_f0_coord = frag0.allAtoms[0].coords
        assert atom_f0_coord[0] == (init_x+10)
        assert init_pdbqt != frag0.pdbqt_str
        frag.init_conformation(0, translation=[-10, 0, 0])
        assert round(atom_f0_coord[0], 3) == round(init_x, 3)
        assert init_pdbqt == frag0.pdbqt_str
        
def test_simple_ligand1_update_conf_torsion_update(ligands_1dof):
    # TODO: there is some issue with split_frags and their torTrees 
    # (might have to investigate this a bit later)
    for lig in ligands_1dof:
        lig_copy = deepcopy(lig)
        frag = DINCFragment(lig_copy)
        frag0 = frag.fragments[0]
        init_x = frag0.allAtoms[-1].coords[0]
        init_pdbqt = frag0.pdbqt_str
        tors = []
        for tor in frag0.torTree.torsionMap:
            tors.append(30)
        frag.init_conformation(0,torsions=tors)
        new_x = frag0.allAtoms[-1].coords[0]
        assert new_x != init_x
        assert init_pdbqt != frag0.pdbqt_str
        tors = []
        for tor in frag0.torTree.torsionMap:
            tors.append(-10)
        frag.init_conformation(0,torsions=tors)
        assert abs(frag0.allAtoms[-1].coords[0] - init_x) < 1
        #assert init_pdbqt == frag0.pdbqt_str
        
def test_simple_ligand5_expand_conf_translaton_update(ligands_5dof):
    # TODO: there is some issue with split_frags and their torTrees 
    # (might have to investigate this a bit later)
    for lig in ligands_5dof:
        lig_copy = deepcopy(lig)
        frag_params = DINC_FRAG_PARAMS
        frag_params.frag_size = 2
        frag_params.frag_new = 1
        frag = DINCFragment(lig_copy, frag_params)
        check_atom_name = list(frag._molecule.atoms.index)[0]
        for i, f in enumerate(frag.fragments[1:]):
            prev_f = frag.fragments[i-1]
            check_atom = f.allAtoms.get(check_atom_name)
            check_atom_prev = prev_f.allAtoms.get(check_atom_name)
            assert round(check_atom.coords[0][0], 3) == round(check_atom_prev.coords[0][0], 3)
        # after changing the conf of initial fragment coords are no longer the same
        frag.init_conformation(0, translation=[10, 0, 0])
        check_atom_first = frag.fragments[0].allAtoms.get(check_atom_name)
        f_first = frag.fragments[0]
        for i, f in enumerate(frag.fragments[1:]):
            check_atom = f.allAtoms.get(check_atom_name)
            assert check_atom.coords[0] != check_atom_first.coords[0]
            assert f.pdbqt_str != f_first.pdbqt_str
        # but when we adjust / extend the conformation they equal out
        frag.expand_conformation(frag.fragments[0].conf, 0)
        
        for i, f in enumerate(frag.fragments[1:]):
            prev_f = frag.fragments[i-1]
            check_atom = f.allAtoms.get(check_atom_name)
            check_atom_prev = prev_f.allAtoms.get(check_atom_name)
            assert round(check_atom.coords[0][0], 3) == round(check_atom_prev.coords[0][0], 3)
        # assert round(check_atom.coords[1][1], 3) == round(check_atom_first.coords[1][1], 3)

        

