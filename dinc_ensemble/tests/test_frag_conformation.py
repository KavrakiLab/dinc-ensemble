import pytest

from dinc_ensemble.ligand import DINCFragment
from dinc_ensemble.ligand.core import DINCMolecule
from dinc_ensemble.parameters import *
from copy import deepcopy

def test_simple_ligand0_update_conf_translation(ligands_0dof):
    for lig in ligands_0dof:
        lig_copy = deepcopy(lig)
        frag = DINCFragment(lig_copy)
        frag._split_to_fragments_()
        frag0 = frag.split_frags[0]
        init_x = frag0._molecule.molkit_molecule.allAtoms[0].coords[0]
        init_pdbqt = frag0._molecule.molkit_molecule.pdbqt_str
        frag0._init_conformation_(translation=[10, 0, 0])
        atom_f0_coord = frag0._molecule.molkit_molecule.allAtoms[0].coords
        assert atom_f0_coord[0] == (init_x+10)
        assert init_pdbqt != frag0._molecule.molkit_molecule.pdbqt_str
        frag0._init_conformation_(translation=[-10, 0, 0])
        assert round(atom_f0_coord[0], 3) == round(init_x, 3)
        #assert init_pdbqt == frag0._molecule.molkit_molecule.pdbqt_str
        
def test_simple_ligand1_update_conf_torsion_update(ligands_1dof):
    # TODO: there is some issue with split_frags and their torTrees 
    # (might have to investigate this a bit later)
    for lig in ligands_1dof:
        lig_copy = deepcopy(lig)
        frag = DINCFragment(lig_copy)
        frag._split_to_fragments_()
        frag0 = frag.split_frags[0]
        init_x = frag0._molecule.molkit_molecule.allAtoms[-1].coords[0]
        init_pdbqt = frag0._molecule.molkit_molecule.pdbqt_str
        frag._init_conformation_(torsions=[10])
        atom_f_coord = frag._molecule.molkit_molecule.allAtoms[-1].coords
        assert atom_f_coord[0] != init_x
        #assert init_pdbqt != frag0._molecule.molkit_molecule.pdbqt_str
        frag._init_conformation_(torsions=[-10])
        assert round(atom_f_coord[0], 3) == round(init_x, 3)
        #assert init_pdbqt == frag0._molecule.molkit_molecule.pdbqt_str
        
def test_simple_ligand5_expand_conf_translaton_update(ligands_5dof):
    # TODO: there is some issue with split_frags and their torTrees 
    # (might have to investigate this a bit later)
    for lig in ligands_5dof:
        lig_copy = deepcopy(lig)
        frag_params = DINC_FRAG_PARAMS
        frag_params.frag_size = 2
        frag_params.frag_new = 1
        frag = DINCFragment(lig_copy, frag_params)
        frag._split_to_fragments_()
        check_atom_name = list(frag.atoms.index)[0]
        for i, f in enumerate(frag.split_frags[1:]):
            prev_f = frag.split_frags[i-1]
            check_atom = f._molecule.molkit_molecule.allAtoms.get(check_atom_name)
            check_atom_prev = prev_f._molecule.molkit_molecule.allAtoms.get(check_atom_name)
            assert round(check_atom.coords[0][0], 3) == round(check_atom_prev.coords[0][0], 3)
        # after changing the conf of initial fragment coords are no longer the same
        frag.split_frags[0]._init_conformation_(translation=[10, 0, 0])
        check_atom_first = frag.split_frags[0]._molecule.molkit_molecule.allAtoms.get(check_atom_name)
        f_first = frag.split_frags[0]
        for i, f in enumerate(frag.split_frags[1:]):
            prev_f = frag.split_frags[i-1]
            check_atom = f._molecule.molkit_molecule.allAtoms.get(check_atom_name)
            assert check_atom.coords[0] != check_atom_first.coords[0]
            assert f._molecule.molkit_molecule.pdbqt_str != f_first._molecule.molkit_molecule.pdbqt_str
            # but when we adjust / extend the conformation they equal out
            f_first._expand_conf_(f)
            assert round(check_atom.coords[0][0], 3) == round(check_atom_first.coords[0][0], 3)

        

