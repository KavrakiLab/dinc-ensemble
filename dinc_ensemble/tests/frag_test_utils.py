from dinc_ensemble.ligand.core import DINCMolecule
from dinc_ensemble.ligand import DINCFragment
from dinc_ensemble.parameters.fragment import *
from copy import deepcopy

def frag_arguments_valid(frag: DINCFragment):

    # nodes, atoms, bonds initialized properly?
    assert len(frag.bfs_ordered_nodes) == frag._molecule.molkit_molecule.torscount
    assert frag.atoms.shape[0] == len(frag._molecule.molkit_molecule.allAtoms)
    assert frag.bonds.shape[0] == len(frag._molecule.molkit_molecule.allAtoms.bonds[0])

def init_fragments_multi_ligand(ligands_list_list:
                                list[list[DINCMolecule]],
                                root_type = DEFAULT_DINC_ROOT_TYPE,
                                root_auto = DEFAULT_DINC_ROOT_AUTO,
                                root_atom_name = None,
                                write_pdbqt = False,
                                write_svg = False,
                                get_df = False):
    for ligands_list in ligands_list_list:
        for tmp_ligand in ligands_list:
            ligand = deepcopy(tmp_ligand)
            root_atom_name = ligand.molkit_molecule.allAtoms[0].name
            active_bonds = ligand.bonds[ligand.bonds.activeTors_==1]
            ntors = len(active_bonds)
            print("Fragment DOF = {}".format(ntors))
            params = DincFragParams()
            params.root_type = root_type
            params.root_auto = root_auto
            params.root_name = root_atom_name
            frag = DINCFragment(ligand, 
                                params)
            assert isinstance(frag, DINCFragment)
            frag_arguments_valid(frag)
            if root_type == DINC_ROOT_TYPE.USER and root_atom_name is None:
                assert frag._root_atom_name == root_atom_name
            if write_pdbqt:
                frag._split_to_fragments_()
                tmp = frag._write_pdbqt_frags_(out_dir="./tmp_test_out")
                assert len(frag.split_frags) == tmp.shape[0]
            if write_svg:
                frag._split_to_fragments_()
                tmp = frag._write_svg_frags_(out_dir="./tmp_test_out")
                assert len(frag.split_frags) == tmp.shape[0]
            if get_df:
                frag._split_to_fragments_()
                tmp = frag._to_df_frags_info_(out_dir="./tmp_test_out")
                assert len(frag.split_frags) == tmp.shape[0]

            
