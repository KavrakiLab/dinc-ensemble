from dinc_ensemble.ligand.core import DINCMolecule
from dinc_ensemble.ligand import DINCFragment
from dinc_ensemble.parameters.fragment import *
from copy import deepcopy

def frag_arguments_valid(frag: DINCFragment,
                         params: DincFragParams):

    # nodes, atoms, bonds initialized properly?
    # first fragment has torscnt == frag_size
    if len(frag.fragments) > 1:
        assert frag.fragments[0].torscount == params.frag_size
        for i, f in enumerate(frag.fragments[1:-1]): 
            assert (f.torscount - frag.fragments[i].torscount) == params.frag_new
    # is root consistent in all fragments?
    root_name = frag.fragments[0].ROOT.name
    for i, f in enumerate(frag.fragments): 
        assert f.ROOT.name == root_name
    # is the user defined root consistent across all fragments?
    if params.root_type == DINC_ROOT_TYPE.USER:
        for i, f in enumerate(frag.fragments): 
            assert f.ROOT.name == params.root_name
    # are the first/last auto roots defines well?
    elif params.root_type == DINC_ROOT_TYPE.AUTO:
        if len(frag.fragments) > 1:
            frag0 = frag.fragments[0]
            heavy_atoms = frag._molecule.molkit_molecule.allAtoms.get(lambda x: x.element != 'H')
            if params.root_auto == DINC_ROOT_AUTO.FIRST:
                assert frag0.ROOT.name == heavy_atoms[0].name
            #if params.root_auto == DINC_ROOT_AUTO.LAST:
            #    assert frag0.ROOT.name == heavy_atoms[-1].name
    # before freezing bonds all are active
    for f in frag.fragments:
        active_tors =[b for b in f.allAtoms.bonds[0] if b.activeTors]
        possible_tors =[b for b in f.allAtoms.bonds[0] if b.possibleTors]
        assert len(active_tors) == len(possible_tors)
    # test freeze fragment
    frag.freeze_bonds()
    # after freezing bonds each fragment has limited number of active bonds!
    for f in frag.fragments:
        active_tors =[b for b in f.allAtoms.bonds[0] if b.activeTors]
        assert len(active_tors) <= params.frag_size

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
            frag_arguments_valid(frag, params)

            if write_pdbqt:
                tmp = frag.write_pdbqt_frags(out_dir="./tmp_test_out")
                assert len(frag.fragments) == tmp.shape[0]
            #if write_svg:
            #    frag._split_to_fragments_()
            #    tmp = frag._write_svg_frags_(out_dir="./tmp_test_out")
            #    assert len(frag.split_frags) == tmp.shape[0]
            if get_df:
                tmp = frag.to_df_info(out_dir="./tmp_test_out")
                assert len(frag.fragments) == tmp.shape[0]

            
