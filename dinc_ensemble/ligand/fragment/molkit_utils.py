
from mglutil.math.statetocoords import StateToCoords
from MolKit import Read as MolKitRead, WritePDB
from MolKit.chargeCalculator import GasteigerChargeCalculator
from MolKit.hydrogenBuilder import HydrogenBuilder
from MolKit.molecule import AtomSet
from MolKit.torTree import TorTree

from AutoDockTools.atomTypeTools import AutoDock4_AtomTyper, LonepairMerger, NonpolarHydrogenMerger
from AutoDockTools.AutoDockBondClassifier import AutoDockBondClassifier
from AutoDockTools.cluster import Clusterer
from AutoDockTools.Conformation import Conformation
from AutoDockTools.Docking import Docking
from AutoDockTools.MoleculePreparation import AD4LigandWriter, AD4FlexibleReceptorPreparation

from copy import deepcopy
from numpy import absolute, arctan2, array, cross, degrees, dot, linspace, matrix
from numpy.linalg import norm
from pathlib import Path

from dinc_ensemble.parameters.fragment import *
from dinc_ensemble.docking.dinc_dock_engine import dinc_probe_vina
from dinc_ensemble.parameters.receptor import DincReceptorParams
from dinc_ensemble.docking.utils.utils_prep_receptor import prepare_bbox

import random

# Build a torsion tree for the given fragment, rooted at the given atom.
#   fragment (modified): molecule for which we will build the torsion tree
#   root_name: name of the atom that will be the root of the torsion tree
#   ligand: ligand from which the fragment was extracted
#   previous_fragment: predecessor of the given fragment in the ligand's fragment sequence
#
def make_tor_tree(fragment, root_name, ligand=None, previous_fragment=None):

    # classify all the fragment's bonds, and initialize their attributes
    classes = AutoDockBondClassifier().classify(fragment.allAtoms.bonds[0])
    fragment.allAtoms.bonds[0].possibleTors = False  # bond associated with a possible torsion
    fragment.allAtoms.bonds[0].activeTors = False    # bond associated with an active torsion
    fragment.allAtoms.bonds[0].incycle = False       # bond that belongs to a cycle
    fragment.allAtoms.bonds[0].hrotator = False      # bond that rotates only hydrogens
    fragment.allAtoms.bonds[0].amdbond = False       # amide bond

    # set the values of the fragment's bond attributes, based on the classification
    classes['cycle'].incycle = True
    classes['amide'].amdbond = True
    for a in classes['hydrogenRotators']:  # this class contains atoms, not bonds
        a.bonds.get(lambda b: b.neighborAtom(a).element != 'H').hrotator = True
    torsions = classes['rotatable'].get(lambda b: not (b.incycle or b.amdbond or b.hrotator))

    # if the ligand has been provided, use it to check the fragment's torsions
    # (in some cases, the fragment might have torsions that are not in the ligand)
    if ligand:
        ligand_torsions = [b for b in ligand.allAtoms.bonds[0] if b.possibleTors]
        for tor in torsions:
            tor_atoms = set([tor.atom1.name, tor.atom2.name])
            if not [b for b in ligand_torsions if set([b.atom1.name, b.atom2.name]) == tor_atoms]:
                torsions.remove(tor)

    # if the previous fragment has been provided,
    # check that all its possible torsions are in the current fragment's torsion tree
    if previous_fragment:
        for tor in [b for b in previous_fragment.allAtoms.bonds[0] if b.possibleTors]:
            bond = next(b for b in fragment.allAtoms.bonds[0] \
                if set([b.atom1.name, b.atom2.name]) == set([tor.atom1.name, tor.atom2.name]))
            if bond not in torsions:
                torsions.append(bond)

    # create the fragment's torsion tree
    torsions.possibleTors = True
    torsions.activeTors = True
    fragment.TORSDOF = len(torsions)    # number of possible torsions
    fragment.torscount = len(torsions)  # number of active torsions

    fragment.ROOT = fragment.allAtoms.get(lambda x: x.name == root_name)[0]
    fragment.ROOT.rnum0 = 0
    # reset the torsion tree index!
    delattr(fragment.allAtoms, 'tt_ind')
    fragment.torTree = TorTree(fragment.parser, fragment.ROOT)

    # check that the torsion tree has been properly built
    check_tor_tree_parent(fragment.torTree.rootNode)
    # delete some nodes from the TorTree which are not in the fragment ??
    # prune_nodes(fragment.torTree, fragment)
    check_tor_tree(fragment.torTree.rootNode)

    # sort the fragment's atoms based on their index in the torsion tree
    # (this is needed because of how StateToCoords works; see mglutil.math.statetocoords)
    fragment.allAtoms = AtomSet(sorted(fragment.allAtoms, key=lambda a: a.tt_ind))


# Prune the nodes of atoms that are not in the ligand
def prune_nodes(torTree, lig):
    remove_nodes = []
    for node in torTree.torsionMap:
        for tt_ind in node.atomList:
            if tt_ind >= len(lig.allAtoms):
                remove_nodes.append(node)
                break
    for node in remove_nodes:
        torTree.torsionMap.remove(node)
        #remove from parent
        parent = node.parent
        parent.children.remove(node)

# Check that the torsion tree has been properly built, and correct it if necessary.
# This is needed because of a bug in the TorTree class from MolKit.torTree.
#
def check_tor_tree_parent(node):
    for child in node.children:
        child.parent = node
        check_tor_tree_parent(child)

# Check that the torsion tree has been properly built, and correct it if necessary.
# This is needed because of a bug in the TorTree class from MolKit.torTree.
#
def check_tor_tree(node, parent=None):
    if parent and node.bond[1] not in parent.atomList:
        parent.atomList.append(node.bond[1])
    for child in node.children:
        check_tor_tree(child, node)

# Perform a breadth-first search in the torsion tree of the given ligand to collect all its nodes
# (the root node is excluded because it does not correspond to any torsion).
# Return the list of nodes from the torsion tree.
#
def get_tor_tree_nodes(ligand):
    N = []
    current_level = [ligand.torTree.rootNode]
    while current_level:
        next_level = [c for n in current_level for c in n.children]
        N.extend(next_level)
        current_level = next_level
    return N



# Split a ligand into a sequence of overlapping fragments, each having their own torsion tree.
# Return the sequence of overlapping fragments extracted from the given ligand.
#   ligand (modified): molecule that is split into fragments and whose torsion tree is constructed
#   params: parameters of the docking job
#
def split_into_fragments(ligand, 
                         root_name,
                         frag_size,
                         frag_new):
    # create the ligand's torsion tree, and get all its nodes
    unprocessed_ligand = deepcopy(ligand)
    make_tor_tree(ligand, root_name)
    N = get_tor_tree_nodes(ligand)

    # otherwise, construct the sequence of fragments
    mol_fragments = []
    # cut does not include the root, so hence, frag_size - 1 
    cut = frag_size
    while cut < len(N):
        # the fragment is a copy of the unprocessed ligand; it will have its own torsion tree
        fragment = deepcopy(unprocessed_ligand)

        # delete the relevant set of atoms to create the current fragment
        # (atomList contains the atoms that are directly affected by a torsion, i.e., a node)
        A = [a for n in N[cut:] for i in n.atomList for a in ligand.allAtoms if a.tt_ind == i]
        del_atoms = [a for at in A for a in fragment.allAtoms if a.name == at.name]
        fragment.allAtoms -= AtomSet(del_atoms)
        
        # remove the bonds associated with the deleted atoms
        map(lambda a: map(lambda b: b.neighborAtom(a).bonds.remove(b), a.bonds), del_atoms)
        # remove any additional bonds maybe associated with atoms that are not in the molecule
        del_ab = []
        for a in fragment.allAtoms:
            for b in a.bonds:
                if not b.atom2 in fragment.allAtoms:
                    del_ab.append((a, b))
                if not b.atom1 in fragment.allAtoms:
                    del_ab.append((a, b))
        for a, b in del_ab:
            a.bonds.remove(b)

        # create the fragment's torsion tree, and add it to the fragment sequence
        if not mol_fragments:
            make_tor_tree(fragment, root_name, ligand)
        else:
            make_tor_tree(fragment, root_name, ligand,  mol_fragments[-1])
        
        mol_fragments.append(fragment)

        cut += frag_new

    mol_fragments.append(ligand)
    return mol_fragments


# Compute the dihedral angle between the points with Cartesian coordinates x1, x2, x3, x4.
# Return the angle of the x2-x3 bond in degrees.
# Raise a ValueError if the angle is not defined.
#
def dihedral(x1, x2, x3, x4):
    b1 = array(x1) - array(x2)
    b2 = array(x4) - array(x3)
    b3 = array(x3) - array(x2)
    b3 /= norm(b3)
    v1 = b1 - dot(b1, b3) * b3
    v2 = b2 - dot(b2, b3) * b3
    if norm(v1) < 0.001 or norm(v2) < 0.001:
        raise ValueError("Dihedral angle undefined: degenerate points")
    return degrees(arctan2(dot(cross(b3, v1), v2), dot(v1, v2)))

# Randomly choose which bonds from the previous fragment will be inactive in the new fragment
# And update the number of active DoFs (torscount)
# NB: the actual number of new bonds in the new fragment is: new_bonds = all_bonds - previous_bonds
#     and the number of "previous bonds" that have to be kept active is frag_size - new_bonds

def freeze_molkit_bonds(prev_frag, new_frag, frag_size, frag_new):
    previous_bonds = [(b.atom1, b.atom2) for b in prev_frag.allAtoms.bonds[0] if b.possibleTors]
    allBonds = [b for b in new_frag.allAtoms.bonds[0] if b.possibleTors]
    # activate all previous bonds in case they were frozen
    for b in allBonds:
        b.activeTors = True
    for _ in range(frag_size - (len(allBonds) - len(previous_bonds))):
        previous_bonds.pop(random.randrange(len(previous_bonds)))
    for b in previous_bonds:
        bond_atoms = set([b[0].name, b[1].name])
        next(b for b in allBonds if set([b.atom1.name, b.atom2.name]) == bond_atoms).activeTors = False
    new_frag.torscount = frag_size



# Expand the given conformation of the previous fragment into the new fragment and
# randomly determine which bonds from the previous fragment are kept active in the new fragment.
# In practice, the expansion is done by applying the given conformation to the new fragment:
# this kinematic transformation involves applying the torsions of the conformation to the new
# fragment (i.e., to the part of the new fragment that is shared with the previous fragment)
# as well as translating and rotating the new fragment so that it fits the given conformation.
# Since applying torsions can create errors that easily propagate along the kinematic chain,
# after the expansion, we correct atom positions in the fragment using coordinates from the
# conformation.
#   conf: conformation that has to be expanded
#   new_frag (modified): fragment into which the given conformation should be expanded
#   prev_frag: fragment corresponding to the given conformation
#   params: parameters of the docking job
#
def expand_conf_to_fragment(conf, new_frag, prev_frag,
                            root_name, frag_size):

    # activate all bonds for this purpose
    allBonds = [b for b in new_frag.allAtoms.bonds[0] if b.possibleTors]
    for b in allBonds:
        b.activeTors = 1
    # activate all bonds for this purpose
    allBonds = [b for b in prev_frag.allAtoms.bonds[0] if b.possibleTors]
    for b in allBonds:
        b.activeTors = 1
    
    # calculate the translation we will apply to the new fragment so that it fits the conformation
    conf.mol.allAtoms.updateCoords(conf.coords, 0)
    conf_root = next(a for a in conf.mol.allAtoms if a.name == root_name)
    frag_root = next(a for a in new_frag.allAtoms if a.name == root_name)
    translation = array(conf_root.coords) - array(frag_root.coords)

    # compute the differences in torsion angles between the conformation and the new fragment:
    # find four atoms defining a torsion, at2-at0-at1-at3, where at0-at1 is a common rotatable bond;
    # for other bonds (i.e., bonds that belong only to the new fragment), store a null difference
    common_bonds = [(b.atom1.name, b.atom2.name) for b in prev_frag.allAtoms.bonds[0] \
                    if b.possibleTors]
    common_bonds.extend([(a2, a1) for (a1, a2) in common_bonds])
    torsions = []
    for node in new_frag.torTree.torsionMap:
        at = [a for i in node.bond for a in new_frag.allAtoms if a.tt_ind == i]  # at0 and at1
        at = list(set(at))
        if (at[0].name, at[1].name) in common_bonds:
            atm = [next(x for x in conf.mol.allAtoms if x.name == a.name) for a in at]
            atm = list(set(atm))
            # at2:
            atm3 = next(a for b in atm[0].bonds for a in set([b.atom1, b.atom2]) - set(atm))
            atm4 = next(a for b in atm[1].bonds for a in set([b.atom1, b.atom2]) - set(atm))
            at3 = next(x for x in new_frag.allAtoms if x.name == atm3.name)
            at4 = next(x for x in new_frag.allAtoms if x.name == atm4.name)
            at.append(at3)
            at.append(at4)
            atm.append(atm3)
            atm.append(atm4)
            try:
                frag_tor = dihedral(at[2].coords, at[0].coords, at[1].coords, at[3].coords)
                conf_tor = dihedral(atm[2].coords, atm[0].coords, atm[1].coords, atm[3].coords)
                torsions.append(conf_tor - frag_tor)
            except ValueError:
                raise
        else:
            torsions.append(0)

    # apply the translation and torsions to the new fragment so that it fits the conformation
    new_frag.stoc = StateToCoords(new_frag, [0, 0, 0], 0)
    Conformation(new_frag, [0, 0, 0], translation, [1, 0, 0, 0], torsions).getCoords()

    # create a reference frame associated with the conformation
    Ac = conf.mol.allAtoms[0:3]
    Xc = array(Ac[1].coords) - array(Ac[0].coords)
    Xc /= norm(Xc)
    Yc = cross(Xc, array(Ac[2].coords) - array(Ac[0].coords))
    Yc /= norm(Yc)
    Zc = cross(Xc, Yc)

    # create the corresponding reference frame associated with the new fragment
    Af = [next(x for x in new_frag.allAtoms if x.name == a.name) for a in Ac]
    Xf = array(Af[1].coords) - array(Af[0].coords)
    Xf /= norm(Xf)
    Yf = cross(Xf, array(Af[2].coords) - array(Af[0].coords))
    Yf /= norm(Yf)
    Zf = cross(Xf, Yf)

    # rotate the new fragment so that it fits the conformation
    rotation = matrix([Xc, Yc, Zc]).T * matrix([Xf, Yf, Zf])
    origin = matrix(frag_root.coords).T
    for a in new_frag.allAtoms:
        a.coords = list((rotation * (matrix(a.coords).T - origin) + origin).flat)

    # correct mistakes committed when applying the conformation to the new fragment:
    # 1) to atoms belonging only to the new fragment, we apply a translation that corresponds to the
    # correction implicitly applied to their closest 'parent' atom belonging to the previous fragment
    '''
    corrections = {}
    new_atoms = list(set([a.name for a in new_frag.allAtoms]) - set([a.name for a in prev_frag.allAtoms]))
    while new_atoms:
        a_name = new_atoms.pop(0)
        at = next(x for x in new_frag.allAtoms if x.name == a_name)
        Af = [a for b in at.bonds for a in [b.neighborAtom(at)] if a.name not in new_atoms]
        # if at has a 'parent' atom which is not part of the remaining new atoms
        if Af:
            Ac = conf.mol.allAtoms.get(lambda x: x.name == Af[0].name)
            if Ac:
                # if at's parent atom belongs to the previous fragment,
                # its associated correction is the difference between the conformation coordinates
                # and the fragment coordinates of its parent
                corrections[at] = array(Ac[0].coords) - array(Af[0].coords)
            else:
                # otherwise, its associated correction is equal to its parent's correction
                corrections[at] = corrections[Af[0]]
        else:
            new_atoms.append(a_name)
    for at in corrections.keys():
        at.coords += corrections[at]
    # 2) to atoms belonging to the previous fragment, we assign the coordinates from the conformation
    for a in conf.mol.allAtoms:
        next(x for x in new_frag.allAtoms if x.name == a.name).coords = a.coords
    '''
    # randomly choose which bonds from the previous fragment will be inactive in the new fragment
    # and update the number of active DoFs (torscount)
    # NB: the actual number of new bonds in the new fragment is: new_bonds = all_bonds - previous_bonds
    #     and the number of "previous bonds" that have to be kept active is frag_size - new_bonds
    previous_bonds = [(b.atom1, b.atom2) for b in prev_frag.allAtoms.bonds[0] if b.possibleTors]
    allBonds = [b for b in new_frag.allAtoms.bonds[0] if b.possibleTors]
    for _ in range(frag_size - (len(allBonds) - len(previous_bonds))):
        if len(previous_bonds) > 0:
            previous_bonds.pop(random.randrange(len(previous_bonds)))
    for b in previous_bonds:
        bond_atoms = set([b[0].name, b[1].name])
        next(b for b in allBonds if set([b.atom1.name, b.atom2.name]) == bond_atoms).activeTors = False
    new_frag.torscount = frag_size





# Select a root atom for a ligand's torsion tree, among the heavy atoms of this ligand,
# based on the selection protocol chosen by the user:
#  - user: the root atom is user-specified
#  - random: the root atom is chosen randomly
#  - automatic: the root atom is chosen automatically
#   lgd: copy of the ligand for which we need to determine a root atom
#   params (modified): parameters of the docking job (the root atom is updated)
#
def select_root_atom(lgd, 
                     root_type,
                     root_name,
                     root_auto,
                     frag_size,
                     # following arguments needed just for probe option
                     probe_dir = "tmp_probe",
                     dince_frag = None,
                     dince_rec = None):
    heavy_atoms = lgd.allAtoms.get(lambda a: a.element != 'H')

    # user mode: the root atom is specified by the user
    if root_type == DINC_ROOT_TYPE.USER:
        if not root_name:
            raise ValueError("DincError: No root atom has been specified")

    # random mode: the root atom is chosen randomly among the ligand's heavy atoms
    elif root_type == DINC_ROOT_TYPE.RANDOM:
        root_name = random.choice(heavy_atoms).name

    # automatic mode: the root atom is chosen automatically, based on the selected strategy
    elif root_type == DINC_ROOT_TYPE.AUTO:
        lgd_copy = deepcopy(lgd)
        # check for special case: when the initial fragment is the whole ligand, any root is fine
        make_tor_tree(lgd_copy, heavy_atoms[0].name)
        if frag_size >= lgd_copy.torscount:
            root_name = heavy_atoms[0].name

        # first: the root atom is the first heavy atom of the ligand
        elif root_auto == DINC_ROOT_AUTO.FIRST:
            root_name = heavy_atoms[0].name

        # last: the root atom is the last heavy atom of the ligand
        elif root_auto == DINC_ROOT_AUTO.LAST:
            root_name = heavy_atoms[-1].name

        # largest: the root atom is the heavy atom producing the largest initial fragment,
        # where the size of a fragment is defined as the number of heavy atoms it contains
        elif root_auto == DINC_ROOT_AUTO.LARGEST:
            max_size = 0
            for atom in heavy_atoms:
                make_tor_tree(lgd_copy, atom.name)
                N = get_tor_tree_nodes(lgd_copy)
                c = min(frag_size, len(N))
                A = [a for n in N[c:] for i in n.atomList for a in lgd_copy.allAtoms if a.tt_ind == i]
                initial_fragment = [a for a in lgd_copy.allAtoms if a not in A and a.element != 'H']
                if len(initial_fragment) > max_size:
                    max_size = len(initial_fragment)
                    root_name = atom.name

        # H_bonds: the root is the atom maximizing the H-bond potential in the initial fragment
        elif root_auto == DINC_ROOT_AUTO.H_BONDS:
            max_bonds = -1
            for atom in heavy_atoms:
                make_tor_tree(lgd_copy, atom.name)
                N = get_tor_tree_nodes(lgd_copy)
                c = min(frag_size, len(N))
                A = [a for n in N[c:] for i in n.atomList for a in lgd_copy.allAtoms if a.tt_ind == i]
                H_bonds = [a for a in lgd_copy.allAtoms if a not in A and a.element in ['F', 'N', 'O']]
                if len(H_bonds) > max_bonds:
                    max_bonds = len(H_bonds)
                    root_name = atom.name

        elif root_auto == DINC_ROOT_AUTO.PROBE:
            # 1 - a random fragmentation
            dince_frag._root_atom_name = dince_frag._molecule.molkit_molecule.allAtoms[0].name
            dince_frag.split_to_fragments()
            # 2 - extract leaves
            dince_frag.split_leafs(leaf_size=frag_size)
            probe_dir = Path(probe_dir)
            probe_dir.mkdir(exist_ok=True, parents=True)
            probe_res = dince_frag.write_pdbqt_frags(str(probe_dir), leaf=True)
            probe_pdbqts = " ".join([str(i) for i in probe_res["leaf_pdbqt_file"]])
            rec_fname = probe_dir / "rec.pdbqt"
            with open(rec_fname, "w") as rec_f:
                rec_f.write(dince_rec.pdbqt_str)
            bbox_fname = probe_dir / "bbox.txt"
            full_lig_fname = "_".join(str(probe_res["leaf_pdbqt_file"][0]).split("_")[:-2])+"_full.pdbqt"
            prepare_bbox(full_lig_fname, 
                            [str(rec_fname)], 
                            bbox_fname,
                            DincReceptorParams())
            probe_out_file = probe_dir / "probe_out.pdbqt"
            # 3 - probe
            res_df, leaf_idx = dinc_probe_vina(probe_pdbqts, str(rec_fname), 
                str(probe_out_file), str(bbox_fname),
                4,  20, 0, 10, 10, 0, 4)
            root_name = dince_frag.leaf_frags[leaf_idx].allAtoms[0].name
        else:
            raise ValueError("DincError: Invalid automatic root selection protocol.")
    else:
        raise ValueError("DincError: Inappropriate root atom selection protocol.")
    
    return root_name


def write_fragment_pdbqt(fragment, path):

    with open(path, "w") as f:
        f.write(fragment.pdbqt_str)
                


