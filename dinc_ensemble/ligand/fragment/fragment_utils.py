from numpy import matrix, array, sum, degrees, arctan2, dot, cross
from numpy.linalg import norm
from math import ceil
from os import path
from copy import deepcopy
import random

from MolKit.molecule import AtomSet
from mglutil.math.statetocoords import StateToCoords
from AutoDockTools.Conformation import Conformation
from AutoDockTools.MoleculePreparation import AD4LigandWriter

from .ligand_torsions import make_tor_tree, get_tor_tree_nodes
from dinc_ensemble.parameters.fragment import *
import logging
logger = logging.getLogger('dinc_ensemble.ligand.fragment')

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
                     frag_size):
    heavy_atoms = lgd.allAtoms.get(lambda a: a.element != "H")
    selected_root_name = None
    # user mode: the root atom is specified by the user
    if root_type == "USER":
        if not root_name:
            raise ValueError("DincError: No root atom has been specified")
        else:
            selected_root_name = root_name

    # random mode: the root atom is chosen randomly among the ligand's heavy atoms
    elif root_type == "RANDOM":
        selected_root_name = random.choice(heavy_atoms).name

    # automatic mode: the root atom is chosen automatically, based on the selected strategy
    elif root_type == "AUTO":
        # check for special case: when the initial fragment is the whole ligand, any root is fine
        make_tor_tree(lgd, heavy_atoms[0].name)
        if frag_size >= lgd.torscount:
            selected_root_name = heavy_atoms[0].name

        # first: the root atom is the first heavy atom of the ligand
        elif root_auto == "FIRST":
            selected_root_name = heavy_atoms[0].name

        # last: the root atom is the last heavy atom of the ligand
        elif root_auto == "LAST":
            selected_root_name = heavy_atoms[-1].name

        # largest: the root atom is the heavy atom producing the largest initial fragment,
        # where the size of a fragment is defined as the number of heavy atoms it contains
        elif root_auto == "LARGEST":
            max_size = 0
            for atom in heavy_atoms:
                make_tor_tree(lgd, atom.name)
                N = get_tor_tree_nodes(lgd)
                c = min(frag_size, len(N))
                A = [
                    a
                    for n in N[c:]
                    for i in n.atomList
                    for a in lgd.allAtoms
                    if a.tt_ind == i
                ]
                initial_fragment = [
                    a for a in lgd.allAtoms if a not in A and a.element != "H"
                ]
                if len(initial_fragment) > max_size:
                    max_size = len(initial_fragment)
                    selected_root_name = atom.name

        # H_bonds: the root is the atom maximizing the H-bond potential in the initial fragment
        elif root_auto == "H_BONDS":
            max_bonds = -1
            for atom in heavy_atoms:
                make_tor_tree(lgd, atom.name)
                N = get_tor_tree_nodes(lgd)
                c = min(frag_size, len(N))
                A = [
                    a
                    for n in N[c:]
                    for i in n.atomList
                    for a in lgd.allAtoms
                    if a.tt_ind == i
                ]
                H_bonds = [
                    a
                    for a in lgd.allAtoms
                    if a not in A and a.element in ["F", "N", "O"]
                ]
                if len(H_bonds) > max_bonds:
                    max_bonds = len(H_bonds)
                    selected_root_name = atom.name

        else:
            raise ValueError("DincError: Invalid automatic root selection protocol.")
    else:
        raise ValueError("DincError: Inappropriate root atom selection protocol.")
    return selected_root_name


# Split a ligand into a sequence of overlapping fragments, each having their own torsion tree.
# Return the sequence of overlapping fragments extracted from the given ligand.
#   ligand (modified): molecule that is split into fragments and whose torsion tree is constructed
#   params: parameters of the docking job
#
def split_into_fragments(ligand, 
                         root_name: str, 
                         frag_size: int,
                         frag_new: int):
    # create the ligand's torsion tree, and get all its nodes
    unprocessed_ligand = deepcopy(ligand)
    make_tor_tree(ligand, root_name)
    N = get_tor_tree_nodes(ligand)

    # construct the sequence of fragments
    mol_fragments = []
    cut = frag_size
    while cut < len(N):
        # the fragment is a copy of the unprocessed ligand; it will have its own torsion tree
        fragment = deepcopy(unprocessed_ligand)

        # delete the relevant set of atoms to create the current fragment
        # (atomList contains the atoms that are directly affected by a torsion, i.e., a node)
        A = [
            a
            for n in N[cut:]
            for i in n.atomList
            for a in ligand.allAtoms
            if a.tt_ind == i
        ]
        del_atoms = [a for at in A for a in fragment.allAtoms if a.name == at.name]
        fragment.allAtoms -= AtomSet(del_atoms)

        # remove the bonds associated with the deleted atoms
        list(
            map(
                lambda a: [b.neighborAtom(a).bonds.remove(b) for b in a.bonds if b in b.neighborAtom(a).bonds],
                del_atoms,
            )
        )

        # create the fragment's torsion tree, and add it to the fragment sequence
        if mol_fragments:
            make_tor_tree(fragment, root_name, ligand, mol_fragments[-1])
            # randomly choose which bonds from the previous fragment will be inactive in the new fragment
            # and update the number of active DoFs (torscount)
            # NB: the actual number of new bonds in the new fragment is: new_bonds = all_bonds - previous_bonds
            #     and the number of "previous bonds" that have to be kept active is frag_size - new_bonds
            prev_frag = mol_fragments[-1]
            previous_bonds = [(b.atom1, b.atom2) for b in prev_frag.allAtoms.bonds[0] if b.possibleTors]
            new_frag = fragment
            allBonds = [b for b in new_frag.allAtoms.bonds[0] if b.possibleTors]
            for _ in range(frag_size - (len(allBonds) - len(previous_bonds))):
                previous_bonds.pop(random.randrange(len(previous_bonds)))
            for b in previous_bonds:
                bond_atoms = set([b[0].name, b[1].name])
                next(b for b in allBonds if set([b.atom1.name, b.atom2.name]) == bond_atoms).activeTors = False
            new_frag.torscount = frag_size

        mol_fragments.append(fragment)
        cut += frag_new

    mol_fragments.append(ligand)
    return mol_fragments


# Select a root atom for the ligand's torsion tree, using the protocol chosen by the user
# Split the ligand into a sequence of overlapping fragments and save its torsion tree
# Return the  sequence of overlapping fragments
#   ligand (modified): molecule that is split into fragments and whose torsion tree is constructed
#   back_trace: dictionary listing, for each ligand's atom, the atoms with which it shares a bond
#   params: parameters of the docking job
#   ligand_fname: the filename of the linput ligabd
#   dir_path: working directory
def fragment_ligand(ligand, 
                    root_type,
                    root_name,
                    root_auto,
                    frag_size, 
                    frag_new):
    # Select a root atom for the ligand's torsion tree, using the protocol chosen by the user

    selected_root_name = select_root_atom(deepcopy(ligand), 
                                            root_type=root_type,
                                            root_name=root_name,
                                            root_auto=root_auto,
                                            frag_size=frag_size)

    logger.info("Selected root atom: {}".format(selected_root_name))

    # Split the ligand into a sequence of overlapping fragments and save its torsion tree
    mol_frags = split_into_fragments(ligand, 
                                     selected_root_name,
                                     frag_size,
                                     frag_new)
    AD4LigandWriter().write(ligand, "", inmem=True)
    return (mol_frags, selected_root_name)


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
#
def expand_conf_to_fragment(conf, 
                            new_frag, 
                            prev_frag, 
                            root_name,
                            frag_size):
    # calculate the translation we will apply to the new fragment so that it fits the conformation
    conf.mol.allAtoms.updateCoords(conf.coords, 0)
    conf_root = next(a for a in conf.mol.allAtoms if a.name == root_name)
    frag_root = next(a for a in new_frag.allAtoms if a.name == root_name)
    translation = array(conf_root.coords) - array(frag_root.coords)

    # compute the differences in torsion angles between the conformation and the new fragment:
    # find four atoms defining a torsion, at2-at0-at1-at3, where at0-at1 is a common rotatable bond;
    # for other bonds (i.e., bonds that belong only to the new fragment), store a null difference
    common_bonds = [
        (b.atom1.name, b.atom2.name)
        for b in prev_frag.allAtoms.bonds[0]
        if b.possibleTors
    ]
    common_bonds.extend([(a2, a1) for (a1, a2) in common_bonds])
    torsions = []
    for node in new_frag.torTree.torsionMap:
        at = [
            a for i in node.bond for a in new_frag.allAtoms if a.tt_ind == i
        ]  # at0 and at1
        if (at[0].name, at[1].name) in common_bonds:
            # at2:
            at.append(
                next(a for b in at[0].bonds for a in set([b.atom1, b.atom2]) - set(at))
            )
            # at3:
            at.append(
                next(a for b in at[1].bonds for a in set([b.atom1, b.atom2]) - set(at))
            )
            atm = [next(x for x in conf.mol.allAtoms if x.name == a.name) for a in at]
            try:
                frag_tor = dihedral(
                    at[2].coords, at[0].coords, at[1].coords, at[3].coords
                )
                conf_tor = dihedral(
                    atm[2].coords, atm[0].coords, atm[1].coords, atm[3].coords
                )
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
    # 1) to atoms belonging only to the new fragment, we apply a translation that correspond to the
    # correction implicitly applied to their closest 'parent' atom belonging to the previous fragment
    corrections = {}
    new_atoms = list(
        set([a.name for a in new_frag.allAtoms])
        - set([a.name for a in prev_frag.allAtoms])
    )
    while new_atoms:
        a_name = new_atoms.pop(0)
        at = next(x for x in new_frag.allAtoms if x.name == a_name)
        Af = [
            a for b in at.bonds for a in [b.neighborAtom(at)] if a.name not in new_atoms
        ]
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
    for at in list(corrections.keys()):
        at.coords += corrections[at]
    # 2) to atoms belonging to the previous fragment, we assign the coordinates from the conformation
    for a in conf.mol.allAtoms:
        next(x for x in new_frag.allAtoms if x.name == a.name).coords = a.coords

    # randomly choose which bonds from the previous fragment will be inactive in the new fragment
    # and update the number of active DoFs (torscount)
    # NB: the actual number of new bonds in the new fragment is: new_bonds = all_bonds - previous_bonds
    #     and the number of "previous bonds" that have to be kept active is frag_size - new_bonds
    previous_bonds = [
        (b.atom1, b.atom2) for b in prev_frag.allAtoms.bonds[0] if b.possibleTors
    ]
    allBonds = [b for b in new_frag.allAtoms.bonds[0] if b.possibleTors]
    for _ in range(frag_size - (len(allBonds) - len(previous_bonds))):
        previous_bonds.pop(random.randrange(len(previous_bonds)))
    for b in previous_bonds:
        bond_atoms = set([b[0].name, b[1].name])
        next(
            b for b in allBonds if set([b.atom1.name, b.atom2.name]) == bond_atoms
        ).activeTors = False
    new_frag.torscount = frag_size


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
