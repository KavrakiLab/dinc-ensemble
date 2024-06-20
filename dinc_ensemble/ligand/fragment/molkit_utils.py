from MolKit.torTree import TorTree
from MolKit.molecule import Molecule as MolKitMolecule
from MolKit.molecule import Atom as MolKitAtom
from MolKit.molecule import AtomSet
from numpy import array, dot, arctan2, cross, degrees
from numpy.linalg import norm


def check_tor_tree(node, parent=None) -> None:
    '''
    This is needed because of a bug in MolKit TorTree construction
    '''
    if parent:
        if node.parent is None:
            node.parent = [parent]
        else:
            node.parent.append(parent)
    if parent and node.bond[1] not in parent.atomList:
        parent.atomList.append(node.bond[1])
    for child in node.children:
        check_tor_tree(child, node)

def create_tor_tree(molkit_mol: MolKitMolecule,
                    root_atom: MolKitAtom) -> None:
    
    molkit_mol.torTree = TorTree(molkit_mol.parser, 
                                    root_atom)
    ## try out the other constructor?
    ## no, the other constructor has some issue 
    ## where it does not connect nodes as children
    #fragment.torTree = TorTree(parser = None, rootAtom = fragment.ROOT)

    # check that the torsion tree has been properly built
    check_tor_tree(molkit_mol.torTree.rootNode)

    # sort the fragment's atoms based on their index in the torsion tree
    # (this is needed because of how StateToCoords works; see mglutil.math.statetocoords)
    molkit_mol.allAtoms = AtomSet(
        sorted(
            molkit_mol.allAtoms, key=lambda a: a.tt_ind if hasattr(a, "tt_ind") else False
        )
    )

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
def expand_conf_to_fragment(conf, new_frag, prev_frag, params):
    # calculate the translation we will apply to the new fragment so that it fits the conformation
    conf.mol.allAtoms.updateCoords(conf.coords, 0)
    conf_root = next(a for a in conf.mol.allAtoms if a.name == params["root_name"])
    frag_root = next(a for a in new_frag.allAtoms if a.name == params["root_name"])
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
    for _ in range(params["frag_size"] - (len(allBonds) - len(previous_bonds))):
        previous_bonds.pop(random.randrange(len(previous_bonds)))
    for b in previous_bonds:
        bond_atoms = set([b[0].name, b[1].name])
        next(
            b for b in allBonds if set([b.atom1.name, b.atom2.name]) == bond_atoms
        ).activeTors = False
    new_frag.torscount = params["frag_size"]