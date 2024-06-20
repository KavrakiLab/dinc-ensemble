from MolKit.molecule import AtomSet
from MolKit.torTree import TorTree
from AutoDockTools.AutoDockBondClassifier import AutoDockBondClassifier


# Build a torsion tree for the given fragment, rooted at the given atom.
#   fragment (modified): molecule for which we will build the torsion tree
#   root_name: name of the atom that will be the root of the torsion tree
#   ligand: ligand from which the fragment was extracted
#   previous_fragment: predecessor of the given fragment in the ligand's fragment sequence
#
def make_tor_tree(fragment, root_name, ligand=None, previous_fragment=None):
    # classify all the fragment's bonds, and initialize their attributes
    classes = AutoDockBondClassifier().classify(fragment.allAtoms.bonds[0])
    fragment.allAtoms.bonds[
        0
    ].possibleTors = False  # bond associated with a possible torsion
    fragment.allAtoms.bonds[
        0
    ].activeTors = False  # bond associated with an active torsion
    fragment.allAtoms.bonds[0].incycle = False  # bond that belongs to a cycle
    fragment.allAtoms.bonds[0].hrotator = False  # bond that rotates only hydrogens
    fragment.allAtoms.bonds[0].amdbond = False  # amide bond

    # set the values of the fragment's bond attributes, based on the classification
    classes["cycle"].incycle = True
    classes["amide"].amdbond = True
    for a in classes["hydrogenRotators"]:  # this class contains atoms, not bonds
        a.bonds.get(lambda b: b.neighborAtom(a).element != "H").hrotator = True
    torsions = classes["rotatable"].get(
        lambda b: not (b.incycle or b.amdbond or b.hrotator)
    )

    # if the ligand has been provided, use it to check the fragment's torsions
    # (in some cases, the fragment might have torsions that are not in the ligand)
    if ligand:
        ligand_torsions = [b for b in ligand.allAtoms.bonds[0] if b.possibleTors]
        for tor in torsions:
            tor_atoms = set([tor.atom1.name, tor.atom2.name])
            if not [
                b
                for b in ligand_torsions
                if set([b.atom1.name, b.atom2.name]) == tor_atoms
            ]:
                torsions.remove(tor)

    # if the previous fragment has been provided,
    # check that all its possible torsions are in the current fragment's torsion tree
    if previous_fragment:
        for tor in [b for b in previous_fragment.allAtoms.bonds[0] if b.possibleTors]:
            bond = next(
                b
                for b in fragment.allAtoms.bonds[0]
                if set([b.atom1.name, b.atom2.name])
                == set([tor.atom1.name, tor.atom2.name])
            )
            if bond not in torsions:
                torsions.append(bond)

    # create the fragment's torsion tree
    torsions.possibleTors = True
    torsions.activeTors = True
    fragment.TORSDOF = len(torsions)  # number of possible torsions
    fragment.torscount = len(torsions)  # number of active torsions
    fragment.ROOT = next(a for a in fragment.allAtoms if a.name == root_name)
    fragment.ROOT.rnum0 = 0

    fragment.torTree = TorTree(fragment.parser, fragment.ROOT)

    # check that the torsion tree has been properly built
    check_tor_tree(fragment.torTree.rootNode)

    # sort the fragment's atoms based on their index in the torsion tree
    # (this is needed because of how StateToCoords works; see mglutil.math.statetocoords)
    fragment.allAtoms = AtomSet(
        sorted(
            fragment.allAtoms, key=lambda a: a.tt_ind if hasattr(a, "tt_ind") else False
        )
    )


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
