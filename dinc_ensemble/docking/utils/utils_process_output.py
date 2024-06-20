from MolKit import Read as MolKitRead
from mglutil.math.statetocoords import StateToCoords
from AutoDockTools.Conformation import Conformation

from numpy import array
from numpy.linalg import norm


def extract_vina_conformations(docking_file: str):

    ligands = MolKitRead(docking_file)
    ligand = ligands[0]
    end = next(i for i, l in enumerate(ligand.parser.allLines) if l == "ENDMDL\n")
    ligand.parser.allLines = ligand.parser.allLines[2:end]
    # TODO: deal with flexible receptor residues
    conformations = []
    for l in ligands:
        l.ROOT = ligand.ROOT
        l.torTree = ligand.torTree
        l.TORSDOF = len(l.torTree.torsionMap)
        l.stoc = StateToCoords(l, [0, 0, 0], 0)
        c = Conformation(l, [0, 0, 0], [0, 0, 0], [1, 0, 0, 0], [0] * l.TORSDOF)
        c.getCoords()
        c.binding_energy = l.vina_energy # type: ignore
        conformations.append(c)

    #valid_conformations = select_clash_free_confs(conformations)
    return conformations
    

#TODO: figure out bonded atoms
def select_clash_free_confs(conformationSet, bonded_atoms):
    clash_free_confs = []
    for conf in conformationSet:
        clash = False
        conf.mol.allAtoms.updateCoords(conf.coords, 0)
        atomList = list(conf.mol.allAtoms.get(lambda a: a.element != "H"))
        while atomList:
            atom = atomList.pop(0)
            for at in atomList:
                if at.name not in bonded_atoms[atom.name]:
                    dist = norm(array(atom.coords) - array(at.coords))
                    if dist < 1.2 * (atom.bondOrderRadius + at.bondOrderRadius):
                        clash = True
                        break
            if clash:
                break
        if not clash:
            clash_free_confs.append(conf)

    return clash_free_confs