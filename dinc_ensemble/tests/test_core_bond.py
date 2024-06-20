from dinc_ensemble import Atom, Bond
import numpy as np

def test_bond(ligand):

    ligand_molkit_atoms = list(ligand.molkit_molecule.allAtoms)
    ligand_dinc_atoms = [Atom(molkit_atom) for molkit_atom in ligand_molkit_atoms]
    for dinc_atom in ligand_dinc_atoms:
        molkit_bonds = dinc_atom._molkit_bonds
        bondize = np.vectorize(lambda x: Bond(x))
        bondize(molkit_bonds)

    print(ligand_molkit_atoms)

    assert len(ligand_molkit_atoms) == len(ligand_dinc_atoms)