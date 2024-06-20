from dinc_ensemble import Atom

def test_atom(ligand):

    ligand_molkit_atoms = list(ligand.molkit_molecule.allAtoms)
    ligand_dinc_atoms = [Atom(molkit_atom) for molkit_atom in ligand_molkit_atoms]
    print(ligand_molkit_atoms)

    assert len(ligand_molkit_atoms) == len(ligand_dinc_atoms)