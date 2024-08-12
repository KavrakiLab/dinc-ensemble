from .ligand_abstract_loader import LigandLoader
from .utils import seq_to_peptide

from dinc_ensemble.ligand.core import DINCMolecule
from MolKit.molecule import Molecule as MolKitMolecule
from MolKit import Read as MolKitRead
from openbabel import pybel
from rdkit.Chem.rdmolfiles import MolFromMol2File
from rdkit.Chem import MolToSmiles

from os import path

import warnings
import fileinput


class _LigandMol2Loader(LigandLoader):
    extension = ".mol2"

    @classmethod
    def specific_load(cls, filename: str, config: dict = None) -> DINCMolecule:
        # 1 - load it into the molkit object
        try:
            ligands = MolKitRead(filename)
        except:
            raise IOError(
                "Problem loading the mol2 file with MolKitRead. \
                        Check that your mol2 file is formatted correctly."
            )

        # if there are no elements, raise an exception
        if len(ligands) == 0:
            raise IOError(
                "Problem loading the mol2 file with MolKitRead. \
                            No molecules in the file. \
                            Check that your mol2 file is formatted correctly."
            )

        # if there are more than one elements, warn the used
        if len(ligands) > 1:
            warnings.warn(
                "Seems like the mol2 file has more than one molecule. \
                            DINC will load the first."
            )

        # if there are more than one elements, load first
        # maybe we should reconsider this?
        molkit_ligand = ligands[0]
        ligand = DINCMolecule(molkit_ligand)
        return ligand


LigandMol2Loader = _LigandMol2Loader()


class _LigandPDBLoader(LigandLoader):
    extension = ".pdb"

    @classmethod
    def specific_load(cls, filename: str, config: dict = None) -> DINCMolecule:
        # 1 - load only ATOM lines
        # remove lines that do not contain typical PDB entries (including empty lines)
        for line in fileinput.input(path.join(filename), inplace=True):
            if (
                line.startswith("ATOM")
                or line.startswith("HETATM")
                or line.startswith("CONECT")
            ):
                print(line, end='')
        fileinput.close()

        # 2 - load with pybel and convert to the mol2
        # TODO: (not sure why this loading is done with pybel and not MolKitRead directly?)
        # TODO: postprocessing the ligand and adding hydrogens as a separate functionality
        try:
            molecule = next(pybel.readfile("pdb", path.join(filename)))
            filename_mol2 = filename[:-3] + "mol2"
            molecule.write("mol2", filename_mol2, overwrite=True)
        except Exception:
            raise IOError(
                "Problem loading the PDB file with pybel. \
                        Check that your PDB file is formatted correctly."
            )

        # 3 - load the mol2 file
        dinc_molecule = LigandMol2Loader.load(filename_mol2) 
        return dinc_molecule


LigandPDBLoader = _LigandPDBLoader()


class _LigandFastaLoader(LigandLoader):
    extension = ".fasta"

    @classmethod
    def specific_load(cls, filename: str, config: dict = None) -> DINCMolecule:
        # 1 - convert fasta sequence to a PDB file
        filename_pdb = seq_to_peptide(filename)

        # 2 - load the PDB file
        dinc_molecule = LigandPDBLoader.load(filename_pdb) 
        return dinc_molecule


LigandFastaLoader = _LigandFastaLoader()

class _LigandSDFLoader(LigandLoader):
    extension = ".sdf"

    @classmethod
    def specific_load(cls, filename: str, config: dict = None) -> DINCMolecule:
        
        molecule = next(pybel.readfile("sdf", path.join(filename)))
        filename_mol2 = filename[:-3] + "mol2"
        molecule.write("mol2", filename_mol2, overwrite=True)

        dinc_mol = LigandMol2Loader.load(filename_mol2)
        return dinc_mol


LigandSDFLoader = _LigandSDFLoader()
