from .ligand_abstract_writer import LigandWriter
from .utils import pybel_convert
from dinc_ensemble.ligand.core import DINCMolecule

from MolKit.molecule import Molecule as MolKitMolecule
from MolKit.pdbWriter import PdbWriter as MolKitPdbWriter

from rdkit import Chem

from os import path, remove

import warnings
import fileinput


class _LigandPDBWriter(LigandWriter):
    extension = ".pdb"

    @classmethod
    def specific_write(cls, molecule: DINCMolecule, filename: str) -> None:
        try:
            writer = MolKitPdbWriter()
            writer.write(filename, molecule.molkit_molecule)
            
        except Exception as e:
            raise IOError(
                "Problem writing the pdb file with MolKitPdbWriter. {}".format(e)
            )



LigandPDBWriter = _LigandPDBWriter()
    

class _LigandMol2Writer(LigandWriter):
    extension = ".mol2"

    @classmethod
    def specific_write(cls, molecule: DINCMolecule, filename: str) -> None:
        try:
            tmp_filename = filename[:filename.rfind(".")]+"_tmp.sdf"
            _LigandSDFWriter.write(molecule, tmp_filename)
            pybel_convert(tmp_filename, "sdf", filename, "mol2", overwrite = True)
            #remove(tmp_filename)
            
        except Exception as e:
            raise IOError(
                "Problem writing the mol2 file with LigandMol2Writer. {}".format(e)
            )


LigandMol2Writer = _LigandMol2Writer()

class _LigandSDFWriter(LigandWriter):
    extension = ".sdf"

    @classmethod
    def specific_write(cls, molecule: DINCMolecule, filename: str) -> None:
        try:
            writer = Chem.SDWriter(filename)
            mol = molecule._rdkit_molecule
            for cid in range(mol.GetNumConformers()):
                writer.write(mol, confId=cid)
             #remove(tmp_filename)
            
        except Exception as e:
            raise IOError(
                "Problem writing the pdb file with LigandSDFWriter. {}".format(e)
            )


LigandSDFWriter = _LigandSDFWriter()
