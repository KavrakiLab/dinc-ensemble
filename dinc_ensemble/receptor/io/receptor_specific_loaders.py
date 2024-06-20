from .receptor_format_registry import ReceptorFormatRegistry
from .receptor_abstract_loader import ReceptorLoader

from MolKit.pdbParser import PdbParser as MolKitPdbParser
from MolKit.pdbParser import PdbqtParser as MolKitPdbqtParser

from ..core import DINCReceptor
from os import path


class _ReceptorPDBLoader(ReceptorLoader):
    extension = ".pdb"

    @classmethod
    def specific_load(cls, filename: str) -> DINCReceptor:
        # 1 - load it into the molkit object
        try:
            parser = MolKitPdbParser(filename)
            proteins = parser.parse()
        except:
            raise IOError(
                "Problem loading the pdb file with MolKitRead. \
                        Check that your pdb file is formatted correctly."
            )
        # if there are more than one elements, load first
        # maybe we should reconsider this?
        protein = proteins[0]
        return DINCReceptor(protein, filename)


ReceptorPDBLoader = _ReceptorPDBLoader()

class _ReceptorPDBQTLoader(ReceptorLoader):
    extension = ".pdbqt"

    @classmethod
    def specific_load(cls, filename: str) -> DINCReceptor:
        # 1 - load it into the molkit object
        try:
            parser = MolKitPdbqtParser(filename)
            proteins = parser.parse()
        except:
            raise IOError(
                "Problem loading the pdb file with MolKitRead. \
                        Check that your pdb file is formatted correctly."
            )
        # if there are more than one elements, load first
        # maybe we should reconsider this?
        protein = proteins[0]
        return DINCReceptor(protein, filename)


ReceptorPDBQTLoader = _ReceptorPDBQTLoader()