
from typing import Dict
from os import path
from abc import ABC, abstractmethod
from dinc_ensemble.ligand.core import DINCMolecule

class LigandLoader(ABC):
    """
    Abstract base class for loading different types of input files as ligands.

    Attributes
    ----------
    extension : str
        The extension of the file to be loaded.

    Methods
    -------
    specific_load(filename: str) -> DINCMolecule
        Abstract method to be implemented by child classes that defines how to load the file.
    load(filename: str) -> DINCMolecule
        Calls specific_load and wraps it with additional checks and functionalities.
    """  
    extension = ""

    @classmethod
    @abstractmethod
    def specific_load(cls, filename: str, mk_config: dict = {}) -> DINCMolecule:
        """
        Abstract method to be implemented by child classes that defines how to load the file.

        Parameters
        ----------
        filename : str
            The name of the file to be loaded.

        Returns
        -------
        DINCMolecule
            The loaded molecule.
        """         
        ...

    @classmethod
    def load(cls, filename: str, mk_config: dict = {}) -> DINCMolecule:
        """
        Calls specific_load and wraps it with additional checks and functionalities.

        Parameters
        ----------
        filename : str
            The name of the file to be loaded.

        Returns
        -------
        DINCMolecule
            The loaded molecule.

        Raises
        ------
        TypeError
            If the extension of the file does not match the expected extension.
        FileNotFoundError
            If the file does not exist.
        """         
        # 1 - check if the extension is correct
        (base, extension) = path.splitext(filename)
        if extension != cls.extension:
            raise TypeError(
                "Expecting a {} file. \
                            Wrong extension provided.".format(
                    cls.extension
                )
            )

        # 2 - check if the file exists
        if not path.exists(filename):
            raise FileNotFoundError("File {} not found.".format(filename))

        # 3 - load using the specific extension-based loader
        molecule = cls.specific_load(filename, mk_config)
        return molecule
