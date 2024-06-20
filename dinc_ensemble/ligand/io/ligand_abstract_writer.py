
from typing import Dict
from os import path
from abc import ABC, abstractmethod
from dinc_ensemble.ligand.core import DINCMolecule

class LigandWriter(ABC):
    """
    Abstract base class for writing ligands to different types of output files.

    Attributes
    ----------
    extension : str
        The extension of the file to be written.

    Methods
    -------
    specific_write(molecule:DINCMolecule, filename: str) -> None
        Abstract method to be implemented by child classes that defines how to write the file.
    write(molecule:DINCMolecule, filename: str) -> None
        Calls specific_write and wraps it with additional checks and functionalities.
    """  
    extension = ""

    @classmethod
    @abstractmethod
    def specific_write(cls, molecule: DINCMolecule, filename: str) -> None:
        """
        Abstract method to be implemented by child classes that defines how to write the file.

        Parameters
        ----------
        molecule : DINCMolecule
            The molecule to write.
        filename : str
            The name of the file to be written.

        Returns
        -------
        None
        """         
        ...

    @classmethod
    def write(cls, molecule: DINCMolecule, filename: str) -> None:
        """
        Calls specific_write and wraps it with additional checks and functionalities.

        Parameters
        ----------
        molecule : DINCMolecule
            The molecule to write.
        filename : str
            The name of the file to be written.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the extension of the file does not match the expected extension.
        FileNotFoundError | PermissionError | OSError
            If the file can not be open for writing.
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
        try:
            with open(filename, 'w+') as file:
                file.write("")
        except (FileNotFoundError, PermissionError, OSError) as e:
            print("Error opening file {}".format(filename))
            raise e
        
        # 3 - write using the specific extension-based writer
        cls.specific_write(molecule, filename)