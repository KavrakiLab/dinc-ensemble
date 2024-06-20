
from typing import Dict
from os import path
from abc import ABC, abstractmethod
from ..core import DINCReceptor

class ReceptorLoader(ABC):
    """
    Abstract base class for loading different types of input files as receptors.

    Attributes
    ----------
    extension : str
        The extension of the file to be loaded.

    Methods
    -------
    specific_load(filename: str) -> DINCReceptor
        Abstract method to be implemented by child classes that defines how to load the file.
    load(filename: str) -> DINCReceptor
        Calls specific_load and wraps it with additional checks and functionalities.
    """  
    extension = ""

    @classmethod
    @abstractmethod
    def specific_load(cls, filename: str) -> DINCReceptor:
        """
        Abstract method to be implemented by child classes that defines how to load the file.

        Parameters
        ----------
        filename : str
            The name of the file to be loaded.

        Returns
        -------
        DINCReceptor
            The loaded protein.
        """         
        ...

    @classmethod
    def load(cls, filename: str) -> DINCReceptor:
        """
        Calls specific_load and wraps it with additional checks and functionalities.

        Parameters
        ----------
        filename : str
            The name of the file to be loaded.

        Returns
        -------
        DINCReceptor
            The loaded protein.

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
        protein = cls.specific_load(filename)
        return protein
