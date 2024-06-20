from typing import Dict
from os import path

from .receptor_abstract_loader import ReceptorLoader
#from MolKit.protein import Protein as MolKitProtein
from ..core import DINCReceptor

class _ReceptorFormatRegistry(object):
    """
    A registry for ReceptorLoader classes.

    This class serves as a registry for ReceptorLoader classes. Users can register specific receptor loaders that they define.
    The keys are the extensions and the ReceptorLoader is the value. This class is inspired by the format registry template used in the mdtraj library.

    Attributes
    ----------
    loaders : Dict[str, ReceptorLoader]
        A dictionary where the keys are the extensions and the values are the ReceptorLoader classes.

    Methods
    -------
    register_loader(extension: str, loader: ReceptorLoader)
        Registers a ReceptorLoader class for a specific extension.

    
    Examples
    --------
    Let's first define a `ReceptorLoader` subclass:
    >>> class PDBReceptorLoader(ReceptorLoader):
    >>>  extension = ".pdb"
     
    >>>  @classmethod
    >>>  def specific_load(cls, filename: str) -> DINCReceptor:
    >>>      # Load a PDB file
    >>>      pass
    
    Now, we can register these loaders with the ReceptorFormatRegistry:
    >>> ReceptorFormatRegistry.register_loader(".pdb", PDBReceptorLoader)
    
    In this example, `load_receptor` will use the appropriate loader for 
    the file extension of the provided filename. If no loader is registered for that extension, 
    it raises an error.

    This approach allows you to easily extend the functionality of your program to support 
    new file formats: simply define a new `ReceptorLoader` subclass and register it with the 
    `ReceptorFormatRegistry`.
    """

    loaders: Dict[str, ReceptorLoader] = {}

    @classmethod
    def register_loader(cls, extension: str, loader: ReceptorLoader):
        """
        Registers a ReceptorLoader class for a specific extension.

        Parameters
        ----------
        extension : str
            The extension for which the ReceptorLoader class should be registered.
        loader : ReceptorLoader
            The ReceptorLoader class to be registered for the specified extension.
        """
        cls.loaders[extension] = loader


# Make a single instance of this class, and then
# get rid of the class object. This should be
# treated as a singleton
ReceptorFormatRegistry = _ReceptorFormatRegistry()


def load_receptor(filename: str) -> DINCReceptor:
    """
    Load a receptor from a file.

    This function loads a receptor from a file specified by the filename parameter.

    Parameters
    ----------
    filename : str
        The path to the file containing the receptor.

    Returns
    -------
    DINCReceptor
        The loaded receptor as a DINCReceptor object.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If filename parameter is an empty list.
    IOError
        If no loader is found for the specified file extension.

    Examples
    --------
    # Load a receptor from a PDB file
    >>> receptor = load_receptor("receptor.pdb")

    # Load a receptor from a MOL2 file
    >>> receptor = load_receptor("receptor.mol2")

    Notes
    -----
    This function relies on the ReceptorFormatRegistry class to 
    retrieve the appropriate loaders for different file extensions.

    """
    # 1 - check if the file exists
    if not path.exists(filename):
        raise FileNotFoundError("File {} not found.".format(filename))

    # 2 - get the extension of the file
    (base, extension) = path.splitext(filename)

    # 3 - make the needed checks
    if len(set(extension)) == 0:
        raise ValueError(
            "Filename was an empty list"
        )

    # 4 - get the right loader
    try:
        loader = ReceptorFormatRegistry.loaders[extension]
    except KeyError:
        raise IOError(
            "Sorry, no loader for filename=%s (extension=%s) "
            "was found. I can only load files "
            "with extensions in %s"
            % (filename[0], extension, ReceptorFormatRegistry.loaders.keys())
        )

    # 5 - load the receptor
    protein = loader.load(filename)
    return protein
