"""
Module for all ligand loading related stuff.
"""
from typing import Dict
from os import path

from .ligand_abstract_loader import LigandLoader
from .ligand_abstract_writer import LigandWriter

from dinc_ensemble.ligand.core import DINCMolecule


class _LigandIOFormatRegistry(object):
    """
    A registry for LigandLoader classes.

    This class serves as a registry for LigandLoader and LigandWriter classes. 
    Users can register specific ligand loaders and writers that they define.
    The keys are the extensions and the LigandLoader/LigandWriter is the value. 
    This class is inspired by the format registry template used in the mdtraj library.

    Attributes
    ----------
    loaders : Dict[str, LigandLoader]
        A dictionary where the keys are the extensions and the values are the LigandLoader classes.

    writers : Dict[str, LigandWriter]
        A dictionary where the keys are the extensions and the values are the LigandWriter classes.

    Methods
    -------
    register_loader(extension: str, loader: LigandLoader)
        Registers a LigandLoader class for a specific extension.

    register_writer(extension: str, writer: LigandWriter)
        Registers a LigandWriter class for a specific extension.
    
    Examples
    --------
    Let's first define a `LigandLoader` subclass:
    >>> class PDBLigandLoader(LigandLoader):
    >>>  extension = ".pdb"
     
    >>>  @classmethod
    >>>  def specific_load(cls, filename: str) -> DINCMolecule:
    >>>      # Load a PDB file
    >>>      pass
    
    Now, we can register these loaders with the LigandFormatRegistry:
    >>> LigandFormatRegistry.register_loader(".pdb", PDBLigandLoader)
    
    In this example, `load_ligand` will use the appropriate loader for 
    the file extension of the provided filename. If no loader is registered for that extension, 
    it raises an error.

    This approach allows you to easily extend the functionality of your program to support 
    new file formats: simply define a new `LigandLoader` subclass and register it with the 
    `LigandFormatRegistry`.

    In the same way you can define a ligand writer and
    extend the formats that the molecule can be written to.
    """

    loaders: Dict[str, LigandLoader] = {}
    writers: Dict[str, LigandWriter] = {}

    @classmethod
    def register_loader(cls, extension: str, loader: LigandLoader):
        """
        Registers a LigandLoader class for a specific extension.

        Parameters
        ----------
        extension : str
            The extension for which the LigandLoader class should be registered.
        loader : LigandLoader
            The LigandLoader class to be registered for the specified extension.
        """
        cls.loaders[extension] = loader


    @classmethod
    def register_writer(cls, extension: str, writer: LigandWriter):
        """
        Registers a LigandWriter class for a specific extension.

        Parameters
        ----------
        extension : str
            The extension for which the LigandWriter class should be registered.
        loader : LigandWriter
            The LigandWriter class to be registered for the specified extension.
        """
        cls.writers[extension] = writer


# Make a single instance of this class, and then
# get rid of the class object. This should be
# treated as a singleton
LigandIOFormatRegistry = _LigandIOFormatRegistry()


def load_ligand(filename: str, mk_config:dict = None) -> DINCMolecule:
    """
    Load a ligand from a file.

    This function loads a ligand from a file specified by the filename parameter.

    Parameters
    ----------
    filename : str
        The path to the file containing the ligand.

    Returns
    -------
    DINCMolecule
        The loaded ligand as a DINCMolecule object.

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
    # Load a ligand from a PDB file
    >>> ligand = load_ligand("ligand.pdb")

    # Load a ligand from a MOL2 file
    >>> ligand = load_ligand("ligand.mol2")

    Notes
    -----
    This function relies on the LigandFormatRegistry class to 
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
        loader = LigandIOFormatRegistry.loaders[extension]
    except KeyError:
        raise IOError(
            "Sorry, no loader for filename=%s (extension=%s) "
            "was found. I can only load files "
            "with extensions in %s"
            % (filename[0], extension, LigandIOFormatRegistry.loaders.keys())
        )

    # 5 - load the ligand
    molecule = loader.load(filename, mk_config)
    return molecule


def write_ligand(molecule: DINCMolecule, filename: str) -> None:
    """
    Write a ligand to a file.

    This function writes a ligand to a file specified by the filename parameter.

    Parameters
    ----------
    molecule : DINCMolecule
        The the molecule to write.

    filename : str
        The path to the file to write.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If filename parameter is an empty list.
    IOError
        If no writer is found for the specified file extension.
    TypeError
        If the extension of the file does not match the expected extension.
    FileNotFoundError | PermissionError | OSError
        If the file can not be open for writing.

    Examples
    --------
    # Load a ligand from a PDB file and write to a PDB file
    >>> ligand = load_ligand("ligand.pdb")
    >>> write_ligand(ligand, "opened_ligand.pdb")

    # Load a ligand from a MOL2 file and write to a PDB file
    >>> ligand = load_ligand("ligand.mol2")
    >>> write_ligand(ligand, "opened_ligand.pdb")

    Notes
    -----
    This function relies on the LigandFormatRegistry class to 
    retrieve the appropriate writers for different file extensions.

    """

    # 1 - get the extension of the file
    (base, extension) = path.splitext(filename)

    # 2 - make the needed checks
    if len(set(extension)) == 0:
        raise ValueError(
            "Filename was an empty list"
        )

    # 3 - get the right loader
    try:
        writer = LigandIOFormatRegistry.writers[extension]
    except KeyError:
        raise IOError(
            "Sorry, no writer for filename=%s (extension=%s) "
            "was found. I can only write files "
            "with extensions in %s"
            % (filename[0], extension, LigandIOFormatRegistry.writers.keys())
        )

    # 4 - write the ligand
    writer.write(molecule, filename)
