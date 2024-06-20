from typing import List
from .preprocess_molecule_registry import PreprocessMoleculeRegistry
from .preprocess_molecule_strategies import PreprocessMoleculeStrategy
from dinc_ensemble.ligand.core import DINCMolecule

def preprocess_molecule(ligand: DINCMolecule, **kwargs) -> DINCMolecule:
    """
    Apply preprocessing strategies to a molecule.

    This function takes a molecule and a set of preprocessing strategies as input. 
    It applies each strategy to the molecule and returns the processed molecule.

    Parameters
    ----------
    ligand : DINCMolecule
        The molecule to be preprocessed.
    **kwargs : dict
        A dictionary of preprocessing strategies to be applied to the molecule. 
        The keys of the dictionary are the names of the strategies, and the values 
        are either 0 or 1 indicating whether or not to apply the strategy.

    Returns
    -------
    DINCMolecule
        The preprocessed molecule.

    Raises
    ------
    KeyError
        If a strategy specified in the dictionary is not implemented.

    Examples
    --------
    First import needed modules:
    >>> from dinc_ensemble.ligand.process import preprocess_molecule, 
    >>> from dinc_ensemble import PREPROCESS_MOLECULE_OPTIONS, load_molecule
    Now load a molecule:
    >>> ligand = load_molecule("ligand.pdb")
    Next, checkout the preprocessing options:
    >>> print(PREPROCESS_MOLECULE_OPTIONS)
    dict_keys(['merge_lonepairs', 'add_hydrogens', 'add_gasteiger_charges', 
    'add_autodock_types', 'merge_nonpolar_hydrogens'])
    >>> preprocess_molecule(molecule, add_hydrogens=1, merge_lonepairs=1)
    Processed ligand
    """
    selected_strategies: List[PreprocessMoleculeStrategy] = []
    for key, value in kwargs.items():
        if key not in PreprocessMoleculeRegistry.strategies:
            raise KeyError("Selected molecule preprocessing strategy {} \
                           is not implemented".format(key))
        if value == 1:
            selected_strategies.append(PreprocessMoleculeRegistry.strategies[key])
    
    for strategy in selected_strategies:
        ligand = strategy.process(ligand)
    return ligand