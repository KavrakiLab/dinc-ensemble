
from typing import Dict

from .preprocess_molecule_strategies import PreprocessMoleculeStrategy

class _PreprocessMoleculeRegistry(object):

    """
    Registry for ligand preprocessing strategies.

    This class serves as a registry for different ligand preprocessing strategies
    which are instances of PreprocessMoleculeStrategy.

    Attributes
    ----------
    strategies : Dict[str, PreprocessMoleculeStrategy]
        Dictionary mapping strategy names to instances of PreprocessMoleculeStrategy.

    Methods
    -------
    register_strategy(strategy: PreprocessMoleculeStrategy)
        Registers a new preprocessing strategy.

    
    Examples
    --------
    First, define a new PreprocessMoleculeStrategy if you wish.
    >>> class ResetCharges(PreprocessMoleculeStrategy):
    >>>     strategy_name = "remove_charges"
    >>>     strategy_description: str = "Reset all the charges of molecule atoms to zero"
    >>>     @classmethod
    >>>     def process_specific(cls, ligand: DINCMolecule) -> DINCMolecule:
    >>>         for atom in ligand.allAtoms:
    >>>             atom.charge = 0
    >>>         return ligand
    Now, register the strategy (preferrably in __init__.py of this subpackage).
    >>> PreprocessMoleculeRegistry.register(ResetCharges())
    You can also access the remove_charges function directly 
    (__init__.py transforms the registered strategies into functions 
    and exposes them globally):
    >>> remove_charges(ligand) # this will also work!
    """

    strategies: Dict[str, PreprocessMoleculeStrategy] = {}

    @classmethod
    def register_strategy(cls, strategy: PreprocessMoleculeStrategy):
        """
        Registers a new preprocessing strategy.

        Parameters
        ----------
        strategy : PreprocessMoleculeStrategy
            The preprocessing strategy to register.
        """

        cls.strategies[strategy.strategy_name] = strategy

PreprocessMoleculeRegistry = _PreprocessMoleculeRegistry()
