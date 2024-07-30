
from abc import ABC, abstractmethod
from typing import Callable
from functools import wraps

from dinc_ensemble.ligand.core import DINCMolecule

import logging
logger = logging.getLogger('dinc_ensemble.ligand')

class PreprocessMoleculeStrategy(ABC):
    """
    Abstract base class to define rules for ligand/molecule preprocessing.

    This class defines a few rules on what the ligand/molecule preprocessing
    must look like and leaves specific preprocessing steps to the classes that inherit it.

    Attributes
    ----------
    strategy_name : str
        The name of the preprocessing strategy.
    strategy_description : str
        The description of the preprocessing strategy.

    Methods
    -------
    process_specific(cls, ligand: DINCMolecule) -> DINCMolecule
        Abstract method to be implemented by subclasses to process the ligand using a specific strategy.
    process(cls, ligand:DINCMolecule) -> DINCMolecule
        Process the ligand using the defined strategy.
    to_function(cls) -> Callable
        Returns a function that processes the ligand using the defined strategy.
    """

    strategy_name: str = ""
    strategy_description: str = ""

    @classmethod
    @abstractmethod
    def process_specific(cls, ligand: DINCMolecule) -> DINCMolecule:
        """
        Abstract method to be implemented by subclasses to process the ligand using a specific strategy.

        Parameters
        ----------
        ligand : DINCMolecule
            The ligand to be processed.

        Returns
        -------
        DINCMolecule
            The processed ligand.
        """
        ...

    @classmethod
    def process(cls, ligand:DINCMolecule) -> DINCMolecule:
        """
        Process the ligand using the defined strategy.

        Parameters
        ----------
        ligand : DINCMolecule
            The ligand to be processed.

        Returns
        -------
        DINCMolecule
            The processed ligand.
        """
        logger.info("Processing ligand with {} strategy".format(cls.strategy_name))
        # this way we will make sure that all the changes are reflected in the ligand properties
        processed_ligand = cls.process_specific(ligand) 
        processed_ligand.__reset__(processed_ligand.molkit_molecule, prepare=False)
        return processed_ligand
    
    @classmethod
    def to_function(cls) -> Callable:
        """
        Returns a function that processes the ligand using the defined strategy.
        Converts the given strategy to a callable function that can be triggered 
        directly on the ligand. 

        Returns
        -------
        Callable
            A function that takes a ligand as input and returns the processed ligand.
        """
        class_strategy_f = cls.process
        @wraps(class_strategy_f)
        def function(ligand: DINCMolecule) -> DINCMolecule:
            return class_strategy_f(ligand)
        function.__name__ = cls.strategy_name
        function.__doc__ = cls.strategy_description
        return function

