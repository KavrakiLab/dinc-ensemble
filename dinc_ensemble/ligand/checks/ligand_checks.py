from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List
from MolKit.molecule import Atom
from dinc_ensemble.ligand.core import DINCMolecule
from math import isnan
from numpy.linalg import norm
from numpy import array
from enum import Enum
from ligand_checks_types_and_conditions import LigandCheckType
from typing import Optional 
import numpy as np

import logging
logger = logging.getLogger('dinc_ensemble.ligand')
    
class AbstractMoleculeCheck(ABC):

    @abstractmethod
    def check(self, molecule: DINCMolecule):
        pass

class BaseCheck(AbstractMoleculeCheck):

    def __init__(self, type:LigandCheckType= LigandCheckType.ESSENTIAL, 
                 next_check: Optional[AbstractMoleculeCheck] = None) -> None:
        super().__init__()
        self.__type = type

    @property
    def type(self) -> LigandCheckType:
        return self.__type
    
    @property
    def next_check(self) -> Optional[AbstractMoleculeCheck]:
        return self.__next_check
    
    @next_check.setter
    def next_check(self, checker: AbstractMoleculeCheck):
        self.__next_check = checker

    @abstractmethod
    def check(self, molecule: DINCMolecule):
        if self.__next_check:
            return self.__next_check.check(molecule)
        return None

class BaseConditionCheck(BaseCheck):
    
    def __init__(self, type: LigandCheckType = LigandCheckType.ESSENTIAL, next_check: None = None) -> None:
        super().__init__(type, next_check)
        
    
    def specific_check(self,  molecule: DINCMolecule):
        pass
    
    def check(self,  molecule: DINCMolecule):
        if self.type.check_condition():
            self.specific_check(molecule)
        return super().check(molecule)

class CheckHeavyAtoms(BaseConditionCheck):

    def __init__(self, type: LigandCheckType = LigandCheckType.ESSENTIAL, 
                 next_check: None = None,
                 max_heavy_atoms: int = 1000) -> None:
        self.__max_heavy_atoms = max_heavy_atoms
        super().__init__(type, next_check)
        
    @property
    def max_heavy_atoms(self):
        if self.__max_heavy_atoms <= 0:
            logger.info("Warning: maximum heavy atoms allowed set to <= 0 !")
        return self.__max_heavy_atoms
        
    def specific_check(self, molecule: DINCMolecule):
        # check the ligand's size
        if molecule.n_heavy_atoms > self.__max_heavy_atoms:
            logger.info("DincError: Unfortunately, DINC cannot process such a large ligand.")
            raise IOError("DINC can't process ligands larger than 1000 heavy atoms. ")


class CheckAD4AtomTypes(BaseConditionCheck):
    
    AD4_ACCEPTED_TYPES = ["Br", "C", "Cl", "F", "H", "N", "O", "P", "S"]
    
    def __init__(self, type: LigandCheckType = LigandCheckType.AD4, next_check: None = None) -> None:
        super().__init__(type, next_check)

    def specific_check(self, molecule: DINCMolecule):
        
        def check_atom_type(a: Atom):
            if a.element not in self.AD4_ACCEPTED_TYPES:
                raise ValueError("DincError: DINC cannot process this ligand using AutoDock 4.\
                                DincError: Unknown atom {}, {}".format(a.atom, a.element))
            if "H" in a.name:
                raise ValueError("DincError: DINC cannot process this ligand using AutoDock 4.\
                                DincError: Unknown atom {}, {}. \
                                Try again after removing hydrogen atoms".format(a.atom, a.element))
        
        vect_check = np.vectorize(check_atom_type)
        atoms_checked = vect_check(molecule.atoms)           


class CheckCoordsNan(BaseConditionCheck):
    
    def __init__(self, type: LigandCheckType = LigandCheckType.ESSENTIAL, next_check: None = None) -> None:
        super().__init__(type, next_check)
    
    def specific_check(self, molecule: DINCMolecule):
        if np.isnan(molecule.coords).any():
            raise ValueError("DincError: Some ligand atoms have invalid coordinates!")
        if np.all(molecule.coords == 0.0):
            ValueError(
                "DincError: All the coordinates of this ligand's atoms are equal to zero!"
            )

class CheckNonbondedAtoms(BaseConditionCheck):
    
    def __init__(self, type: LigandCheckType = LigandCheckType.ESSENTIAL, next_check: None = None) -> None:
        super().__init__(type, next_check)
    
    def specific_check(self, molecule: DINCMolecule):

        # check that the ligand does not contain any non-bonded heavy atom
        
        heavy_atom_bond_mat = molecule.bonds_mat_idx[molecule.heavy_atom_idx,
                                                     molecule.heavy_atom_idx]
        # does any bond matrix row sum up to zero?
        if not heavy_atom_bond_mat.sum(axis=1).all():
            ValueError(
                    "DincError: This ligand contains non-bonded atoms. Please check your file."
                )

class CheckSterickClashes(BaseConditionCheck):

    def __init__(self, type: LigandCheckType = LigandCheckType.ESSENTIAL, next_check: None = None) -> None:
        super().__init__(type, next_check)
    
    def specific_check(self, molecule: DINCMolecule):
        heavy_atoms = molecule.heavy_atoms
        heavy_atom_bonds = molecule.bonds_mat[molecule.heavy_atom_idx,
                                              molecule.heavy_atom_idx]
        heavy_atom_coords = molecule.coords[molecule.heavy_atom_idx]
        heavy_atom_dist = np.linalg.norm(heavy_atom_coords[:, None, :] -
                                        heavy_atom_coords[None, :, :], 
                                         axis=-1)
        def bond_thr(b):
            return 1.2 * (b.atom1.bondOrderRadius + b.atom2.bondOrderRadius)
        bond_clash_thr = np.vectorize(bond_thr)(heavy_atom_bonds)

        if np.any(heavy_atom_dist<bond_clash_thr):
            ValueError(
                            "DincError: This ligand contains steric clashes. Please check your file."
                        )

class CheckFlatLigand(BaseConditionCheck):

    def __init__(self, type: LigandCheckType = LigandCheckType.ESSENTIAL, next_check: None = None) -> None:
        super().__init__(type, next_check)
    
    def specific_check(self, molecule: DINCMolecule):
        heavy_atoms = molecule.heavy_atoms
        # check if there are colinear dimensions - check by linear dependence
        # matrix rank == 3 if they are not in the same plane
        heavy_atom_coords = molecule.coords[molecule.heavy_atom_idx]
        if np.linalg.matrix_rank(heavy_atom_coords) < 3:
            ValueError(
                "DincError: This ligand has a flat and not a proper 3D structure."
            )
    

def check_molecule(molecule:DINCMolecule):

    checkers: list[type[BaseConditionCheck]] = BaseConditionCheck.__subclasses__()

    first_check = checkers[0]()
    for checker in checkers[1:]:
        checker_instance = checker()
        first_check.next_check = checker_instance
    first_check.check(molecule)