from MolKit.molecule import BondSet, Bond as MolKitBond
from AutoDockTools.AutoDockBondClassifier import AutoDockBondClassifier

from dataclasses import dataclass
from .atom import Atom
import numpy as np
from typing import Optional, ClassVar, Callable, List
import pandas as pd

@dataclass
class Bond:

    _molkit_bond: MolKitBond
    
    _atom1: Atom
    _atom2: Atom
    molkit_full_name: str
    atom1_molkit_unique_name: str
    atom2_molkit_unique_name: str
    origin: str
    bond_order: int
    rdkit_idx: int


    # These properties are set by AutodockBondClassifier - this is slow per-bond but fast for the bond set
    # That's why these properties will be initialized at dataframe construction or such
    bond_properties_initialized = False
    
    # underscored fields will be ignored in the conversion to dataframe (see molecule.py)
    _is_column: ClassVar[Callable] = lambda x: x[0] != "_"

    def __init__(self, molkit_bond: MolKitBond):

        self._molkit_bond = molkit_bond
        
        self._atom1 = Atom(molkit_bond.atom1)
        self._atom2 = Atom(molkit_bond.atom2)
        self.molkit_full_name = molkit_bond.full_name()
        self.atom1_molkit_unique_name = molkit_bond.atom1.name
        self.atom2_molkit_unique_name = molkit_bond.atom2.name
        self.origin = molkit_bond.origin
        self.bond_order = molkit_bond.bondOrder

        if hasattr(self._molkit_bond, "rdkit_idx"):
            self.rdkit_idx = molkit_bond.rdkit_idx
        else:
            self.rdkit_idx = None

        if hasattr(self._molkit_bond, "initialized") and self._molkit_bond.initialized:
            self.__init_molkit_properties__()
        
    
    def __init_molkit_properties__(self):
        self.incycle_ = self._molkit_bond.incycle
        self.hrotator_ = self._molkit_bond.hrotator
        self.amdbond_ = self._molkit_bond.amdbond
        self.possibleTors_ = self._molkit_bond.possibleTors
        self.activeTors_ = self._molkit_bond.activeTors

    @property
    def incycle(self) -> bool:
        if self.bond_properties_initialized is False:
            print("Warning: accessing a bond property that was not initialized")
        if hasattr(self._molkit_bond, "incycle"):
            self.incycle_ = self._molkit_bond.incycle
            return self.incycle_
        raise AttributeError("Attribute incycle not initialized!")
    
    @property
    def hrotator(self) -> bool:
        if self.bond_properties_initialized is False:
            print("Warning: accessing a bond property that was not initialized")
        if hasattr(self._molkit_bond, "hrotator"):
            self.hrotator = self._molkit_bond.hrotator
            return self.hrotator_
        raise AttributeError("Attribute hrotator not initialized!")
    
    @property
    def amdbond(self) -> bool:
        if self.bond_properties_initialized is False:
            print("Warning: accessing a bond property that was not initialized")
        if hasattr(self._molkit_bond, "amdbond"):
            self.amdbond_ = self._molkit_bond.amdbond
            return self.amdbond_
        raise AttributeError("Attribute amdbond not initialized!")
    
    @property
    def possibleTors(self) -> bool:
        if self.bond_properties_initialized is False:
            print("Warning: accessing a bond property that was not initialized")
        if hasattr(self._molkit_bond, "possibleTors"):
            self.possibleTors_ = self._molkit_bond.possibleTors
            return self.possibleTors_
        raise AttributeError("Attribute possibleTors not initialized!")
    
    @property
    def activeTors(self) -> bool:
        if self.bond_properties_initialized is False:
            print("Warning: accessing a bond property that was not initialized")
        if hasattr(self._molkit_bond, "activeTors"):
            self.activeTors_ = self._molkit_bond.activeTors
            return self.activeTors_
        raise AttributeError("Attribute activeTors not initialized!")

    @classmethod
    def __init_bond_properties__(cls, bonds: BondSet):
        
        classes = AutoDockBondClassifier(detectAll=False).classify(bonds)
        # set the values of the fragment's bond attributes, based on the classification

        bonds.amdbond = 0
        bonds.hrotator = 0
        classes["amide"].amdbond = 1
        for a in classes["hydrogenRotators"]:  # this class contains atoms, not bonds
            a.bonds.get(lambda b: b.neighborAtom(a).element != "H").hrotator = 1
        bonds.initialized = 1
        

    @classmethod
    def to_df(cls, bonds: BondSet) -> pd.DataFrame:
        bonds_dict = [bond.__dict__ for bond in bonds]
        bonds_df = pd.DataFrame(bonds_dict)
        bonds_df['molkit_full_name'] = bonds_df.apply(lambda row: row["molkit_full_name"]+" | {}".format(row.name), axis=1)
        bonds_df['atom1_molkit_unique_name'] = bonds_df["_atom1"].apply(lambda x: x.molkit_unique_name)
        bonds_df['atom2_molkit_unique_name'] = bonds_df["_atom2"].apply(lambda x: x.molkit_unique_name)
        bond_cols = [key for key in bonds_df.columns if cls._is_column(key)]
        bonds_df = bonds_df[bond_cols]
        
        if not bonds_df['molkit_full_name'].is_unique:
            raise ValueError("Molkit Molecule has multiple bonds with same ID.")
        bonds_df.set_index('molkit_full_name', inplace=True)
        return bonds_df

    bondize: ClassVar  = np.vectorize(lambda x: Bond(x))
