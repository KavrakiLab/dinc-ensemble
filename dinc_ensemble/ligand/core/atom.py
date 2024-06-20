from MolKit.molecule import Atom as MolKitAtom, BondSet as MolKitBondSet
from dataclasses import dataclass, astuple
import numpy as np
import numpy.typing as npt
from typing import ClassVar, Callable, List

from PyBabel.babelElements import babel_elements
import pandas as pd

#from dinc_ensemble.ligand.core.bond import Bond

#------------------------- ATOM ELEMENENT TYPE specifications ---------------------------------------#
# Things like specific element radius ets.


# Atom Specifications from babel_elements - related to element types
@dataclass 
class ElementSpecs:
     element_name: str #TODO: QUESTION: could we type these as numpy types for better peformence maybee??
     bond_ord_rad: float
     cov_rad: float 
     bs_rad: float 
     vdw_rad: float 
     max_bonds: int 
     num: int 
     rgb: tuple

# extend babel elements with element name
[e_specs.update({"element_name": e_name}) for e_name, e_specs in babel_elements.items()]

# Babel element specs will be a constant in the code
# TODO: QUESTION: is there a way to hint more specific types here than just DataFrame types, maybe py pandera or pydantic?
BABEL_ELEMENTS: pd.DataFrame = pd.DataFrame( [
     ElementSpecs(**be) for be in babel_elements.values()
] )  


#------------------------- ATOM charge related properties ---------------------------------------#
# In MolKitAtom, there can be a list of charges for each atom, coming from a different source (e.g., mol2 or others)
# It comes in a dictionary e.g., {'mol2': 0.374, 'pdb': 0.739, ...}
@dataclass
class AtomCharge:
     selected_charge: str
     charges: dict[str, float]

#------------------------- ATOM conformation related properties ---------------------------------------#

@dataclass
class AtomConformations:
     selected_conformation: int
     confs: npt.ArrayLike 
     molkit_unique_name: str

     @classmethod
     def to_df(cls, confs: List) -> pd.DataFrame:
          confs_dict = [conf.__dict__ for conf in confs]
          return pd.DataFrame(confs_dict)


#------------------------- ATOM ELEMENENT TYPE specifications ---------------------------------------#

# Atom Specifications - specific to particular elements in read molecules
@dataclass
class Atom:
     
     _molkit_atom: MolKitAtom
     _molkit_bonds: MolKitBondSet
     
     # as the atom can have multiple conformations, here are the currently selected
     # underscored fields will be ignored in the conversion to dataframe (see molecule.py)
     _conformations: AtomConformations 
     
     molkit_unique_name: str # we ensure the name is unique with prepare_ligand4!

     rdkit_name: str
     rdkit_idx: int

     number: int
     # TODO: not sure about the difference between element and chemElement is in MolKitAtom, 
     # chemElement seems to be more used for indexing babel_elems so, I will only keep the chemElem for now
     # it will be pointing at the specific element specs
     element: str
     organic: bool

     # charge stuff
     charge: AtomCharge

     # underscored fields will be ignored in the conversion to dataframe (see molecule.py)
     _is_column: ClassVar[Callable] = lambda x: x[0] != "_"
     
     # allow to initialize an atom with MolKitMolecule Atom
     def __init__(self, molkit_atom: MolKitAtom):
          
          self._molkit_atom = molkit_atom
          
          self.molkit_unique_name = molkit_atom.name
          if hasattr(molkit_atom, "number"):
               self.number = molkit_atom.number
          else:
               self.number = -1

          if hasattr(molkit_atom, "rdkit_idx"):
               self.rdkit_idx = molkit_atom.rdkit_idx
          else:
               self.rdkit_idx = None
          self.rdkit_name = None

          self.element = molkit_atom.chemElem
          self.organic = molkit_atom.organic
          if molkit_atom.chargeSet is None or not molkit_atom._charges:
               self.charge = AtomCharge("none", {"none": 0})
          else:
               self.charge = AtomCharge(molkit_atom.chargeSet, molkit_atom._charges)
          conformations = np.array( molkit_atom._coords)
          self._molkit_bonds = molkit_atom.bonds
          self._conformations = AtomConformations(molkit_atom.conformation, conformations, self.molkit_unique_name)

     @classmethod
     def to_df(cls, atoms: List) -> pd.DataFrame:

          atoms_dict = [atom.__dict__ for atom in atoms]
          #print(atoms_dict)
          atoms_df = pd.DataFrame(atoms_dict)
          atoms_df['charge'] = atoms_df['charge'].apply(lambda x: x.charges[x.selected_charge])
          # make sure that the atom names are unique
          atoms_df['idx'] = atoms_df.apply(lambda row: row["_molkit_atom"]._uniqIndex, axis=1)

          df_columns = [key for key in atoms_df.columns if cls._is_column(key)]
          atoms_df = atoms_df[df_columns]
          if not atoms_df['molkit_unique_name'].is_unique:
               raise ValueError("Molkit Molecule has multiple atoms with same ID.")
          atoms_df.set_index('molkit_unique_name', inplace=True)
          return atoms_df
     
     atomize : ClassVar = np.vectorize(lambda x: Atom(x))