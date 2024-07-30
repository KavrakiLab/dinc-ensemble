
from MolKit.molecule import Molecule as MolKitMolecule
from AutoDockTools.MoleculePreparation import AD4LigandWriter
from rdkit.Chem.rdchem import Mol as RDKitMol
from rdkit.Chem import AllChem
from rdkit import Chem
from openbabel import pybel

from typing import Optional, List, ClassVar
import pandas as pd
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import os

from .atom import Atom
from .bond import Bond
from .basic_autodock_prepare_ligand import prepare_ligand4
from .convert_molecule import molkit_to_rdkit

@dataclass
class DINCMolecule():
    """
    A class representing a molecule in the DINC format.

    Attributes
    ----------
    molkit_molecule : MolKitMolecule
        The original MolKitMolecule object.
    atoms : pd.DataFrame #[Atom] TODO: how to specify the type of the dataframe, maybe pandera? https://github.com/unionai-oss/pandera
        A DataFRame of all atoms in the molecule.
    bonds : pd.DataFrame #[Bond] TODO: how to specify the type of the dataframe, maybe pandera? https://github.com/unionai-oss/pandera
        A DataFRame of all bonds in the molecule.
    n_atoms : int
        The number of atoms in the molecule.
    coords : npt.NDArray[np.float32]
        An array of coordinates of all atoms in the molecule.
    """ 
    molkit_molecule: MolKitMolecule
    # bookkeeping related to Meeko loading
    _rdkit_molecule: Optional[RDKitMol] = None
    
    # all these are properties extracted from molkit_molecule, reshaped for smoother use
    _atoms: Optional[np.ndarray] = None 
    _bonds: Optional[np.ndarray] = None
    _atoms_df: Optional[pd.DataFrame] = None # TODO: QUESTION: is there a way to hint more specific types here than just DataFrame types, maybe py pandera or pydantic?
    _bonds_df: Optional[pd.DataFrame] = None # TODO: QUESTION: is there a way to hint more specific types here than just DataFrame types, maybe py pandera or pydantic?
    _conformations_df: Optional[pd.DataFrame] = None
    _heavy_atoms_df: Optional[pd.DataFrame] = None
    _mol_name: Optional[str] = None
    # these are class variables - available for the whole DINC program to access, updated by instances registering new molecules
    # TODO: is this optimal - what if molecules are registered in parallel - could this be an issue?
    MOL_CNT: ClassVar[int] = 0
    UNIQUE_MOLECULE_NAMES: ClassVar[dict] = {}
    
    def __register_molkit_molecule__(self, update_name = True, update_rdkit = True):
        if self.molkit_molecule is not None:
            # give a unique name for each registered molecule in a single run
            if update_name:
                unique_name = "mol_{}".format(self.MOL_CNT)
                while unique_name in self.UNIQUE_MOLECULE_NAMES:
                    self.MOL_CNT += 1
                    unique_name = "mol_{}".format(self.MOL_CNT)
                self.UNIQUE_MOLECULE_NAMES[unique_name] = 1
                self._mol_name = unique_name
                
            # get atoms info in df and numpy formats
            # keep the rdkit molecule updated
            if update_rdkit and hasattr(self.molkit_molecule, "pdbqt_str"):
                self._rdkit_molecule, self._atom_map, self._bond_map = molkit_to_rdkit(self.molkit_molecule)

            allAtoms_array = np.array(self.molkit_molecule.allAtoms)

            self._atoms = Atom.atomize(allAtoms_array)
            self._atoms_df = Atom.to_df(self._atoms)
            
            self._heavy_atoms_df = self._atoms_df[~self._atoms_df.element.str.contains("H")]
            
            # get bonds info in df and numpy formats
            Bond.__init_bond_properties__(self.molkit_molecule.allAtoms.bonds[0])
            if len(self.molkit_molecule.allAtoms.bonds[0]) > 0:
                self._bonds = Bond.bondize(np.array(self.molkit_molecule.allAtoms.bonds[0]))
                self._bonds_df = Bond.to_df(self._bonds)
                # connect the bonds with atoms ids
                full_bond_list = pd.merge(left=self._bonds_df, right=self._atoms_df.add_suffix("_atom1"), 
                        left_on="atom1_molkit_unique_name",
                        right_on = "molkit_unique_name", how="left")

                full_bond_list = pd.merge(left=full_bond_list, right=self._atoms_df.add_suffix("_atom2"), 
                        left_on="atom2_molkit_unique_name",
                        right_on = "molkit_unique_name", how="left")
                columns_to_keep = list(self._bonds_df.columns) + ["idx_atom1", "idx_atom2"]
                self._bonds_df = full_bond_list[columns_to_keep]
            else:
                self._bonds = []
                self._bonds_df = None



    def __init__(self, 
                 molkit_molecule: MolKitMolecule,
                 prepare: bool = True,
                 prep4_repairs: str = "bonds_hydrogens"):
        self.molkit_molecule = molkit_molecule
        if prepare:
            prepare_ligand4(self.molkit_molecule, 
                            repairs=prep4_repairs)
        elif hasattr(self.molkit_molecule, "torTree") and hasattr(self.molkit_molecule, "torscount"):
            AD4LigandWriter().write(self.molkit_molecule, "", inmem=True)
            AD4LigandWriter().write(self.molkit_molecule, "", inmem=True)
            
        self.__register_molkit_molecule__()     
    
    def __reset__(self, 
                  molkit_molecule: MolKitMolecule,
                  prepare: bool = False,
                  prep4_repairs: str = "bonds_hydrogens",
                  prep4_bonds_to_inactivate: Optional[str]="",
                  reset_rdkit: bool = False):
        '''
        bonds_to_inactivate             string of bonds to inactivate composed of 
                                    'ero-based atom indices eg 5_13_2_10  
                                    wi'nactivate atoms[5]-atoms[13] bond 
                                    a'toms[2]-atoms[10] bond 
                                    (defau's not to inactivate any specific bonds)
        '''
        self.molkit_molecule = molkit_molecule
        if prepare:
            prepare_ligand4(self.molkit_molecule,
                            repairs=prep4_repairs,
                            bonds_to_inactivate=prep4_bonds_to_inactivate, 
                            build_bonds_by_dist=False)
        elif hasattr(self.molkit_molecule, "torTree") and hasattr(self.molkit_molecule, "torscount"):
            AD4LigandWriter().write(self.molkit_molecule, "", inmem=True)
            AD4LigandWriter().write(self.molkit_molecule, "", inmem=True)

        self.__register_molkit_molecule__(update_name=False)
        
    
    @property
    def atoms(self) -> pd.DataFrame:
        if self._atoms_df is not None:
            return self._atoms_df
        else:
            raise AttributeError

    @property
    def bonds(self) -> pd.DataFrame:
        if self._bonds_df is not None:
            return self._bonds_df
        else:
            raise AttributeError
    @property
    def heavy_atoms(self) -> pd.DataFrame:
        if self._heavy_atoms_df is not None:
            return self._heavy_atoms_df
        else:
            return AttributeError
         
    @property
    def n_atoms(self) -> int:
        return self.atoms.shape[0]
        
    @property
    def n_heavy_atoms(self) -> int:
        return self.heavy_atoms.shape[0]

    @property
    def coords(self) -> pd.DataFrame:
        if self._conformations_df is not None:
            return self._conformations_df
        else:
            return AttributeError
         
    
        