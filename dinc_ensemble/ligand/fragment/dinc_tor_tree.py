
from dinc_ensemble.ligand.core import DINCMolecule
from MolKit.torTree import TorTree
from typing import Optional
from AutoDockTools.AutoDockBondClassifier import AutoDockBondClassifier
from MolKit.molecule import AtomSet

from dataclasses import dataclass
from typing import Optional
import pandas as pd
from __future__ import annotations

@dataclass
class DINCTorNode:

    node_idx: int # unique note index
    atoms_df: pd.DataFrame # atoms that are part of this rigid chunk
    bonds_df: pd.DataFrame # bonds that are part of this rigid chunk (both atoms in the rigis chunk)
    flex_bonds_df: pd.DataFrame # bonds that are flexible (one rigid atom, other a part of a different node)
    children: Optional[DINCTorNode] # children of this node




class DINCTorTree:
    '''
    Class for the full molecule torsion tree
    The fragments will work off this initial torsion tree
    Depending on the choice of the root and active torsions
    The torsion tree is just a template - all the bonds are kept rigid
    '''
    def __init__(self, molecule:DINCMolecule) -> None:
        
        self._molecule = molecule
        self.TORSDOF = molecule.bonds[molecule.bonds.possibleTors_].shape[0]  # number of possible torsions
        self.torscount = 0  # number of active torsions
        self.ROOT = 0 # root atom

        