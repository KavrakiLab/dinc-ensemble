from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from MolKit.molecule import Molecule as MolKitMolecule
    from .molecule import DINCMolecule
    from rdkit.Chem.rdchem import Mol as RDKitMol

from rdkit.Chem import Draw 
from rdkit.Chem import AllChem
from .convert_molecule import to_rdkit
from IPython.display import SVG
import pandas as pd
from math import isnan

def draw_atom_property_rdkit(molecule: DINCMolecule, property: pd.Index) -> SVG:
    rdkit_mol = to_rdkit(molecule)
    AllChem.Compute2DCoords(rdkit_mol)
    selected_atoms = molecule.atoms[property]

    rdkit_atoms = [int(x) for x in 
                          selected_atoms["rdkit_idx"]
                          if not isnan(x)]

    d = Draw.rdMolDraw2D.MolDraw2DSVG(200, 200) # or MolDraw2DCairo to get PNGs
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, rdkit_mol, highlightAtoms=rdkit_atoms)
    d.FinishDrawing()
    return SVG(d.GetDrawingText())

def draw_bond_property_rdkit(molecule: DINCMolecule, 
                             property: pd.Index, 
                             display_atom_labels: bool = False) -> SVG:

    rdkit_mol = to_rdkit(molecule)
    AllChem.Compute2DCoords(rdkit_mol)

    if display_atom_labels:
        # Iterate over the atoms
        for i, atom in enumerate(rdkit_mol.GetAtoms()):
            # For each atom, set the property "molAtomMapNumber" to a custom number, let's say, the index of the atom in the molecule
            atom.SetProp("molAtomMapNumber", str(atom.GetPDBResidueInfo().GetName()))

    selected_bonds = molecule.bonds[property]

    rdkit_selected_ids = [int(x) for x in 
                          selected_bonds["rdkit_idx"]
                          if not isnan(x)]


    d = Draw.rdMolDraw2D.MolDraw2DSVG(200, 200) # or MolDraw2DCairo to get PNGs
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, rdkit_mol, highlightBonds=rdkit_selected_ids)
    d.FinishDrawing()
    return SVG(d.GetDrawingText())

def draw_flat_rdkit(molecule: DINCMolecule) -> SVG:

    rdkit_mol = to_rdkit(molecule)
    AllChem.Compute2DCoords(rdkit_mol)

    d = Draw.rdMolDraw2D.MolDraw2DSVG(200, 200) # or MolDraw2DCairo to get PNGs
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(d, rdkit_mol)
    d.FinishDrawing()
    return SVG(d.GetDrawingText())
 