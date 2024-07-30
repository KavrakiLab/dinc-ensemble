from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List
from pathlib import Path

if TYPE_CHECKING:
    from .fragment import DINCFragment
    
from rdkit.Chem.Draw import rdMolDraw2D
from math import isnan
from IPython.display import SVG
from ..core.convert_molecule import to_rdkit

import pymol

yellow = (255/255, 255/255, 0)
green = (153/255, 250/255, 157/255)
blue = (100/255, 169/255, 255/255)
red = (1, 0, 0)

def create_fragment_scene(full_ligand:str, 
						fragment_list: List[str], 
                        out_dir: str):
    pymol.finish_launching(['pymol', '-cq'])
    pymol.cmd.delete("all")
    pymol.cmd.load(full_ligand, 'full_ligand')
    pymol.cmd.color("purple", "all and element C")
    pymol.cmd.set("stick_transparency", 0.6, "full_ligand")
    for i, frag in enumerate(fragment_list):
        pymol.cmd.load(frag, 'fragment', state=i+1)
    pymol.cmd.show("sticks", "all")
    pymol.cmd.color("pink", "fragment and element C")
    output_scene = Path(out_dir) / "fragments.pse"
    pymol.cmd.zoom("all")
    pymol.cmd.save(output_scene)
    pymol.cmd.delete("all")

        



def draw_fragment(frag: DINCFragment, 
                w: int = 400, h: int = 400) -> SVG:

    color_atoms = {
    }
    color_bonds = {
    }
    arads = {}

    rdkit_mol = to_rdkit(frag._molecule, flatten=True)

    # color atoms in node
    for i, row in frag.atoms.iterrows():
        if not row["rdkit_idx"] or isnan(row["rdkit_idx"]):
            continue
        a_id = int(row["rdkit_idx"])
        node_id = int(row["node"])
        molkit_name = row.name
        arads[a_id] = -.3
        if molkit_name == frag._root_atom_name:
            color_atoms[a_id] = [red]
        elif node_id == 0:
            color_atoms[a_id] = [yellow]
        else:
            color_atoms[a_id] = [green]

    # color active bonds between nodes
    for i, row in frag.bonds[frag.bonds.activeTors_==1].iterrows():
        if isnan(row["rdkit_idx"]):
            continue
        b_id = int(row["rdkit_idx"])
        if row["node"] != -1:
            color_bonds[b_id] = [blue]
    
    # color frozedn bonds between nodes
    for i, row in frag.bonds[(frag.bonds.activeTors_==0) & (frag.bonds.possibleTors_==1)].iterrows():
        if isnan(row["rdkit_idx"]):
            continue
        b_id = int(row["rdkit_idx"])
        if row["node"] != -1:
            color_bonds[b_id] = [yellow]

    # color bonds between atoms in the same node
    '''
    for b in rdkit_mol.GetBonds():
        b_id = b.GetIdx()
        a1_id = b.GetBeginAtomIdx()
        a1_node_id = frag.atoms[frag.atoms["rdkit_idx"]==a1_id].node.iloc[0]
        a2_id = b.GetEndAtomIdx()
        a2_node_id = frag.atoms[frag.atoms["rdkit_idx"]==a2_id].node.iloc[0]
        if a1_node_id == a2_node_id:
            color_bonds[b_id] = color_atoms[a1_id]
    '''

    d = rdMolDraw2D.MolDraw2DSVG(400, 400)
    d.DrawMoleculeWithHighlights(rdkit_mol,"",
                                dict(color_atoms),
                                dict(color_bonds),
                                arads,{})
    d.FinishDrawing()
    return SVG(d.GetDrawingText())