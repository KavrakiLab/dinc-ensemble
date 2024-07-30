     
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from MolKit.molecule import Molecule as MolKitMolecule
    from .molecule import DINCMolecule
    from rdkit.Chem.rdchem import Mol as RDKitMol

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw 
from rdkit.Chem import rdmolops

from IPython.display import SVG
import pandas as pd
from copy import deepcopy


import logging
logger = logging.getLogger('dinc_ensemble.ligand')

AD4_ATOM_TYPES_MAP = {
    "A"     : "C",
    "HD"    : "H",
    "HA"    : "H",
    "HAD"   : "H",
    "HDA"   : "H",
    "OD"    : "O",
    "OA"    : "O",
    "OAD"   : "O",
    "ODA"   : "O",
    "ND"    : "N",
    "NA"    : "N",
    "NAD"   : "N",
    "NDA"   : "N",
    "SD"    : "S",
    "SA"    : "S",
    "SAD"   : "S",
    "SDA"   : "S"
}


AD4_BOND_TYPES_MAP = {"1": Chem.rdchem.BondType().SINGLE,
                "2": Chem.rdchem.BondType().DOUBLE,
                "3": Chem.rdchem.BondType().TRIPLE,
                "4": Chem.rdchem.BondType().QUADRUPLE,
                "5": Chem.rdchem.BondType().QUINTUPLE,
                "6": Chem.rdchem.BondType().HEXTUPLE,
                "ar": Chem.rdchem.BondType().AROMATIC,
                "am": Chem.rdchem.BondType().SINGLE,
                "du": Chem.rdchem.BondType().SINGLE,
                "un": Chem.rdchem.BondType().UNSPECIFIED,
                "nc": Chem.rdchem.BondType().ZERO}
    
def pdbqt_str_rdkit_prep(pdbqt_string: str) -> str:
    '''
    prepare the MolKit pdbqt string to be imported in the RDKit molecule.
    '''
    pdbqt_string_lines = pdbqt_string.split("\n")
    clean_pdbqt_lines = []
    for line in pdbqt_string_lines:
        if line.startswith("ATOM"):
            atom_type = line[77:].split()[0]
            if atom_type in AD4_ATOM_TYPES_MAP:
                line = line[:76]+AD4_ATOM_TYPES_MAP[atom_type]
                clean_pdbqt_lines.append(line)
            else:
                line = line[:76]+line[77:]
                clean_pdbqt_lines.append(line)
    clean_pdbqt_str = "\n".join(clean_pdbqt_lines)
    return clean_pdbqt_str

def molkit_to_rdkit(molkit_mol: MolKitMolecule) -> tuple[RDKitMol, 
                                                    Optional[pd.DataFrame],
                                                    Optional[pd.DataFrame]]:
    '''
    Convert the MolKit molecule to the equivalent RDKit molecule.
    Using the PDBQT string and the MolKit Bond information
    '''
    # TODO: reconfigure this and test more
    # one idea - do not rely on uniqIndex but on names somehow?
    pdbqt_string = molkit_mol.pdbqt_str
    pdbqt_str_clean = pdbqt_str_rdkit_prep(pdbqt_string)
    rdkit_mol = AllChem.rdmolfiles.MolFromPDBBlock(pdbqt_str_clean, 
                                                    removeHs=False,
                                                    sanitize=False)
    if rdkit_mol is None:
        logger.info("DINC Warning: Unable to convert the DINCMolecule to Rdkit!")
        return (None, None, None)
    try:
        Chem.SanitizeMol(rdkit_mol)
    except Chem.AtomValenceException:
        # here you may handle the error condition, or just ignore it
        logger.info("DINC Warning: Rdkit atom valence error!")

    rdkit_bonds = {frozenset([b.GetBeginAtom().GetPDBResidueInfo().GetName().split()[0], 
        b.GetEndAtom().GetPDBResidueInfo().GetName().split()[0]]): b 
        for b in rdkit_mol.GetBonds()}
    
    molkit_bonds = {frozenset([b.atom1.name, b.atom2.name]): b for b in  
                    molkit_mol.allAtoms.bonds[0]}

    for key, rdkit_bond in rdkit_bonds.items():
        if key in molkit_bonds:
            molkit_bond = molkit_bonds[key]
            if hasattr(molkit_bond, "type"):
                rdkit_bond.SetBondType(AD4_BOND_TYPES_MAP[molkit_bond.type])
            elif hasattr(molkit_bond, "bondOrder"):
                rdkit_bond.SetBondType(AD4_BOND_TYPES_MAP[str(molkit_bond.bondOrder)])
        else:
            logger.info("DINC Warning: Molkit molecule and RDKit molecule are inconsistent!")
    
    ## after setting these bonds sanitize?
    try:
        Chem.SanitizeMol(rdkit_mol)
    except Chem.AtomValenceException:
        # here you may handle the error condition, or just ignore it
        logger.info("DINC Warning: Rdkit atom valence error!")
    except Chem.KekulizeException:
        logger.info("DINC Warning: Rdkit kekulize exception!")

    # map atoms to one another
    rdkit_atoms = {a.GetPDBResidueInfo().GetName().split()[0]: a
        for a in rdkit_mol.GetAtoms()}
    molkit_atoms = [(a._uniqIndex, a.name, a)
                    for a in molkit_mol.allAtoms]

    # get the atom mapping
    atom_map = pd.DataFrame({"rdkit_atom": [], 
                             "molkit_atom": []})
    for molkit_a in molkit_atoms:
        mk_name = molkit_a[1]
        #try to find a match
        if mk_name in rdkit_atoms:
            rd_a = rdkit_atoms[mk_name]
            new_row = pd.DataFrame({
                "rdkit_atom": [rd_a], 
                "molkit_atom": [molkit_a[2]]
            })
            atom_map = pd.concat([atom_map, new_row])
    atom_map["rdkit_idx"] = atom_map["rdkit_atom"] .apply(lambda x: int(x.GetIdx()))
    atom_map["molkit_idx"] = atom_map["molkit_atom"] .apply(lambda x: int(x._uniqIndex))
    atom_map["name"] = atom_map["molkit_atom"] .apply(lambda x: x.name)
    # bond map
    bond_map = pd.DataFrame({"rdkit_bonds": [], 
                            "molkit_bonds": [], 
                            "rdkit_idx": [], 
                             "molkit_a_id1": [], 
                             "molkit_a_id2": []})
    rdkit_bonds = {frozenset([b.GetBeginAtom().GetPDBResidueInfo().GetName().split()[0], 
        b.GetEndAtom().GetPDBResidueInfo().GetName().split()[0]]): b 
        for b in rdkit_mol.GetBonds()}
    molkit_bonds = {frozenset([b.atom1.name, b.atom2.name]): b for b in  
                molkit_mol.allAtoms.bonds[0]}

    for key, molkit_bond in molkit_bonds.items():
        a1_name = list(key)[0]
        a2_name = list(key)[1]
        rdkit_a1_id = atom_map[
            atom_map["name"]==a1_name
            ]
        if len(rdkit_a1_id) > 0:
            molkit_a1_id = rdkit_a1_id["molkit_idx"].iloc[0]
            rdkit_a1_id = rdkit_a1_id["rdkit_idx"].iloc[0]
        else:
            continue
        rdkit_a2_id = atom_map[
            atom_map["name"]==a2_name
            ]
        if len(rdkit_a2_id) > 0:
            molkit_a2_id = rdkit_a2_id["molkit_idx"].iloc[0]
            rdkit_a2_id = rdkit_a2_id["rdkit_idx"].iloc[0]
        else:
            continue
        rdkit_bond = rdkit_mol.GetBondBetweenAtoms(
            int(rdkit_a1_id), int(rdkit_a2_id)
        )
        if rdkit_bond is not None:
            new_row = pd.DataFrame({
                "molkit_bonds": [molkit_bond],
                "rdkit_bonds": [rdkit_bond],
                "rdkit_idx": [rdkit_bond.GetIdx()], 
                "molkit_a_id1": [molkit_a1_id],
                "molkit_a_id2": [molkit_a2_id],
            })
            bond_map = pd.concat([bond_map, new_row])
    bond_map["rdkit_idx"] = bond_map["rdkit_idx"] .apply(lambda x: int(x))
    bond_map["molkit_a_id1"] = bond_map["molkit_a_id1"] .apply(lambda x: int(x))
    bond_map["molkit_a_id2"] = bond_map["molkit_a_id2"] .apply(lambda x: int(x))
    
    for a in molkit_mol.allAtoms:
        map_elem = atom_map[atom_map["molkit_idx"]==a._uniqIndex]
        if len(map_elem) > 0:
            rdkit_idx = map_elem["rdkit_idx"].iloc[0]
            a.rdkit_idx = int(rdkit_idx)
        else:
            logger.info("DINCEnsemble warning: not all atoms were mapped between the RDKit and MolKit molecules!")
    for b in molkit_mol.allAtoms.bonds[0]:
        map_elem = bond_map[
            bond_map["molkit_bonds"].apply(
                lambda x: x.name==b.name
            )
            ]
        if len(map_elem) > 0:
            rdkit_idx = map_elem["rdkit_idx"].iloc[0]
            b.rdkit_idx = int(rdkit_idx)
        else:
            logger.info("DINCEnsemble warning: not all bonds were mapped between the RDKit and MolKit molecules!")
        
    return rdkit_mol, atom_map, bond_map

def to_rdkit(molecule: DINCMolecule,
             flatten: bool = False,
             removeH: bool = False) -> RDKitMol:
    
    rdkit_mol = deepcopy(molecule._rdkit_molecule)
    if flatten:
        previous_conformers = [x.GetId() for x in rdkit_mol.GetConformers()]
        [rdkit_mol.RemoveConformer(i) for i in previous_conformers]
    if removeH:
        rdkit_mol = rdmolops.RemoveHs(rdkit_mol)
    return rdkit_mol
