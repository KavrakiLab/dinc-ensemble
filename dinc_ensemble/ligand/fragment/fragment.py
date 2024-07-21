

from dinc_ensemble.ligand.core import DINCMolecule
from MolKit.torTree import TorTree
from MolKit.molecule import Molecule as MolKitMolecule

from typing import Optional
from AutoDockTools.AutoDockBondClassifier import AutoDockBondClassifier
from MolKit.molecule import AtomSet
from MolKit.molecule import Atom as MolKitAtom
from MolKit.molecule import Bond as MolKitBond
from MolKit.pdbParser import PdbqtParser
from MolKit import makeMoleculeFromAtoms
from mglutil.math.statetocoords import StateToCoords
from AutoDockTools.Conformation import Conformation

import random
from math import isnan
from copy import deepcopy
from pathlib import Path
import pandas as pd
from numpy import array, ndarray

from scipy.spatial.transform import Rotation as R
from numpy import cross, matrix
from numpy.linalg import norm
import collections
import typing

from ...parameters.fragment import *
from ...parameters.core import *
from .molkit_utils import create_tor_tree, \
                            check_tor_tree, \
                            dihedral
from .draw import draw_fragment


import logging
logger = logging.getLogger('dinc_ensemble.ligand')

class DINCFragment:
    
    # this is used for root selection - can be changed here if needed
    HBOND_ELEMENTS = ["N", "O", "F"]
    # threshold for significant displacement
    CONF_THR = 1e-3

    def __init__(self, 
                molecule:DINCMolecule,
                frag_params: DincFragParams = DincFragParams()
                ) -> None:
        
        '''
        Arguments
        ----------
        molecule : DINCMolecule
            The input molecule to be subject to fragmentation
        frag_params: DincFragParams
            Parameters for fragmenting the ligand.
            Includes:
            frag_mode : DINC_FRAGMENT_MODE
                The fragmentation mode. 
            frag_size : int
                The total number of active degrees-of-freedom (i.e., torsion angles) in a fragment. 
            frag_new : int
                The maximum number of "new" degrees-of-freedom in a fragment (other than the initial fragment). 
            root_type : DINC_ROOT_TYPE
                The protocol for selecting the root atom of the ligand's torsion tree. 
            root_auto : DINC_ROOT_AUTO
                The protocol for selecting the root atom if root_type is "automatic". 
            root_name : str
                The name of the root atom (if root type is set to user defined)
        '''
        self._molecule  = molecule

        self._frag_mode = frag_params.frag_mode
        self._frag_size = frag_params.frag_size
        self._frag_new  = frag_params.frag_new
        self._root_type = frag_params.root_type
        self._root_auto = frag_params.root_auto
        self._root_atom_name = frag_params.root_name

        if frag_params.root_name is not None:
            self._root_atom = self._get_molkit_atom_(frag_params.root_name)
        else:
            self._root_atom = None

        self.node_dict = {}
        self.molkit_atom_node_dict = {}
        self.molkit_bond_node_dict = {}
        self.split_frags = None
        self._conformation = None

        self._init_tortree_()
        if self._frag_mode == DINC_FRAGMENT_MODE.AUTO:
            # automatic values for fragment growing
            # minimum frag size is 6
            # maximum new frag is 3
            torscnt = molecule.molkit_molecule.torscount
            self._frag_new = min(max(torscnt - 6, 0), 3)
            self._frag_size = torscnt - self._frag_new

        self._select_root_atom_()
        self._init_conformation_()
        
    # ------------ ^ 1. INITIALIZING NODES AND TORSION TREE ^ ---------- #
    def get_bfs_tor_tree_nodes(self, 
                               molkit_molecule: MolKitMolecule):
        # Perform a breadth-first search in the torsion tree of the given ligand to collect all its nodes
        # (the root node is excluded because it does not correspond to any torsion).
        # Return the list of nodes from the torsion tree.
        #
        N = []
        current_level = [molkit_molecule.torTree.rootNode]
        while current_level:
            next_level = [c for n in current_level for c in n.children]
            N.extend(next_level)
            current_level = next_level
        return N


    def __define_nodes__(self):
        '''
        Convert info from torTree into dictionaries for rdkit and molkit reference.
        '''        
        logger.info("-----------------------")
        logger.info("DINCEnsemble: defining node of the fragment #{}".format(self._molecule.molkit_molecule.name))
        logger.info("-----------------------")
        self.node_dict = {}
        self.molkit_atom_node_dict = {}
        self.molkit_bond_node_dict = {}
        logger.info("Defining root node")
        self.molkit_atom_node_dict, \
            self.molkit_bond_node_dict, \
            self.node_dict = self.__add_node_info_to_dict__(self.torTree.rootNode,
                            self.molkit_atom_node_dict, 
                            self.molkit_bond_node_dict, 
                            self.node_dict)
        
        logger.info("Defining other nodes")
        for node in self.torTree.torsionMap:
            
            self.molkit_atom_node_dict, \
            self.molkit_bond_node_dict, \
            self.node_dict = self.__add_node_info_to_dict__(node,
                                self.molkit_atom_node_dict, 
                                self.molkit_bond_node_dict, 
                                self.node_dict)
        self.atoms = self._molecule.atoms
        # assign node ids
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info("Adding node info to atom and bond dataframes!")
        self.atoms["node"] = self.atoms.apply(lambda row:
                                self.molkit_atom_node_dict[row.name] if row.name in self.molkit_atom_node_dict 
                                else -1,
                                axis = 1)
        self.bonds = self._molecule.bonds
        # asign node ids
        self.bonds["node"] = self.bonds.apply(lambda x:
                                              self.molkit_bond_node_dict[x.name] if x.name in self.molkit_bond_node_dict 
                                              else -1,
                                              axis=1)
        
        logger.info("--------------------------")

                                                                        
    def __add_node_info_to_dict__(self, 
                          node, 
                          atoms_dict, 
                          bonds_dict, 
                          node_dict) -> typing.Tuple[typing.Dict, typing.Dict, typing.Dict]:
        
        '''
        Use info from the node to extract important rdkit and molkit info for a dictionary.
        This is useful because of the MolKit convoluted node and atom annotation.
        We will keep everything in dictionaries and dataframes for later easier access.
        atom_dict: atom_dict[molkit_atm_unique_name] = node_idx
        bond_dict: bond_dict[molkit_bond_name] = node_idx
        node_dict: node_dict[node_idx] = node
        Returns updated atom_dict, bond_dict, node_dict
        '''
        logger.info("-----------")
        logger.info("Extracting node info for node #{}".format(node.number))
        ligand_dinc_mol = self._molecule
        node_id = node.number
        logger.info("Adding node #{} to dict".format(node.number))
        node_dict[node_id] = node

        # set atoms info
        logger.info("Node #{} contains {} atoms with tt_inds: {}".format( \
            node.number,
            len(node.atomList),
            node.atomList))

        atom_list = ligand_dinc_mol.molkit_molecule.allAtoms.get(
                        lambda x: x.tt_ind in node.atomList
                    )
        atom_list_name = [a.name for a in atom_list]
        logger.info("Adding {} atoms to the dict.".format(len(atom_list)))
        logger.info("Adding atoms to the dict with unique ids: {} ".format(atom_list_name))
        for i in atom_list_name: 
            if i not in atoms_dict:
                atoms_dict[i] = node_id

        # set bond info
        bond_molkit = node.bond
        if bond_molkit[0] is not None and bond_molkit[1] is not None:
            molkit_atoms = ligand_dinc_mol.molkit_molecule.allAtoms.get(
                        lambda x: x.tt_ind in list(bond_molkit)
                     )
            if len(molkit_atoms) < 2:
                logger.warning("DINC Warning: unable to extract bond info from torsion node")
                return (atoms_dict, bonds_dict, node_dict)
            molkit_at1 = molkit_atoms[0].name
            molkit_at2 = molkit_atoms[1].name
            bonds_df = ligand_dinc_mol.bonds
            sele_bond = bonds_df[(
                (bonds_df["atom1_molkit_unique_name"]==molkit_at1) & 
                (bonds_df["atom2_molkit_unique_name"]==molkit_at2) 
                ) | (
                (bonds_df["atom1_molkit_unique_name"]==molkit_at2) & 
                (bonds_df["atom2_molkit_unique_name"]==molkit_at1) 
                )]
            if len(sele_bond) > 0:
                sele_b = sele_bond.iloc[0]
                bond_idx = sele_b.name
                logger.info("~~~~~~~~~~")
                logger.info("Adding {} bond to the dict.".format(sele_b.name))
                if bond_idx not in bonds_dict:
                    bonds_dict[bond_idx] = node_id
            else:
                logger.info("~~~~~~~~~~")
                logger.warning("DINC Warning: unable to extract bond info from torsion node")
       
        else:
            logger.info("~~~~~~~~~~")
            logger.warning("DINC Warning: unable to extract bond info from torsion node")
        
        logger.info("-----------")
        return (atoms_dict, bonds_dict, node_dict)


    def _init_tortree_(self) -> None:

        '''
        Generates an initial torsion tree - the nodes are ever the same (defined by active torsions).
        If root atom is none, the root picked here will be the one defined in the molkit at construction.
        '''
        molkit_mol = self._molecule.molkit_molecule
        if self._root_atom is None:
            root_atom = molkit_mol.ROOT
        else:
            root_atom = self._root_atom

        create_tor_tree(molkit_mol, root_atom)
        check_tor_tree(molkit_mol.torTree.rootNode, None)

        self.torTree = self._molecule.molkit_molecule.torTree
        self.__define_nodes__()
        
        self.bfs_ordered_nodes = self.get_bfs_tor_tree_nodes(self._molecule.molkit_molecule)

    # ------------ ~ INITIALIZING NODES AND TORSION TREE ~ ---------- #

    # ------------ ^ SELECTING THE ROOT ATOM ^ ---------- #

    def _find_max_ini_fragment_(self, atoms_df):
        '''
        Finds a set of nodes in the initial fragment such that the number of atoms given in 
        atoms_df is maximized.
        Performs bfs across all nodes with the given frag_size and count atoms in fragment
        Useful for selecting the root atom based on a given condition

        Returns: list of nodes in the maximum initial fragment
        '''
        debug = False
        frag = self
        initial_frag_size = self._frag_size
        logger.info("DINC: selecting initial fragment to maximize condition")
        logger.info("------------------------------------------------------")
        if initial_frag_size >= len(frag.node_dict):
            logger.info("initial sragment is full ligand, all nodes")
            logger.info("------------------------------------------------------")
            all_nodes = deepcopy(frag.bfs_ordered_nodes)
            all_nodes.append(self.node_dict[0])
            return all_nodes
        max_ini_frag_size = 0   
        max_frag = []      
        for node_id, node in frag.node_dict.items():
            tmp_root_node = node
            logger.info("current root in consideration #{}".format(tmp_root_node.number))
            logger.info("with atoms {}".format(tmp_root_node.atomList))
            visited = set()
            queue = collections.deque([tmp_root_node])
            visited.add(tmp_root_node)
            while queue and len(visited) < initial_frag_size:
                cur_node = queue.popleft()
                neighbors = set()
                if cur_node.parent is not None:
                    neighbors = neighbors.union(set(cur_node.parent))
                if cur_node.children is not None:
                    neighbors = neighbors.union(set(cur_node.children))
                for neighbor in list(neighbors):
                    if len(visited) >= initial_frag_size:
                        break
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            ini_frag_nodes = [v.number for v in list(visited)]
            ini_frag_atoms = atoms_df[atoms_df.node.isin(ini_frag_nodes)]
            logger.info("resulting initial fragment")
            logger.info(ini_frag_nodes)
            logger.info("with {} atoms".format(len(ini_frag_atoms)))
            if len(ini_frag_nodes) > max_ini_frag_size:
                logger.info("updating max fragment")
                max_ini_frag_size = len(ini_frag_atoms)
                max_frag = ini_frag_nodes
        logger.info("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        logger.info("resulting max fragment")
        logger.info(max_frag)
        logger.info("with {} atoms".format(max_ini_frag_size))
        logger.info("------------------------------------------------------")
        max_frag_nodes = [frag.node_dict[i] for i in max_frag]
        return max_frag_nodes

    def _select_root_atom_(self) -> None:

        molkit_heavy_atoms = self._molecule.molkit_molecule.allAtoms.get(lambda a: a.element != "H")
        logger.info(molkit_heavy_atoms)
        selected_root_name = None
        logger.info("DINC: Selecting fragment root atom.")
        logger.info(self._root_type)
        logger.info(self._root_auto)
        logger.info(self._root_atom_name)
        if self._root_type is None:
            raise ValueError("DINC Error: Fragment root type not given!")
        elif self._root_type is DINC_ROOT_TYPE.RANDOM:
            selected_root_name = random.choice(molkit_heavy_atoms).name

        elif self._root_type is DINC_ROOT_TYPE.USER: 
            if self._root_atom_name is None:
                raise ValueError("DINC Error: Fragment root type is user defined, \
                but the root atom name not given!")
            else:
                selected_root_name = self._root_atom_name

        elif self._root_type is DINC_ROOT_TYPE.AUTO:
            # TODO: Make sure this is actually done for the initial fragment and not just the root node
            if self._root_auto is None:
                raise ValueError("DINC Error: Fragment root type is automatic, \
                but the root type not given!")
            elif self._root_auto is DINC_ROOT_AUTO.FIRST:
                selected_root_name = molkit_heavy_atoms[0].name
            elif self._root_auto is DINC_ROOT_AUTO.LAST:
                selected_root_name = molkit_heavy_atoms[-1].name
            elif self._root_auto is DINC_ROOT_AUTO.LARGEST:
                # largest: the root atom is the heavy atom producing the largest initial fragment,
                # where the size of a fragment is defined as the number of heavy atoms it contains
                heavy_atoms_df = self.atoms[self.atoms.element != "H"]
                max_frag = self._find_max_ini_fragment_(heavy_atoms_df)
                if len(max_frag) > 0:
                    max_nodes = max_frag
                    max_node_ids = [max_node.number for max_node in max_nodes] 
                    max_atom_name = heavy_atoms_df[heavy_atoms_df["node"].isin(max_node_ids)].iloc[0].name
                else:
                    raise ValueError("DINCEnsemble: No largest fragment found!")
                selected_root_atom = self._molecule.\
                    molkit_molecule.allAtoms.get(lambda x: x.name == max_atom_name)[0]
                selected_root_name = selected_root_atom.name
                
            elif self._root_auto is DINC_ROOT_AUTO.H_BONDS:
                # H_bonds: the root is the atom maximizing the H-bond potential in the initial fragment
                # H bond potential atoms - O / N / F
                sele_atoms_df = self.atoms[self.atoms.element.isin(self.HBOND_ELEMENTS)]
                if len(sele_atoms_df) == 0:
                    # if there are no potential H bond atoms resort to the heavy atoms
                    sele_atoms_df = self.atoms[self.atoms.element != "H"]
                    
                max_frag = self._find_max_ini_fragment_(sele_atoms_df)
                if len(max_frag) > 0:
                    max_nodes = max_frag
                    max_node_ids = [max_node.number for max_node in max_nodes] 
                    max_atom_name = sele_atoms_df[sele_atoms_df["node"].isin(max_node_ids)].iloc[0].name
                else:
                    raise ValueError("DINCEnsemble: No largest hbond fragment found!")
                selected_root_atom = self._molecule.\
                    molkit_molecule.allAtoms.get(lambda x: x.name == max_atom_name)[0]
                selected_root_name = selected_root_atom.name
            else:
                ValueError("DINCEnsemble: wrong root auto given to fragment. Can't select root.")
        else:
            ValueError("DINCEnsemble: wrong root type given to fragment. Can't select root.")

        # confirm the root selection and re-init the torsion tree
        logger.info("DINC-Ensemble selected root atom: {}".format(selected_root_name))
        selected_root_atom = self._get_molkit_atom_(selected_root_name)
        if selected_root_atom != self._root_atom:
            self._root_atom = selected_root_atom
            self._root_atom_name = selected_root_name
            self._init_tortree_()

    # ------------ ~ SELECTING THE ROOT ATOM ~ ---------- #

    # ------------ ^ SPLITTING FRAGMENTS ^ ---------- #

    def _split_to_fragments_(self):
        '''
        Splits the current fragment into consecutive fragments of frag_size.
        Initializing self.split_frags
        Using self._cut_fragment_ to chop up pieces of the molecule

        Returns: the split fragments
        '''
        nodes_N = len(self.bfs_ordered_nodes)
        cut_tor_size = self._frag_size
        i_frag = 0
        self.split_frags = []
        logger.info("---------------------")
        logger.info("DINCEnsemble: splitting fragments of {}".format(self._molecule.molkit_molecule.name))
        logger.info("---------------------")
        logger.info("Splitting ligand of tors size {} to fragments of size {}".format(nodes_N, cut_tor_size))
        
        prev_mol = None
        while cut_tor_size < nodes_N:
            logger.info("---------------------")
            logger.info("Splitting fragment #{}".format(i_frag))
            logger.info("---------------------")
            tmp_mol = self._cut_fragment_(self._molecule,
                                cut_tor_size, frag_id=i_frag)
            tmp_params = DincFragParams(frag_mode=DINC_FRAGMENT_MODE.MANUAL,
                                    frag_new=self._frag_new, frag_size=self._frag_size,
                                    root_type=DINC_ROOT_TYPE.USER, 
                                    root_name=self._root_atom_name)
            tmp_frag = DINCFragment(tmp_mol, tmp_params)
            self.split_frags.append(tmp_frag)
            i_frag += 1
            cut_tor_size += self._frag_new
        self.split_frags.append(deepcopy(self))
        # adding final full fragment to the list too - fully flexible!!

        logger.info("---------------------")
        logger.info("Split into {} fragmens".format(len(self.split_frags)))
        logger.info("~~~~~~~~~~~~~~~~~~~~~")
        return self.split_frags


    def _delete_molkit_atoms_(self, 
                             molkit_molecule: MolKitMolecule,
                             del_atoms: list[MolKitAtom]):
        
        # TODO: this can go to molkit_utils
        # remove from the neighbor atom bond list
        # the bonds associated with the deleted atoms
        list(
            map(
                lambda a:[ #remove bond from the neighboring atoms
                           b.neighborAtom(a).bonds.remove(b) \
                           #for all bonds belonging to the atom being deleted
                           for b in a.bonds 
                           #if the bond is exists in the neighbor atom
                           if b in b.neighborAtom(a).bonds],
                del_atoms,
            )
        )
        # remove nonbonded atoms if any remain
        for a in molkit_molecule.allAtoms:
            if len(a.bonds) == 0:
                del_atoms.append(a)
        # delete all those atoms
        molkit_molecule.allAtoms -= AtomSet(del_atoms)
        return molkit_molecule

    def _reconstruct_bonds_(self, 
                            old_molkit_mol: MolKitMolecule, 
                            new_molkit_mol: MolKitMolecule):
        # TODO: this can go to molkit_utils
        # reconstruct bonds from the old molecule to the new one
        # important for the atom names to be the same in the two molecules
        old_bonds = old_molkit_mol.allAtoms.bonds[0]
        new_atoms = new_molkit_mol.allAtoms
        new_bonds = []
        for old_bond in old_bonds:
            old_at1 = old_bond.atom1
            new_a1 = [a for a in new_atoms if a.name==old_at1.name]
            if len(new_a1) == 0:
                continue
            new_a1 = new_a1[0]
            old_at2 = old_bond.atom2
            new_a2 = [a for a in new_atoms if a.name==old_at2.name]
            if len(new_a2) == 0:
                continue
            new_a2 = new_a2[0]
            if new_a1.isBonded(new_a2): continue
            bond = MolKitBond(new_a1, new_a2, check=0)
            new_bonds.append(bond)
            if hasattr(old_bond, "type"):
                bond.type = old_bond.type
            if hasattr(old_bond, "bondOrder"):
                bond.bondOrder = old_bond.bondOrder
        if len(new_bonds) != 0:
            new_molkit_mol.bondsflag = 1
            new_molkit_mol.hasBonds = 1

    def _cut_fragment_(self, 
                       molecule: DINCMolecule,
                       frag_size: int, 
                       frag_id: Optional[int] = None
                       ) -> DINCMolecule:
        '''
        From the given DINCMolecule create a new DINCMolecule with a subset of its atoms.
        The atom subset is chosen as a number of nodes from the initial torTree to include 
        the nodes before velonging to frag_size.

        Returns:
        DINCMolecule that represents the cut
        '''
        #molkit_mol = deepcopy(self._molecule.molkit_molecule)
        logger.info("DINCEnsemble: cutting a fragment from molecule {}".format(molecule.molkit_molecule.name))
        logger.info("---------------------")
        logger.info("Frag size = {}".format(frag_size))
        
        bfs_nodes = self.get_bfs_tor_tree_nodes(molecule.molkit_molecule)
        
        if frag_size >= len(bfs_nodes):
            logger.info("---------------------")
            logger.info("Fragment size larger or equal than number of nodes.\n\
                            No need to split")
            logger.info("~~~~~~~~~~~~~~~~~~~~~")
            return self._molecule
        
        fragment_molkit = deepcopy(molecule.molkit_molecule)

        # delete atoms that don't fall in this fragment
        cutoff =  frag_size
        logger.info("Deleting atoms belonging to nodes in the bfs further than\
                        {} nodes".format(cutoff))
        del_atoms = [
            a
            for n in bfs_nodes[cutoff:]
            for i in n.atomList
            for a in fragment_molkit.allAtoms
            if a.tt_ind == i
        ]
        logger.info("Deleting {} atoms from the molecule.".format(len(del_atoms)))
        self._delete_molkit_atoms_(fragment_molkit, del_atoms)

        # generate this as a new molkit molecule
        # TODO: freeze previously explored bonds - 
        # probably not at this point, but as part of the split
        # fragment initial torTrees have to be all with same active bonds
        # for the conformation expansion
        # designated bonds can be frozen a bit later
        
        # convert this new cut into a molecule
        # 1 - initialize the new pdbqt parser
        # 2 - make molecule from atoms
        # 3 - reassign bond orders appropriately to match the previous bonds
        # 4 - initialize
        logger.info("Converting the remaining atoms into a molecule")
        new_mol_name = "tmp"
        if frag_id is not None:
            new_mol_name = molecule.molkit_molecule.name + "_frag_{}".format(frag_id)
        # parser is necessary to initialize properly DINCMolecule later 
        tmp_parser = PdbqtParser(filename = new_mol_name + ".pdbqt", 
                         allLines = fragment_molkit.pdbqt_str)
        
        for a in fragment_molkit.allAtoms:
            if not hasattr(a, "occupancy"):
                a.occupancy = 1
            if not hasattr(a, "temperatureFactor"):
                a.temperatureFactor = 50
        frag_new_tmp = makeMoleculeFromAtoms(new_mol_name, fragment_molkit.allAtoms)
        frag_new = deepcopy(frag_new_tmp)
        frag_new.parser = tmp_parser
        # reassign bond orders to make sure they are not lost!
        # frag_new.allAtoms.bonds[0] = fragment_molkit.allAtoms.bonds[0]
        self._reconstruct_bonds_(old_molkit_mol=fragment_molkit,
                                 new_molkit_mol=frag_new)
        # gotta make sure when we initialize 
        frag_new_mol = DINCMolecule(frag_new, prepare=False)
        return frag_new_mol
    

    def _to_df_frags_info_(self, 
                    out_dir: str = ".",
                    start: int = 0,
                    end: int = None):
        
        if self.split_frags is None:
            logger.info("First split the fragments before writing them!")
            return 
        p = Path(out_dir)
        frag_id = []
        frag_svg = []
        frag_dofs = []
        frag_pdbqts = []
        for i, elem in enumerate(self.split_frags[start:end]):
            elem_molkit_mol = elem._molecule.molkit_molecule
            suffix = i
            if i == len(self.split_frags)-1:
                suffix = "full"
            elem_path_svg = p / (elem_molkit_mol.name+"_frag_{}.svg".format(suffix))
            elem_path_pdbqt = p / (elem_molkit_mol.name+"_frag_{}.pdbqt".format(suffix))
            
            tmp_svg = draw_fragment(elem)
            frag_svg.append(tmp_svg.data)
            frag_id.append(i)
            frag_dofs.append(len(elem.torTree.torsionMap))
            frag_pdbqts.append(elem_path_pdbqt)

        data_df = pd.DataFrame(
            {
                "frag_id": frag_id,
                "frag_dof": frag_dofs,
                "frag_pdbqt": frag_pdbqts,
                "frag_svg": frag_svg
            }
        )
        return data_df

    def _write_svg_frags_(self, 
                    out_dir: str = ".",
                    start: int = 0,
                    end: int = None):
        if self.split_frags is None:
            logger.info("First split the fragments before writing them!")
            return 
        p = Path(out_dir)
        frag_id = []
        frag_svg = []
        for i, elem in enumerate(self.split_frags[start:end]):
            elem_molkit_mol = elem._molecule.molkit_molecule
            suffix = i
            if i == len(self.split_frags)-1:
                suffix = "full"
            elem_path_svg = p / (elem_molkit_mol.name+"_frag_{}.svg".format(suffix))
    
            tmp_svg = draw_fragment(elem)
            frag_svg.append(tmp_svg.data)
            with open(elem_path_svg, "w") as f:
                f.write(tmp_svg.data)
            frag_id.append(i)
        data_df = pd.DataFrame(
            {
                "frag_id": frag_id,
                "frag_svg": frag_svg
            }
        )
        return data_df

    def _write_pdbqt_frags_(self, 
                    out_dir: str = ".",
                    start: int = 0,
                    end: int = None):
        if self.split_frags is None:
            logger.info("First split the fragments before writing them!")
            return 
        p = Path(out_dir)
        frag_pdbqt_file = []
        frag_id = []
        for i, elem in enumerate(self.split_frags[start:end]):
            elem_molkit_mol = elem._molecule.molkit_molecule
            suffix = i
            if i == len(self.split_frags)-1:
                suffix = "full"
            elem_path_pdbqt = p / (elem_molkit_mol.name+"_frag_{}.pdbqt".format(suffix))
            
            with open(elem_path_pdbqt, "w") as f:
                f.write(elem_molkit_mol.pdbqt_str) 

            frag_pdbqt_file.append(elem_path_pdbqt)
            frag_id.append(i)
        data_df = pd.DataFrame(
            {
                "frag_id": frag_id,
                "frag_pdbqt_file": frag_pdbqt_file
            }
        )
        ligand_name = self._molecule.molkit_molecule.name
        outfile = p / (ligand_name+"_pdbqt_info.csv")
        data_df[["frag_id", "frag_pdbqt_file"]].to_csv(outfile)
        return data_df
    # ------------ ~ SPLITTING FRAGMENTS ~ ---------- #
    
    # ------------ ~ FRAGMENT CONFORMATIONS ~ ---------- #

    def _init_conformation_(self,
                            origin = [0, 0, 0],
                            translation = [0, 0, 0],
                            quat = [1, 0, 0, 0],
                            torsions = None,
                            coords = None) -> None:
        '''
        Function to intialize this fragment's conformation.
        This can't be applied to the molecule itself, 
        because we need a torsion tree initialized.
        '''
        molkit_mol = self._molecule.molkit_molecule
        if torsions == None:
            torsions = [0] * molkit_mol.TORSDOF
        
        molkit_mol.stoc = StateToCoords(
            molkit_mol, 
            [0, 0, 0], 
              0)
        c = Conformation(molkit_mol, 
                 origin, 
                 translation,
                quat, 
                torsions,
                coords)
        c.getCoords()
        self._conformation = c
        # fix cases where coords were assignes as matrices
        # molkit bug after quarternion application
        for a in self._molecule.molkit_molecule.allAtoms:
            if isinstance(a.coords, ndarray):
                a.coords = list(a.coords)
            x, y, z =  a.coords
            if type(x) is matrix:
               a.coords[0] = x[0,0]
            if type(y) is matrix:
               a.coords[1] = y[0,0]
            if type(z) is matrix:
               a.coords[2] = z[0,0]

        # also update nodes 
        # to include info on the dihedral angle atoms
        # and the dihedral andgle of each tor node
        self._update_dihedral_angle_info()
        self._molecule.__reset__(self._molecule.molkit_molecule, reset_rdkit=False)

    def _update_dihedral_angle_info(self):
        for node_id, node in self.node_dict.items():
            logger.info("Updating dihedral info for node {}".format(node_id))
            if node_id == 0:
                continue
            if not hasattr(node, "dihedral_atoms"):
                a1 = node.a
                b1 = node.b
                if a1 == b1:
                    parent = node.parent[0]
                    b1_list = list(set(parent.atomSet)-set([a1]))
                    while len(b1_list) == 0 and parent.parent is not None and len(parent.parent)>0:
                        parent = parent.parent[0]
                        b1_list = list(set(parent.atomSet)-set([a1]))
                    if len(b1_list) == 0:
                        ValueError("Unable to calculate dihedral angle!")
                    else:
                        b1 = b1_list[0]
                        node.b = b1
                possibleAtoms = list(set(node.atomSet) - set([a1, b1]))
                nodes_order = [i for i, j in enumerate(self.bfs_ordered_nodes) if j == node][0]
                children = collections.deque(self.bfs_ordered_nodes[nodes_order:])
                while len(possibleAtoms) < 2 and len(children) > 0:
                    child = children.popleft()
                    possibleAtoms = list(set(child.atomSet) - set([a1, b1]))

                if len(possibleAtoms) >= 2:
                    a2 = possibleAtoms[0]
                    b2 = possibleAtoms[1]
                else:
                    a2_posibilities = [a for b in a1.bonds 
                            for a in 
                            set([b.atom1, b.atom2]) - set([a1, b1])]
                    
                    if len(a2_posibilities) == 0:
                        raise ValueError("Unable to calculate dihedral angle!")
                    else:
                        a2 = a2_posibilities[0]
                    
                    b2_posibilities = [a for b in b1.bonds 
                            for a in 
                            set([b.atom1, b.atom2]) - set([a1, b1, a2])]
                    
                    if len(b2_posibilities) == 0:
                        raise ValueError("Unable to calculate dihedral angle!")
                    else:
                        b2 = b2_posibilities[0]

                dihedral_atoms = [a2, a1, b1, b2]
                node.dihedral_atoms = dihedral_atoms
            else:
                dihedral_atoms = node.dihedral_atoms
            #get atoms associated with the given bond
            node.dihedral_angle = dihedral(
                                    dihedral_atoms[0].coords,
                                    dihedral_atoms[1].coords,
                                    dihedral_atoms[2].coords,
                                    dihedral_atoms[3].coords
                                    )


    def _get_reference_frame_(self):
        
        Ac = self._molecule.molkit_molecule.allAtoms[0:3]
        Xc = array(Ac[1].coords) - array(Ac[0].coords)
        Xc /= norm(Xc)
        Yc = cross(Xc, array(Ac[2].coords) - array(Ac[0].coords))
        Yc /= norm(Yc)
        Zc = cross(Xc, Yc)
        return [Xc, Yc, Zc]

    def _expand_conf_(self, 
                      new_frag):
        '''
         Expand the given conformation of the self into the new fragment.
         In practice, the expansion is done by applying the given conformation to the new fragment:
         this kinematic transformation involves applying the torsions of the conformation to the new
         fragment (i.e., to the part of the new fragment that is shared with the previous fragment)
         as well as translating and rotating the new fragment so that it fits the given conformation.
         Since applying torsions can create errors that easily propagate along the kinematic chain,
         after the expansion, we correct atom positions in the fragment using coordinates from the
         conformation.
           self: fragment with the conformation that has to be expanded
           new_frag (modified): fragment into which the given conformation should be expanded
        '''
        # calculate the translation we will apply to the new fragment so that it fits the conformation
        ref_root_coord = self._root_atom.coords
        mobile_root_coord = new_frag._root_atom.coords
        translation = array(ref_root_coord) - array(mobile_root_coord)
        for i, elem in enumerate(translation):
            if translation[i] < self.CONF_THR:
                translation[i] = 0
        # get torsion mappings
        torsions = []
        ref_node_atoms = {}
        for node_id, node in self.node_dict.items():
            node_atoms = self.atoms[self.atoms.node == node_id]
            node_atom_names = set(node_atoms.index)            
            ref_node_atoms[frozenset(node_atom_names)] = node

        for mobile_node in new_frag.torTree.torsionMap:
            mobile_node_id = mobile_node.number
            if mobile_node_id == 0:
                torsions.append(0)
                continue
            mobile_node = new_frag.node_dict[mobile_node_id]
            mobile_node_atoms = new_frag.atoms[new_frag.atoms.node == mobile_node_id]
            mobile_node_atom_names = frozenset(mobile_node_atoms.index)
            
            if mobile_node_atom_names in ref_node_atoms:
                ref_node = ref_node_atoms[mobile_node_atom_names]
                tor_val = ref_node.dihedral_angle - mobile_node.dihedral_angle
                if tor_val > self.CONF_THR:
                    torsions.append(tor_val)
                else:
                    torsions.append(0)
            else:
                torsions.append(0)
    
        # apply the translation and torsions to the new fragment so that it fits the conformation
        new_frag._init_conformation_(
                            translation = translation,
                            torsions = torsions)   
        
        # get rotation mappings 
        # create a reference frame associated with the conformation
        Ac = self._molecule.molkit_molecule.allAtoms[0:3]
        Xc = array(Ac[1].coords) - array(Ac[0].coords)
        Xc /= norm(Xc)
        Yc = cross(Xc, array(Ac[2].coords) - array(Ac[0].coords))
        Yc /= norm(Yc)
        Zc = cross(Xc, Yc)

        # create the corresponding reference frame associated with the new fragment
        Af = [next(x for x in new_frag._molecule.molkit_molecule.allAtoms if x.name == a.name) for a in Ac]
        Xf = array(Af[1].coords) - array(Af[0].coords)
        Xf /= norm(Xf)
        Yf = cross(Xf, array(Af[2].coords) - array(Af[0].coords))
        Yf /= norm(Yf)
        Zf = cross(Xf, Yf)

        # rotate the new fragment so that it fits the conformation
        rotation = matrix([Xc, Yc, Zc]).T * matrix([Xf, Yf, Zf])
        origin = matrix(mobile_root_coord).T
        for a in new_frag._molecule.molkit_molecule.allAtoms:
            a.coords = list((rotation * (matrix(a.coords).T - origin) + origin).flat)
        # TODO: update dihedral angles and the conformation

        # correct mistakes committed when applying the conformation to the new fragment:
        # 1) to atoms belonging only to the new fragment, we apply a translation that correspond to the
        # correction implicitly applied to their closest 'parent' atom belonging to the previous fragment
        corrections = {}
        new_atoms = list(
            set([a.name for a in new_frag._molecule.molkit_molecule.allAtoms])
            - set([a.name for a in self._molecule.molkit_molecule.allAtoms])
        )
        while new_atoms:
            a_name = new_atoms.pop(0)
            at = next(x for x in new_frag._molecule.molkit_molecule.allAtoms if x.name == a_name)
            Af = [
                a for b in at.bonds for a in [b.neighborAtom(at)] if a.name not in new_atoms
            ]
            # if at has a 'parent' atom which is not part of the remaining new atoms
            if Af:
                Ac = self._molecule.molkit_molecule.allAtoms.get(lambda x: x.name == Af[0].name)
                if Ac:
                    # if at's parent atom belongs to the previous fragment,
                    # its associated correction is the difference between the conformation coordinates
                    # and the fragment coordinates of its parent
                    corrections[at] = array(Ac[0].coords) - array(Af[0].coords)
                else:
                    # otherwise, its associated correction is equal to its parent's correction
                    corrections[at] = corrections[Af[0]]
            else:
                new_atoms.append(a_name)

        for at in list(corrections.keys()):
            at.coords += corrections[at]

        # 2) to atoms belonging to the previous fragment, we assign the coordinates from the conformation
        for a in self._molecule.molkit_molecule.allAtoms:
            for x in new_frag._molecule.molkit_molecule.allAtoms:
                if x.name == a.name:
                    x.coords = a.coords

        # initialize the conformation
        new_frag._init_conformation_(coords = [a.coords for a in new_frag._molecule.molkit_molecule.allAtoms])
        # propagate all to the pdbqt_str for writing!
        new_frag._molecule.__reset__(new_frag._molecule.molkit_molecule, reset_rdkit = False)

        new_frag.bonds = new_frag._molecule.bonds
        # asign node ids
        new_frag.bonds["node"] = new_frag.bonds.apply(lambda x:
                                                new_frag.molkit_bond_node_dict[x.name] if x.name in new_frag.molkit_bond_node_dict 
                                                else -1,
                                                axis=1)

    def _activate_all_bonds(self):
        allBonds = [b for b in self._molecule.molkit_molecule.allAtoms.bonds[0] if b.possibleTors]
        for b in allBonds:
            b.activeTors = 1

        self._molecule.__reset__(self._molecule.molkit_molecule, reset_rdkit = False)
        self.bonds = self._molecule.bonds
        self.bonds = self._molecule.bonds
        # asign node ids
        self.bonds["node"] = self.bonds.apply(lambda x:
                                                self.molkit_bond_node_dict[x.name] if x.name in self.molkit_bond_node_dict 
                                                else -1,
                                                axis=1)


    def _freeze_prev_bonds(self, new_frag):
        # randomly choose which bonds from the previous fragment will be inactive in the new fragment
        # and update the number of active DoFs (torscount)
        # NB: the actual number of new bonds in the new fragment is: new_bonds = all_bonds - previous_bonds
        #     and the number of "previous bonds" that have to be kept active is frag_size - new_bonds
        previous_bonds = [
            (b.atom1, b.atom2) for b in self._molecule.molkit_molecule.allAtoms.bonds[0] if b.possibleTors
        ]
        allBonds = [b for b in new_frag._molecule.molkit_molecule.allAtoms.bonds[0] if b.possibleTors]
        for _ in range(self._frag_size - (len(allBonds) - len(previous_bonds))):
            previous_bonds.pop(random.randrange(len(previous_bonds)))
        for b in previous_bonds:
            bond_atoms = set([b[0].name, b[1].name])
            next(
                b for b in allBonds if set([b.atom1.name, b.atom2.name]) == bond_atoms
            ).activeTors = 0
        new_frag._molecule.molkit_molecule.torscount = self._frag_size
        new_frag._init_conformation_(coords = [a.coords for a in new_frag._molecule.molkit_molecule.allAtoms])
        # propagate all to the pdbqt_str for writing!
        new_frag._molecule.__reset__(new_frag._molecule.molkit_molecule, reset_rdkit = False)

        new_frag.bonds = new_frag._molecule.bonds
        # asign node ids
        new_frag.bonds["node"] = new_frag.bonds.apply(lambda x:
                                                new_frag.molkit_bond_node_dict[x.name] if x.name in new_frag.molkit_bond_node_dict 
                                                else -1,
                                                axis=1)

    def _get_molkit_atom_(self, atom_name:str) -> MolKitAtom:
        logger.info(atom_name)
        atoms = self._molecule.molkit_molecule.allAtoms.get(atom_name)
        if len(atoms) > 0:
            return atoms[0]
        else:
            return None 
