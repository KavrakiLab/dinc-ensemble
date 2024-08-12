

from dinc_ensemble.ligand.core import DINCMolecule
from MolKit.torTree import TorTree
from MolKit.molecule import Molecule as MolKitMolecule

from typing import List
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
from ...receptor.core import DINCReceptor

from .molkit_utils import *
from .draw import draw_fragment, create_fragment_scene
from AutoDockTools.MoleculePreparation import AD4LigandWriter

import logging
logger = logging.getLogger('dinc_ensemble.ligand')

class DINCFragment:
    
    # this is used for root selection - can be changed here if needed
    HBOND_ELEMENTS = ["N", "O", "F"]
    # threshold for significant displacement
    CONF_THR = 1e-3

    def __init__(self, 
                molecule:DINCMolecule,
                frag_params: DincFragParams = DincFragParams(),
                # needed for probing
                ref_receptor: Optional[DINCReceptor] = None
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
        if self._frag_mode == DINC_FRAGMENT_MODE.AUTO:
            torscnt = self._molecule.bonds[self._molecule.bonds.possibleTors_ == 1].shape[0]
            self._frag_new  = min(max(torscnt - 6, 0), 3)
            self._frag_size = torscnt - self._frag_new
        else:
            self._frag_size = frag_params.frag_size
            self._frag_new  = frag_params.frag_new
        self._root_type = frag_params.root_type
        self._root_auto = frag_params.root_auto
        self._root_atom_name = frag_params.root_name
        self._ref_receptor = ref_receptor

        if frag_params.root_name is not None:
            self._root_atom = self.get_molkit_atom(frag_params.root_name)
        else:
            self._root_atom = None
            
        self.fragments: List[MolKitMolecule] = None
        self.fragments_dincmol: List[DINCMolecule] = None
        
        self.leaf_frags: List[MolKitMolecule] = None
        self.leaf_frags_dincmol: List[DINCMolecule] = None
        self.select_root()
        self.split_to_fragments()
        #self.split_leafs()


    def select_root(self):
        logger.info("Selecting root atom")
        self._root_atom_name = select_root_atom(self._molecule.molkit_molecule,
                            self._root_type,
                            self._root_atom_name,
                            self._root_auto,
                            self._frag_size,
                            probe_dir=self._molecule.molkit_molecule.name+"_probe",
                            dince_frag=self, dince_rec=self._ref_receptor)
        self._root_atom = self.get_molkit_atom(self._root_atom_name)
        logger.info("Selected atom: {}".format(self._root_atom_name))
    

    def split_to_fragments(self):
        logger.info("Splitting into fragments")
        self.fragments = split_into_fragments(self._molecule.molkit_molecule,
                                self._root_atom_name,
                                self._frag_size,
                                self._frag_new)
        self.fragments_dincmol = [DINCMolecule(f, prepare=False) for f in self.fragments]
        logger.info("Split into: {} fragments".format(len(self.fragments)))

    def split_leafs(self, leaf_size=1):
        logger.info("Splitting leaves")
        template_lig = self.fragments[-1]
        # include root if it has just one child
        leaf_nodes = []
        if len(template_lig.torTree.rootNode.children) <= 1:
            leaf_nodes.append(template_lig.torTree.rootNode)
        # find leaves
        for n in template_lig.torTree.torsionMap:
            if len(n.children) == 0:
                leaf_nodes.append(n)
        leaf_frags = []
        # create frags of them
        for ln in leaf_nodes:
            ln_root_atom = template_lig.allAtoms[ln.atomList[0]]
            template_lig_copy = deepcopy(template_lig)
            splt_frgs = split_into_fragments(template_lig_copy, 
                                             ln_root_atom.name, leaf_size,100)
            leaf_frags.append(splt_frgs[0])
        self.leaf_frags = leaf_frags
        self.leaf_frags_dincmol = [DINCMolecule(l, prepare=False) for l in leaf_frags]
        logger.info("Extracted {} leaves".format(len(leaf_frags)))
        

    def expand_conformation(self, conformation, conf_frag_index):
        
        logger.info("Expanding conformation of fragment")
        prev_frag = self.fragments[conf_frag_index]
        for i, next_frag in enumerate(self.fragments[conf_frag_index:]):
            expand_conf_to_fragment(conformation, 
                                    next_frag,
                                    prev_frag=prev_frag,
                                    root_name=self._root_atom_name,
                                    frag_size=self._frag_size)
            self.fragments_dincmol[conf_frag_index+i] = DINCMolecule(next_frag, prepare=False)
            logger.info("Expanded conformation {}".format(conf_frag_index+i))
            
    def init_conformation(self,
                        frag_index,
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
        
        logger.info("Initializing conformation of fragment")

        frag = self.fragments[frag_index]
        
        # if initializing the state by coordinates 
        # has to be a coords assignment first, because else
        # statetocoords and conformation will ignore them
        if coords is not None:
            logger.info("Updating coordinates")
            for i, a in enumerate(frag.allAtoms):
                a.coords = coords[i]
            frag.allAtoms.updateCoords(coords)

        if torsions == None:
            torsions = [0] * len(frag.torTree.torsionMap)

        frag.stoc = StateToCoords(
            frag, 
            [0, 0, 0], 
              0)
        c = Conformation(frag, 
                 origin, 
                 translation,
                quat, 
                torsions)
        c.getCoords()
        frag.conf = c
        # fix cases where coords were assignes as matrices
        # molkit bug after quarternion application
        for a in frag.allAtoms:
            if isinstance(a.coords, ndarray):
                a.coords = list(a.coords)
            x, y, z =  a.coords
            if type(x) is matrix:
               a.coords[0] = x[0,0]
            if type(y) is matrix:
               a.coords[1] = y[0,0]
            if type(z) is matrix:
               a.coords[2] = z[0,0]
        self.fragments_dincmol[frag_index] = DINCMolecule(frag, prepare=False)
        logger.info("Finished initializing coordinates")

    def freeze_bonds(self):

        self.fragments_dincmol = [self.fragments_dincmol[0]]
        for i, frag in enumerate(self.fragments[:-1]):
            prev_frag = frag
            new_frag = self.fragments[i+1]
            freeze_molkit_bonds(prev_frag, new_frag, 
                                self._frag_size, self._frag_new)
            self.fragments_dincmol.append(DINCMolecule(new_frag, 
                                                       prepare=False))
            
    def activate_all_bonds(self):

        self.fragments_dincmol = []
        for i, frag in enumerate(self.fragments):
            for b in frag.allAtoms.bonds[0]:
                if b.possibleTors:
                    b.activeTors = True
            self.fragments_dincmol.append(DINCMolecule(frag, 
                                                       prepare=False))

    def to_df_info(self, 
                    out_dir: str = ".",
                    start: int = 0,
                    end: int = None, 
                    leaf = False):
        
        if leaf:
            frags = self.leaf_frags
        else:
            frags = self.fragments 

        if frags is None:
            logger.info("First split the fragments before writing them!")
            return 
        p = Path(out_dir)
        frag_id = []
        #frag_svg = []
        frag_dofs = []
        frag_pdbqts = []
        for i, elem in enumerate(frags[start:end]):
            
            suffix = i
            suffix = ""
            if i == len(frags)-1:
                suffix = "_full"
            else:
                suffix = "_frag_{}".format(i)

            elem_path_pdbqt = p / (elem.name+"{}.pdbqt".format(suffix))
            
            #tmp_svg = draw_fragment(elem)
            #frag_svg.append(tmp_svg.data)
            frag_id.append(i)
            frag_dofs.append(len(elem.torTree.torsionMap))
            frag_pdbqts.append(elem_path_pdbqt)

        data_df = pd.DataFrame(
            {
                "frag_id": frag_id,
                "frag_dof": frag_dofs,
                "frag_pdbqt": frag_pdbqts
                #"frag_svg": frag_svg
            }
        )
        return data_df

    def write_pdbqt_frags(self, 
                    out_dir: str = ".",
                    start: int = 0,
                    end: int = None, 
                    leaf = False):
        logger.debug("Writing the PDBQT fragments and info to dir :{}".format(out_dir))
        frags = []
        if leaf:
            frags = self.leaf_frags
        else:
            frags = self.fragments

        if frags is None:
            logger.warning("First split the fragments before writing them!")
            return 
        p = Path(out_dir)
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        frag_pdbqt_file = []
        frag_id = []
        for i, elem in enumerate(frags[start:end]):
            #print(i)
            #print(len(elem.allAtoms))
            elem_molkit_mol = elem
            suffix = ""
            if not leaf:
                if i == len(frags)-1:
                    suffix = "_full"
                else:
                    suffix = "_frag_{}".format(i)
            else:
                suffix = "_leaf_{}".format(i)
                

            elem_path_pdbqt = p / (elem.name+"{}.pdbqt".format(suffix))
            # needed to initialize the pdbqt string
            write_fragment_pdbqt(elem, elem_path_pdbqt)
            
            frag_pdbqt_file.append(elem_path_pdbqt)
            frag_id.append(i)
        if not leaf:
            logger.debug("Fragments found:")
            logger.debug(frag_pdbqt_file)
            create_fragment_scene(frag_pdbqt_file[-1], frag_pdbqt_file, out_dir)
        else:
            logger.debug("Leaves found:")
            logger.debug(frag_pdbqt_file)
            # write also the full_ligand
            elem_path_pdbqt = p / (elem.name+"_full.pdbqt")
            elem = self.fragments[-1]
            # needed to initialize the pdbqt string
            write_fragment_pdbqt(elem, elem_path_pdbqt)
            create_fragment_scene(elem_path_pdbqt, frag_pdbqt_file, out_dir)
            
        id_lab = "frag_id"
        pdbqt_lab = "frag_pdbqt_file"
        if leaf:
            id_lab = "leafid"
            pdbqt_lab = "leaf_pdbqt_file"
            
        data_df = pd.DataFrame(
            {
                id_lab: frag_id,
                pdbqt_lab: frag_pdbqt_file
            }
        )
        ligand_name = self._molecule.molkit_molecule.name
        outfile = p / (ligand_name+"_pdbqt_info.csv")
        data_df[[id_lab, pdbqt_lab]].to_csv(outfile)
        return data_df

    def get_molkit_atom(self, name: str):
        
        atoms = self._molecule.molkit_molecule.allAtoms.get(name)
        if len(atoms) > 0:
            return atoms[0]
        else:
            return None 



    