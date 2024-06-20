from os import path
from pathlib import Path
import pandas as pd
from typing import List

from ...ligand import DINCMolecule
from ...ligand.fragment.fragment import DINCFragment
from ...parameters.core import DincCoreParams, DINC_JOB_TYPE
from ...receptor.core import DINCReceptor 
from ...analysis.rmsd import calculate_rmsd 
from ..dinc_job_elem import DINCThreadElem
from ..utils.utils_process_output import extract_vina_conformations
from .dock_engine_thread import DINCDockEngineThread
from ...parameters.dock_engine_vina import VinaEngineParams, SCORE_F

from dinc_ensemble import DINC_RECEPTOR_PARAMS, DINC_CORE_PARAMS
from vina import Vina
debug = True

class DINCDockThreadVina(DINCDockEngineThread):

    def __init__(self, 
                 job_name: str,
                 ligand: DINCMolecule, 
                 receptor: DINCReceptor,
                 output_dir: Path,
                 fragment_index: int = -1,
                 replica_id: int = 0,
                 vina_engine_params: VinaEngineParams = VinaEngineParams()
                 ):
        
        super().__init__(job_name, ligand, receptor, output_dir, fragment_index, replica_id)
        self.vina = self.prepare_vina_object()
        
        results_outfile = self.output_dir / Path("results_frag_{}_rep{}.csv".format(self.frag_index, self.replica)) # type: ignore
        self.results_file = results_outfile
        
        ligand_output = self.output_dir / Path(self.ligand_name+"frag_{}_rep_{}_out.pdbqt".format(self.frag_index,self.replica))
        self.results_pdbqt = ligand_output     

    @property
    def conformations(self):
        if self._result_conformations is None:
            raise ValueError("DINCEnsemble Error: docking not run yet, no conformations!")
        else:
            return self._result_conformations
        
    @property
    def energies(self):
        if self._result_energies is None:
            raise ValueError("DINCEnsemble Error: docking not run yet, no energies!")
        else:
            return self._result_energies
    
    def prepare(self):
        self.prepare_and_set_receptor()
        self.prepare_and_set_ligand()
        self.prepare_and_set_scoring()
    
    def optimize(self):
        self.vina.optimize()
    
    def dock(self):
        self.vina.dock()                                                                                   
        self.vina.write_poses(str(self.results_pdbqt), overwrite=True)
        self._result_conformations = extract_vina_conformations(str(self.results_pdbqt))
        self._result_energies = self.vina.energies()[:, 0]
    
    def randomize(self):
        self.vina.randomize()
    
    def score(self):
        self.vina.score()

    def prepare_vina_object(self):
        v = Vina(sf_name=self.params.score_f,
                 cpu = self.params.cpu_count, 
                 seed = self.params.seed,
                 verbosity=2)
        return v

    def prepare_and_set_receptor(self, dir_path:Path):
        receptor_pdbqt_str = self.receptor.pdbqt_str
        rec_pdbqt_outname = dir_path / Path(self.receptor.name+".pdbqt") # type: ignore
        with open(rec_pdbqt_outname, "w") as f:
            f.write(receptor_pdbqt_str)
        self._receptor_pdbqt_fname = rec_pdbqt_outname
        self.vina.set_receptor(str(rec_pdbqt_outname))

    def prepare_and_set_ligand(self):
        # if iterative docking, set the apropriate ligand for the iteration
        lig_molkit = self.ligand.molkit_molecule
        if self.frag_index != -1:
            lig_molkit = self.fragment.split_frags[self.frag_index]._molecule.molkit_molecule
        pdbqt_str = lig_molkit.pdbqt_str
        
        frag_pdbqt_name = "frag_{}_rep_{}.pdbqt".format(self.frag_index, self.replica_id)
        with open(frag_pdbqt_name, "w") as f:
            f.write(pdbqt_str)
        self.vina.set_ligand_from_string(pdbqt_str)

    def prepare_and_set_scoring(self):

        if self.params.score_f == SCORE_F.VINA or self.score == self.params.score_f == SCORE_F.VINARDO:
            bbox_center = [DINC_RECEPTOR_PARAMS.bbox_center_x,
                           DINC_RECEPTOR_PARAMS.bbox_center_y,
                           DINC_RECEPTOR_PARAMS.bbox_center_z]
            bbox_dims = [DINC_RECEPTOR_PARAMS.bbox_dim_x,
                        DINC_RECEPTOR_PARAMS.bbox_dim_y,
                        DINC_RECEPTOR_PARAMS.bbox_dim_z]
        
            self.vina.compute_vina_maps(
                bbox_center,
                bbox_dims)
        '''
        if self.score == DINC_SCORING.AD4:
            # check that the receptor ad4 maps exist
            # TODO: map generation should be done somewhere on entry
            if not path.exists(self.fragment._gpf_file):
                raise ValueError("DINCError: did not precompute the AD4 gpf file!")
            if not path.exists(self.fragment._maps_fld_file):
                raise ValueError("DINCError: did not precompute the AD4 map fld file!")
            self.vina.load_maps(self.fragment._maps_prefix)
        '''
    
    
        
    def run(self): 

        # note that the job started
        self.prepare()
        self.score()
        self.optimize()
        self.dock()