from os import path
from pathlib import Path
import pandas as pd
from typing import List
import threading

from ...parameters import DINC_JOB_TYPE, DINC_RECEPTOR_PARAMS, DINC_CORE_PARAMS, SCORE_F
from ...analysis.rmsd import calculate_rmsd 
from ..dinc_job_elem import DINCThreadElem
from ..utils.utils_process_output import extract_vina_conformations, cluster_conformations
from .dinc_thread import DINCDockThread
from ...parameters.dock_engine_vina import VinaEngineParams

from vina import Vina

import logging
logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)


class DINCDockThreadVina(DINCDockThread):

    def __init__(self, 
                 dinc_thread_elem: DINCThreadElem,
                 vina_engine_params: VinaEngineParams
                 ):
        
        super().__init__(dinc_thread_elem)
        
        self.output_dir = self.thread_elem.job_info.job_root
        self.params = vina_engine_params
        self.vina = self.prepare_vina_object()
        results_outfile = self.output_dir / Path("results_frag_{}_rep{}.csv".format(self.frag_index, self.replica)) # type: ignore
        self.results_file = results_outfile
        ligand_output = self.output_dir / Path(self.ligand_name+"frag_{}_rep_{}_out.pdbqt".format(self.frag_index,self.replica))
        self.results_pdbqt = ligand_output
        self.conformations = None
        self.exception = None
        if self.results_pdbqt.exists():
            conformations = extract_vina_conformations(str(self.results_pdbqt))
            self.conformations = conformations     
        self.prepare() 

    def prepare(self):
        self.prepare_and_set_receptor()
        self.prepare_and_set_ligand()
        self.prepare_and_set_scoring()
    
    def optimize(self):
        try:
            self.vina.optimize()
        except Exception as e:
            logger.error("Failed optimizing with Vina thread \n {}".format(e))
            self.exception = e
    
    def dock(self):
        try:
            self.vina.dock(exhaustiveness=self.params.exhaustive,
                       n_poses=self.params.n_poses,
                       max_evals=self.params.max_evals,
                       min_rmsd=self.params.min_rmsd)
        except Exception as e:
            logger.error("Failed docking with Vina thread \n {}".format(e))
            self.exception = e
    
    def randomize(self):
        try:
            self.vina.randomize()
        except Exception as e:
            logger.error("Failed randomizing with Vina thread \n {}".format(e))
            self.exception = e
    
    def score(self):
        try:
            self.vina.score()
        except Exception as e:
            logger.error("Failed scoring with Vina thread \n {}".format(e))
            self.exception = e

    def prepare_vina_object(self):
        try:
            v = Vina(sf_name=self.params.score_f,
                    cpu = self.params.cpu_count, 
                    seed = self.params.seed,
                    verbosity=2)
        except Exception as e:
            logger.error("Failed initializing Vina thread \n {}".format(e))
            self.exception = e
        return v

    def prepare_and_set_receptor(self):
        try:
            receptor_pdbqt_str = self.receptor.pdbqt_str
            dir_path = self.output_dir 
            receptor_name = self.receptor_name
            rec_pdbqt_outname = dir_path / Path(receptor_name+".pdbqt") # type: ignore
            with open(rec_pdbqt_outname, "w") as f:
                f.write(receptor_pdbqt_str)
            self._receptor_pdbqt_fname = rec_pdbqt_outname
            self.vina.set_receptor(str(rec_pdbqt_outname))
        except Exception as e:
            logger.error("Failed preparing receptor for the Vina thread \n {}".format(e))
            self.exception = e

    def prepare_and_set_ligand(self):
        try:
            # if iterative docking, set the apropriate ligand for the iteration
            lig_molkit = self.ligand.molkit_molecule
            if self.fragment is not None:
                lig_molkit = self.fragment.split_frags[self.frag_index]._molecule.molkit_molecule
            pdbqt_str = lig_molkit.pdbqt_str
            
            frag_pdbqt_name = "frag_{}_rep_{}.pdbqt".format(self.frag_index, self.replica)
            
            with open(self.output_dir / frag_pdbqt_name, "w") as f:
                f.write(pdbqt_str)
            self.vina.set_ligand_from_string(pdbqt_str)
        except Exception as e:
            logger.error("Failed preparing ligand for the Vina thread \n {}".format(e))
            self.exception = e

    def prepare_and_set_scoring(self):

        try:
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
        except Exception as e:
            logger.error("Failed ligand maps for the Vina thread \n {}".format(e))
            self.exception = e
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
        
    
    def write_results_load_conf(self, update_fragment=True):
        #1 - write poses                                                                                    
        self.vina.write_poses(str(self.results_pdbqt), 
                              n_poses=self.params.n_poses,
                              overwrite=True)
        conformations = extract_vina_conformations(str(self.results_pdbqt))
        self.conformations = conformations

    def analyze_results(self):
        
        ref_ligand = self.ligand.molkit_molecule
        if DINC_CORE_PARAMS.dock_type == DINC_JOB_TYPE.CROSSDOCK:
            ref_ligand = self.conformations[0].molecule
        #2 - get energies
        energies = self.vina.energies(self.params.n_poses)[:, 0]
        print(len(energies))
        #3 - compute RMSD
        model_ids = []
        rmsds = []
        for i, conf in enumerate(self.conformations):
            rmsd = calculate_rmsd(conf, ref_ligand, self.output_dir)
            rmsds.append(rmsd)
            model_ids.append(i)
        print(len(self.conformations))
        #4 - TODO: plot RMSD vs energy plot
        df_results = pd.DataFrame({"energies": energies,
                                   "rmsds": rmsds,
                                   "model_id":model_ids})
        results_outfile = self.output_dir / Path("results_frag_{}_rep{}.csv".format(self.frag_index, self.replica)) # type: ignore
        df_results.to_csv(results_outfile)
    
    def run(self): 
    # if the thread had already finished just continue to next
        if self.results_pdbqt.exists():

            logger.info("Continuing the job (found output files for thread).")
            logger.info("Continuing from iter step #{}".format(self.frag_index))
            if self.fragment is not None:
                conformations = extract_vina_conformations(str(self.results_pdbqt))
                self.conformations = conformations
            return
        else:
            logger.info("Starting thread")

        # note that the job started
        #self.prepare()
        if self.frag_index == 0:
            logger.info("Randomize and dock")
            self.randomize_and_dock()
        else:
            logger.info("Dock")
            self.dock()
        self.write_results_load_conf()
        self.analyze_results()
        # note that the job ended
        job_name = self.thread_elem.job_info.job_name
        replica = self.thread_elem.data.replica

            
    @classmethod
    def next_step(cls, dinc_job_threads: List, fragment_idx):
        # step between ligand increments in incremental jobs
        # 1 - get all recults data
        all_results = pd.DataFrame({"energies": [],
                                   "rmsds": [],
                                   "model_id":[],
                                   "replica_id":[],
                                   "receptor_id":[],
                                   "fragment_id":[],
                                   "thread_id":[]})
        all_conformations = []
        n_thr = len(dinc_job_threads)
        for i, t in enumerate(dinc_job_threads):
            frag_idx = t.frag_index
            results_file = t.output_dir / Path("results_frag_{}_rep{}.csv".format(frag_idx, t.replica)) # type: ignore
            # results file will have
            res = pd.read_csv(results_file)
            res["energies"] = res["energies"].apply(lambda x: float(x))
            res["replica_id"] = res["energies"].apply(lambda x: t.replica)
            res["receptor_id"] = res["energies"].apply(lambda x: t.receptor_name)
            res["fragment_id"] = res["energies"].apply(lambda x: fragment_idx)
            res["thread_id"] = res["energies"].apply(lambda x: i)
            all_results = pd.concat([all_results, res])
            all_conformations.extend(t.conformations)
        #cluster conformations add that info
        cluster_conformations(all_conformations)
        all_results["clust_energy_rank"] = all_results.apply(lambda x:
                                            dinc_job_threads[int(x["thread_id"])].conformations[int(x["model_id"])].clust_nrg_rank
                                            , axis=1)
        all_results["clust_size_rank"] = all_results.apply(lambda x:
                                            dinc_job_threads[int(x["thread_id"])].conformations[int(x["model_id"])].clust_size_rank
                                            , axis=1)
        all_results = all_results.sort_values(by = ["energies", "rmsds"]).reset_index(drop=True)
        intermediate_result = t.output_dir / Path("results_frag_{}_collection.csv".format(frag_idx))
        all_results.to_csv(intermediate_result)
        # 2 - get best energy conformations
        # 3 - initialize next fragments in threads with those conformations
        next_iter_threads = []
        for i, t in enumerate(dinc_job_threads):
            rec_name = t.receptor_name
            replica_id = t.replica
            rec_res = all_results[all_results["receptor_id"]==rec_name]
            # remove redundancy, get unique clusters
            rec_res = rec_res.sort_values(["clust_energy_rank", "clust_size_rank"]).groupby("clust_size_rank").head(1).reset_index()
            print(rec_res)
            if len(rec_res) > replica_id:
                selected_res = rec_res.iloc[replica_id]
            else:
                selected_res = rec_res.iloc[0]
            print(i)
            print(selected_res)
            top_i_conf_thr =  int(selected_res.thread_id)
            top_i_conf_model =  int(selected_res.model_id)
            top_i_conf = dinc_job_threads[top_i_conf_thr].conformations[top_i_conf_model]
            print(top_i_conf.coords)
            print(t.fragment.split_frags[fragment_idx]._molecule.molkit_molecule.pdbqt_str)
            t.fragment.split_frags[fragment_idx]._init_conformation_(coords = top_i_conf.coords)
            
            if fragment_idx+1 < len(t.fragment.split_frags):
                cur_frag = t.fragment.split_frags[fragment_idx]
                next_frag = t.fragment.split_frags[fragment_idx+1]
                # check if the ligand is in the box!
                cur_frag._expand_conf_(next_frag)
                logger.debug("WRITING EXPANDED CONFORMATIONS")
                trans_dir = t.output_dir / "transition_frag_{}_pdbqts".format(fragment_idx)
                logger.debug("TO: {}".format(trans_dir))
                t.fragment._write_pdbqt_frags_(trans_dir)
                t.thread_elem.data.iterative_step += 1
                new_thread = DINCDockThreadVina(t.thread_elem, t.params)
                new_thread.frag_index = fragment_idx+1
                next_iter_threads.append(new_thread)
        return next_iter_threads
        

    def join(self):
        threading.Thread.join(self)
        if self.exception:
            raise self.exception



