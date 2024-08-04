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

import numpy as np

import logging
logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)


class DINCDockThreadVina(DINCDockThread):

    def __init__(self, 
                 dinc_thread_elem: DINCThreadElem,
                 vina_engine_params: VinaEngineParams,
                 bbox_center = None,
                 bbox_dims = None,
                 leaf_multi = False,
                 frag_index = None):
        
        super().__init__(dinc_thread_elem)
        
        self.output_dir = self.thread_elem.job_info.job_root
        self.params = vina_engine_params
        self.vina = self.prepare_vina_object()
        if self.fragment is None or self.frag_index < len(self.fragment.fragments):
            results_outfile = self.output_dir / Path("results_frag_{}_rep{}.csv".format(self.frag_index, self.replica)) # type: ignore
            self.results_file = results_outfile
            ligand_output = self.output_dir / Path(self.ligand_name+"frag_{}_rep_{}_out.pdbqt".format(self.frag_index,self.replica))
            self.results_pdbqt = ligand_output
        else:
            results_outfile = self.output_dir / Path("results_final_rep{}.csv".format(self.frag_index, self.replica)) # type: ignore
            self.results_file = results_outfile
            ligand_output = self.output_dir / Path(self.ligand_name+"final_rep_{}_out.pdbqt".format(self.frag_index,self.replica))
            self.results_pdbqt = ligand_output

        self.conformations = None
        self.exception = None
        self.bbox_center = bbox_center
        self.bbox_dims = bbox_dims
        self.leaf_multi = leaf_multi
        if self.results_pdbqt.exists():
            conformations = extract_vina_conformations(str(self.results_pdbqt), 
                                                       self.ligand.molkit_molecule)
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
                    verbosity=0)
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
            if not self.leaf_multi:
                lig_molkit = self.ligand.molkit_molecule
                if self.fragment is not None:
                    if self.frag_index == len(self.fragment.fragments):
                        lig_molkit = self.fragment.fragments[-1] 
                    else:
                        lig_molkit = self.fragment.fragments[self.frag_index]
                pdbqt_str = lig_molkit.pdbqt_str
                
                frag_pdbqt_name = "frag_{}_rep_{}.pdbqt".format(self.frag_index, self.replica)
                
                with open(self.output_dir / frag_pdbqt_name, "w") as f:
                    f.write(pdbqt_str)
                self.vina.set_ligand_from_string(pdbqt_str)
            else:
                leaf_pdbqts = []
                for i, leaf in enumerate(self.fragment.leaf_frags):
                    pdbqt_str = leaf.pdbqt_str
                    leaf_pdbqt_name = "leaf_{}_rep_{}.pdbqt".format(i, self.replica)
                    with open(self.output_dir / leaf_pdbqt_name, "w") as f:
                        f.write(pdbqt_str)
                    leaf_pdbqts.append(pdbqt_str)
                self.vina.set_ligand_from_string(leaf_pdbqts)

                


        except Exception as e:
            logger.error("Failed preparing ligand for the Vina thread \n {}".format(e))
            self.exception = e

    def prepare_and_set_scoring(self):

        try:
            if self.params.score_f == SCORE_F.VINA or \
                self.score == self.params.score_f == SCORE_F.VINARDO:
                if self.bbox_center is None:
                    self.bbox_center = [DINC_RECEPTOR_PARAMS.bbox_center_x,
                                DINC_RECEPTOR_PARAMS.bbox_center_y,
                                DINC_RECEPTOR_PARAMS.bbox_center_z]
                if self.bbox_dims is None:
                    self.bbox_dims = [DINC_RECEPTOR_PARAMS.bbox_dim_x,
                                DINC_RECEPTOR_PARAMS.bbox_dim_y,
                                DINC_RECEPTOR_PARAMS.bbox_dim_z]
                if self.fragment is None or self.frag_index == len(self.fragment.fragments):
                    self.bbox_dims = [self.bbox_dims[0]+1,
                                      self.bbox_dims[1]+1,
                                      self.bbox_dims[2]+1]
                logger.info("Vina preparing bbox : {} {}".format(self.bbox_dims, self.bbox_center))
                self.vina.compute_vina_maps(
                    self.bbox_center,
                    self.bbox_dims)
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
        
    
    def write_results_load_conf(self, update_fragment=True, write=True):
        #1 - write poses
        if write:
            if len(self.vina.poses()) > 0:
                self.vina.write_poses(str(self.results_pdbqt), 
                                n_poses=self.params.n_poses,
                                overwrite=True)
        if Path(self.results_pdbqt).exists():
            conformations = extract_vina_conformations(str(self.results_pdbqt),
                                                    self.ligand.molkit_molecule)
        else:
            # if no poses were generated by vina, select a random pose
            self.randomize()
            self.vina.write_poses(str(self.results_pdbqt), 
                              n_poses=self.params.n_poses,
                              overwrite=True)
            conformations = extract_vina_conformations(str(self.results_pdbqt),
                                                    self.ligand.molkit_molecule)
        self.conformations = conformations

    def analyze_results(self):
        
        ref_ligand = self.ligand.molkit_molecule
        if DINC_CORE_PARAMS.dock_type == DINC_JOB_TYPE.CROSSDOCK:
            ref_ligand = self.conformations[0].molecule
       
        model_ids = []
        rmsds = []
        energies = []
        for i, conf in enumerate(self.conformations):
            rmsd = calculate_rmsd(conf, ref_ligand, self.output_dir)
            rmsds.append(rmsd)
            model_ids.append(i)
            energies.append(conf.binding_energy)
        #print(len(self.conformations))
        #4 - TODO: plot RMSD vs energy plot ?
        df_results = pd.DataFrame({"energies": energies,
                                   "rmsds": rmsds,
                                   "model_id":model_ids})
        results_outfile = self.output_dir / Path("results_frag_{}_rep{}.csv".format(self.frag_index, self.replica)) # type: ignore
        df_results.to_csv(results_outfile)
    
    def run(self): 
    # if the thread had already finished just continue to next
        if (self.fragment is None or self.frag_index < len(self.fragment.fragments)) \
            and self.results_pdbqt.exists():
            logger.info("Continuing the job (found output files for thread).")
            logger.info("Continuing from iter step #{}".format(self.frag_index))
            if self.fragment is not None:
                conformations = extract_vina_conformations(str(self.results_pdbqt), 
                                                           self.ligand.molkit_molecule)
                self.conformations = conformations
            return
        else:
            logger.info("Starting thread")

        # note that the job started
        #self.prepare()
        if self.frag_index == 0:
            logger.info("Randomize and dock")
            self.randomize_and_dock()
        elif self.fragment is not None and self.frag_index < len(self.fragment.fragments):
            logger.info("Dock")
            self.dock()
        #if the final step is in question - activate all bonds and optimize?
        else:
            logger.info("Optimize")
            self.optimize()
            self.params.exhaustive = 1
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
            t.write_results_load_conf()
            #print(t.frag_index)
            #print(t.replica)
            #print(t.receptor)
            #print(t.results_pdbqt)
            #print(t.conformations)
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

        if all_conformations > 0:
            #cluster conformations add that info
            cluster_conformations(all_conformations)
            all_results["clust_energy_rank"] = all_results.apply(lambda x:
                                                dinc_job_threads[int(x["thread_id"])].conformations[int(x["model_id"])].clust_nrg_rank
                                                , axis=1)
            all_results["clust_size_rank"] = all_results.apply(lambda x:
                                                dinc_job_threads[int(x["thread_id"])].conformations[int(x["model_id"])].clust_size_rank
                                                , axis=1)
            all_results = all_results.sort_values(by = ["energies", "clust_size_rank", "clust_energy_rank"]).reset_index(drop=True)
            top_results = all_results.groupby(["replica_id"]).head(10).reset_index()
        
        intermediate_result = t.output_dir / Path("results_frag_{}_collection.csv".format(frag_idx))
        all_results.to_csv(intermediate_result)
        # 2 - get best energy conformations
        # 3 - initialize next fragments in threads with those conformations
        next_iter_threads = []
        for i, t in enumerate(dinc_job_threads):
            #print(t.receptor_name)
            #print(t.replica)
            #print(t.result_pdbqt)
            rec_name = t.receptor_name
            replica_id = t.replica
            rec_res = all_results[all_results["receptor_id"]==rec_name]
            # remove redundancy, get unique clusters
            rec_res = rec_res.sort_values(["clust_energy_rank", "clust_size_rank"]).groupby("clust_size_rank").head(1).reset_index()
            #print(rec_res)
            if len(rec_res) > replica_id:
                selected_res = rec_res.iloc[replica_id]
            else:
                selected_res = rec_res.iloc[0]

            top_i_conf_thr =  int(selected_res.thread_id)
            top_i_conf_model =  int(selected_res.model_id)
            top_i_conf = dinc_job_threads[top_i_conf_thr].conformations[top_i_conf_model]
            
            if fragment_idx+1 < len(t.fragment.fragments):
                
                t.fragment.init_conformation(fragment_idx, coords = top_i_conf.coords)
                cur_frag = t.fragment.fragments[fragment_idx]
                # check if the ligand is in the box!
                t.fragment.expand_conformation(cur_frag.conf, fragment_idx)
                bbox_d = None
                bbox_c = None
                t.thread_elem.data.iterative_step += 1
                new_thread = DINCDockThreadVina(t.thread_elem, 
                                                t.params,
                                                frag_index=frag_idx,
                                                bbox_dims=bbox_d,
                                                bbox_center=bbox_c)
                new_thread.frag_index = fragment_idx+1
                next_iter_threads.append(new_thread)
        return next_iter_threads

            
    @classmethod
    def final_step(cls, dinc_job_threads: List, fragment_idx):
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
            #print(t.frag_index)
            #print(t.conformations)
            #print(r.replica)
            #print(t.receptor_name)
            #print(t.results_pdbqt)
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
        all_results = all_results.sort_values(by = ["energies", "clust_size_rank", "clust_energy_rank"]).reset_index(drop=True)
    
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
            #print(rec_res)
            if len(rec_res) > replica_id:
                selected_res = rec_res.iloc[replica_id]
            else:
                selected_res = rec_res.iloc[0]
            #print(i)
            #print(selected_res)
            top_i_conf_thr =  int(selected_res.thread_id)
            top_i_conf_model =  int(selected_res.model_id)
            top_i_conf = dinc_job_threads[top_i_conf_thr].conformations[top_i_conf_model]
            new_coords = []
            cur_fragment = t.fragment.fragments[-1]
            for a in cur_fragment.allAtoms:
                coord = top_i_conf.mol.allAtoms.get(lambda x: x.name == a.name).coords[0]
                new_coords.append(coord)
            t.fragment.init_conformation(-1, coords = new_coords)
            t.fragment.activate_all_bonds()
            t.thread_elem.data.iterative_step += 1
            bbox_d = None
            bbox_c = None
            new_thread = DINCDockThreadVina(t.thread_elem, 
                                            t.params,
                                            frag_index=frag_idx,
                                            bbox_dims=bbox_d,
                                            bbox_center=bbox_c)
            next_iter_threads.append(new_thread)
        return next_iter_threads

    #def join(self):
    #    super.join()
    #    if self.exception:
    #        raise self.exception



