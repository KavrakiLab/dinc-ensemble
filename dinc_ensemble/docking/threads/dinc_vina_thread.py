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

from sklearn.cluster import DBSCAN
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
        if self.frag_index < len(self.fragment.fragments):
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
                if self.frag_index == len(self.fragment.fragments):
                    self.bbox_dims = [self.bbox_dims[0]+2,
                                      self.bbox_dims[1]+2,
                                      self.bbox_dims[2]+2]
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
        
    
    def write_results_load_conf(self, update_fragment=True):
        #1 - write poses                                                                                    
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
        if self.frag_index < len(self.fragment.fragments) and self.results_pdbqt.exists():
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
        elif self.frag_index < len(self.fragment.fragments):
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
        top_results = all_results.groupby(["replica_id"]).head(10).reset_index()
        
        '''
        # --- novelty - bounding box generation ----
        top_conformations = []
        for i, row in top_results.iterrows():
            top_i_conf_thr =  int(row.thread_id)
            top_i_conf_model =  int(row.model_id)
            top_i_conf = dinc_job_threads[top_i_conf_thr].conformations[top_i_conf_model]
            top_conformations.extend(top_i_conf.coords)
        # do DBSCAN on this data?
        data_array = np.array(top_conformations)
        n_points = max(10, round(len(top_conformations)*0.2))
        clustering = DBSCAN(eps=2, min_samples=n_points).fit(data_array)
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print("Estimated number of clusters: %d" % n_clusters_)
        print("Estimated number of noise points: %d" % n_noise_)
        bbox_cnt = n_clusters_
        bbox_dims = []
        bbox_centers = []
        for cluster_label in range(n_clusters_):
            cluster_data = data_array[clustering.labels_==cluster_label]
            # get the bounding box
            x_min, x_max = min(cluster_data[:, 0]), max(cluster_data[:, 0])
            x_d = x_max-x_min
            x_c = np.mean(cluster_data[:, 0])
            y_min, y_max = min(cluster_data[:, 1]), max(cluster_data[:, 1])
            y_d = y_max-y_min
            y_c = np.mean(cluster_data[:, 1])
            z_min, z_max = min(cluster_data[:, 2]), max(cluster_data[:, 2])
            z_d = z_max-z_min
            z_c = np.mean(cluster_data[:, 2])
            print("In cluster x {} points".format(cluster_data.shape[0]))
            print("Box of dimensions: x={} y={} z={}".format(x_d, y_d, z_d))
            bbox_dims.append([x_d, y_d, z_d])
            bbox_centers.append([x_c, y_c, z_c])
            print("Box with center: x={} y={} z={}".format(x_c, y_c, z_c))
        '''
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
            #print(top_i_conf.coords)
            #print(t.fragment.fragments[fragment_idx].pdbqt_str)
            '''
            # if this is the final round - do redocking with all active bonds!
            print("here")
            print(fragment_idx)
            print(len(t.fragment.fragments))
            if fragment_idx+1 >= len(t.fragment.fragments):
                print("here")
                cur_frag = t.fragment.fragments[-1]
                new_coords = []
                for a in cur_frag.allAtoms:
                    new_coords.append(top_i_conf.mol.allAtoms.get(lambda x: x.name==a.name).coords[0])
                print(new_coords)
                print(top_i_conf.coords)
                t.fragment.init_conformation(-1, coords = top_i_conf.coords)
                t.fragment.activate_all_bonds()
                np_conf = np.array(cur_frag.allAtoms.coords)
                frag_dims = [abs(max(np_conf[:, 0]) - min(np_conf[:, 0]))+2,
                            abs(max(np_conf[:, 1]) - min(np_conf[:, 1]))+2,
                            abs(max(np_conf[:, 2]) - min(np_conf[:, 2]))+2]
                frag_centers = [
                    min(np_conf[:, 0])+frag_dims[0]/2,
                    min(np_conf[:, 1])+frag_dims[1]/2,
                    min(np_conf[:, 2])+frag_dims[2]/2
                ]
                bbox_d = frag_dims
                bbox_c = frag_centers
                t.thread_elem.data.iterative_step += 1
                new_thread = DINCDockThreadVina(t.thread_elem, 
                                                t.params,
                                                frag_index=fragment_idx,
                                                bbox_dims=bbox_d,
                                                bbox_center=bbox_c)
                
                next_iter_threads.append(new_thread)
            '''
            if fragment_idx+1 < len(t.fragment.fragments):
                
                t.fragment.init_conformation(fragment_idx, coords = top_i_conf.coords)
                cur_frag = t.fragment.fragments[fragment_idx]
                next_frag = t.fragment.fragments[fragment_idx+1]
                # check if the ligand is in the box!
                t.fragment.expand_conformation(cur_frag.conf, fragment_idx)

                '''
                bbox_c = None
                bbox_d = None
                np_conf = np.array(next_frag.allAtoms.coords)
                frag_dims = [max(np_conf[:, 0]) - min(np_conf[:, 0]),
                            max(np_conf[:, 1]) - min(np_conf[:, 1]),
                            max(np_conf[:, 2]) - min(np_conf[:, 2]),]
                frag_centers = [
                    np.mean(np_conf[:, 0]),
                    np.mean(np_conf[:, 1]),
                    np.mean(np_conf[:, 2])
                ]
                if bbox_cnt > 0:
                    bbox_id = t.replica % bbox_cnt
                    bbox_c = bbox_centers[bbox_id]
                    bbox_d = bbox_dims[bbox_id]
                    # extend dimensions if any dimension is too small
                    for i, dim in enumerate(bbox_d):
                        if dim < frag_dims[i]+3:
                            bbox_d[i] = frag_dims[i]+2
                else:
                    bbox_c = frag_centers,
                    bbox_d = frag_dims
                '''
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

    def join(self):
        threading.Thread.join(self)
        if self.exception:
            raise self.exception



