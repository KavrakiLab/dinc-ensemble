
from .dinc_init import init_dinc_ensemble_threads
from .threads.dinc_vina_thread import DINCDockThreadVina

from typing import List
import pandas as pd
from dinc_ensemble.parameters import *
from dinc_ensemble import write_ligand
from dinc_ensemble.ligand import DINCMolecule
from pathlib import Path

import logging
logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)


def dinc_full_run(ligand_file: str,
            receptor_files: List[str]):
    
    # initialize jobs/threads:
    # 0 - initialize directories
    # 1 - initialize receptors (align and binding box)
    # 2 - initialize ligands
    # 3 - initialize fragments
    dinc_thread_info_df, dinc_thread_elem, dinc_run_info = init_dinc_ensemble_threads(ligand_file, receptor_files)


    root_dir = dinc_run_info.root
    analysis_dir = dinc_run_info.analysis
    fragments_dir = dinc_run_info.ligand.parent

    
    fh = logging.FileHandler(str(Path(root_dir) / Path('dinc_ensemble_run.log')))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # for tracking progress:
    # 0 - scheduled
    # 1 - started
    # 2 - finished
    progress_file = root_dir / Path("progress.csv")
    progress_df = pd.read_csv(progress_file)
    

    if DINC_CORE_PARAMS.dock_type == DINC_DOCK_TYPE.INCREMENTAL:
        
        frag_cnt = 0
        if len(dinc_thread_elem) >= 1:
            frag_cnt = len(dinc_thread_elem[0].data.fragment.split_frags)
        
        # Initialize the threads for jobs
        dinc_frag_threads = []
        dinc_frag_threads_per_job = {}
        for elem in dinc_thread_elem:
            job_name = elem.job_info.job_name
            replica = elem.data.replica
            fragment = elem.data.iterative_step
            dinc_thread = DINCDockThreadVina(elem, 
                                                VINA_ENGINE_PARAMS)
            dinc_frag_threads.append(dinc_thread)
            if job_name not in dinc_frag_threads_per_job:
                dinc_frag_threads_per_job[job_name] = []
            dinc_frag_threads_per_job[job_name].append(dinc_thread)

        for job, initial_job_threads in dinc_frag_threads_per_job.items():
            current_job_threads = initial_job_threads
            # for each iteration of the fragments
            for frag_idx in range(frag_cnt):
                logger.info("-------------------------------------")
                logger.info("DINC-Ensemble: Thread for - Job #{}; Fragment{};".format(job,frag_idx))
                logger.info("-------------------------------------")
                # for all receptor threads (replicas)
                for t in current_job_threads:
                    t.start()
                # wait for one iterative round per job
                for t in current_job_threads:
                    t.join()
                dinc_frag_threads_per_job[job] = current_job_threads
                current_job_threads = DINCDockThreadVina.next_step(current_job_threads, frag_idx)
            

        logger.info("-------------------------------------")
        logger.info("DINC-Ensemble: Finished all threads - summarizing results")
        logger.info("-------------------------------------")
    
        all_results = None
        for job, job_threads in dinc_frag_threads_per_job.items():
            for t in job_threads:
                res_file = t.output_dir / Path("results_frag_{}_collection.csv".format(frag_cnt-1))
                res_df = pd.read_csv(res_file)
                res_df["job_id"] = res_df["thread_id"].apply(lambda x: job)
                if all_results is None:
                    all_results = res_df
                else:
                    all_results = pd.concat([all_results, res_df])
        all_results = all_results.sort_values(by = ["energies", "rmsds"]).reset_index(drop=True)
        all_results.to_csv(dinc_run_info.analysis / Path("results.csv"))

        logger.info("-------------------------------------")
        logger.info("DINC-Ensemble: Extracting best poses")
        logger.info("-------------------------------------")

        n_out = DINC_CORE_PARAMS.n_out
        out_results = all_results[all_results["fragment_id"]==frag_cnt-1][:n_out]
        for i, res in out_results.iterrows():
            thr_id = int(res["thread_id"])
            job_id = res["job_id"]
            model_id = int(res["model_id"])
            thr = dinc_frag_threads_per_job[job_id][thr_id]
            conf = thr.conformations[model_id]
            ligand = DINCMolecule(conf.mol, prepare=False)
            write_ligand(ligand, str(dinc_run_info.analysis / Path("result_top{}.pdb".format(i))))

    
    # DOCK TYPE CLASSIC
    # No incremental and procedure for ligands
    # Just multi receptor and multi replica
    if DINC_CORE_PARAMS.dock_type == DINC_DOCK_TYPE.CLASSIC:
        
        # start threads (per receptor docking)
        dinc_job_threads = []
        for elem in dinc_thread_elem:
            job_name = elem.job_info.job_name
            replica = elem.data.replica
            fragment = elem.data.iterative_step
            logger.info("-------------------------------------")
            logger.info("DINC-Ensemble: Thread for - Job #{}; Replica {}; Fragment{};".format(job_name, replica, fragment))
            logger.info("-------------------------------------")
            dinc_thread = DINCDockThreadVina(elem, VINA_ENGINE_PARAMS)
            dinc_job_threads.append(dinc_thread)
            dinc_thread.start()

        # wait for all jobs to finish
        for t in dinc_job_threads:
            t.join()

        logger.info("-------------------------------------")
        logger.info("DINC-Ensemble: Finished all threads - summarizing results")
        logger.info("-------------------------------------")
        
        # analyze jobs
        all_results = pd.DataFrame({"energies": [],
                                   "rmsds": [],
                                   "model_id":[],
                                   "replica_id":[],
                                   "receptor_id":[],
                                   "thread_id":[]})
        
        for i, t in enumerate(dinc_job_threads):
            # results file will have
            res = pd.read_csv(t.results_file)
            res["energies"] = res["energies"].apply(lambda x: float(x))
            res["replica_id"] = res["energies"].apply(lambda x: t.replica)
            res["receptor_id"] = res["energies"].apply(lambda x: t.receptor_name)
            res["thread_id"] = res["energies"].apply(lambda x: i)
            all_results = pd.concat([all_results, res])
        all_results = all_results.sort_values(by = ["energies", "rmsds"]).reset_index(drop=True)
        all_results.to_csv(dinc_run_info.analysis / Path("results.csv"))
        
        logger.info("-------------------------------------")
        logger.info("DINC-Ensemble: Extracting best poses")
        logger.info("-------------------------------------")

        n_out = DINC_CORE_PARAMS.n_out
        out_results = all_results[:n_out]
        for i, res in out_results.iterrows():
            thr_id = int(res["thread_id"])
            model_id = int(res["model_id"])
            thr = dinc_job_threads[thr_id]
            conf = thr.conformations[model_id]
            ligand = DINCMolecule(conf.mol, prepare=False)
            write_ligand(ligand, str(dinc_run_info.analysis / Path("result_top{}.pdb".format(i))))

    logger.info("-------------------------------------")
    logger.info("DINC-Ensemble: Finished running DINC-Ensemble!")
    logger.info("DINC-Ensemble: See the results here: {}".format(dinc_run_info.analysis))
    logger.info("-------------------------------------")