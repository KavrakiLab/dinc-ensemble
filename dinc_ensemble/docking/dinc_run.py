from typing import List
import pandas as pd
import time
from dinc_ensemble import write_ligand
from dinc_ensemble.ligand import DINCMolecule
import multiprocessing as mp
from pathlib import Path

from .dinc_dock import dinc_run_single_vina
from .utils import TEMPLATE_RES_CROSSDOCK, TEMPLATE_RES_REDOCK, TEMPLATE_NO_RES
from .utils.utils_process_output import cluster_conformations
from ..parameters import DINC_CORE_PARAMS, DINC_JOB_TYPE, VINA_ENGINE_PARAMS
from .utils.utils_files import dinc_prepare_inputs, create_task_dir, generate_ligand_task_fname
from .utils.utils_process_output import extract_vina_conformations
from .utils.utils_loggers import init_logging
from ..analysis.rmsd import calculate_rmsd 

import os
from copy import deepcopy
import logging
logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)

        
def dinc_full_run(ligand_file: str,
            receptor_files: List[str]):
    
    # STEP 0.0 - create the root directory for the work
    Path(DINC_CORE_PARAMS.output_dir).mkdir(exist_ok=True, parents=True)

    # STEP 0.1 - initialize logging and start time
    init_logging(verbosity=DINC_CORE_PARAMS.verbose, 
                 out_dir=DINC_CORE_PARAMS.output_dir)    
    start_time = time.time()
    
    # STEP 1.0 - initialize jobs/threads:
    # 0 - initialize directories
    # 1 - initialize receptors (align and binding box)
    # 2 - initialize ligands (load to DINCMolecule)
    # 3 - initialize fragments (load to DINCFragment - used for expanding conformations inbetweek runs)
    dinc_run_info, ligand, fragment, frag_files, receptors = dinc_prepare_inputs(ligand_file, receptor_files)
    # some useful directories here
    root_dir = dinc_run_info.root
    
    # STEP 1.1 - initialize the task dataframe:
    # receptors
    # fragments 
    # replicas
    # for each combo we call the docking engine
    frag_df = pd.DataFrame({"fragment": [i for i, frag in enumerate(fragment.fragments)]})
    replica_df = pd.DataFrame({"replica": [i for i in range(DINC_CORE_PARAMS.replica_num)]})
    receptor_df = pd.DataFrame({"receptor": [rec for rec in dinc_run_info.receptors]})
    tasks_df = frag_df.merge(replica_df, how="cross")
    tasks_df = tasks_df.merge(receptor_df, how="cross")

    # STEP 1.2 - generate other data for the tasks:
    # outdir (per task for intermediate files)
    # ligand (names per task)
    # rec_name (shorter receptor name / not full path)
    # status (progress info)
    # outfile (file with docking results)
    # - save the info to a progress csv
    tasks_df["outdir"] = tasks_df["receptor"].apply(lambda x: create_task_dir(x, root_dir))
    tasks_df["ligand"] = tasks_df.apply(lambda x: generate_ligand_task_fname(x, frag_files), axis=1)
    tasks_df["receptor_name"] = tasks_df["receptor"].apply(lambda x: Path(x).stem)
    tasks_df["status"] = tasks_df.apply(lambda x: "initialized", axis=1)
    tasks_df["outfile"] = tasks_df.apply(lambda x: x.outdir / "frag_{}_rep_{}_out.pdbqt".format(
                                                    x.fragment, x.replica), axis=1)
    
    tasks_df[["receptor_name", "fragment", "replica", "ligand", "status"]].to_csv(root_dir / "progress.csv")
    

    logger.info("---------------------------------")
    logger.info("Starting DINC-Ensemble docking!")
    logger.info("{} receptors; {} replicas; {} fragments/rounds".format(len(receptor_files), 
                                                                        DINC_CORE_PARAMS.replica_num,
                                                                        len(fragment.fragments)))
    logger.info("---------------------------------")
    n_processes = None
    if DINC_CORE_PARAMS.cpu_count != 0:
        n_processes = DINC_CORE_PARAMS.cpu_count
    # STEP 2 - start docking the tasks
    for frag_iter, frag in enumerate(fragment.fragments):
        task_subset = tasks_df[tasks_df["fragment"] == frag_iter]
        # STEP 2.1 - dock
        logger.info("- Docking round {}".format(frag_iter))
        
        tasks_df["status"] = tasks_df.apply(lambda x: "started" if x.fragment ==frag_iter
                                                     else x.status, axis=1)
        tasks_df[["receptor_name", "fragment", "replica", "ligand", "status"]].to_csv(root_dir / "progress.csv")
        
        with mp.Pool(processes=n_processes) as pool:
            pool_data = list(task_subset.apply(
                lambda row:
                (row.ligand, 
                row.receptor, 
                row.outfile,
                (row.fragment==0),
                dinc_run_info.bbox_file
                ), axis=1))
            pool.starmap(dinc_run_single_vina, pool_data)
        
        # STEP 2.2 - aggregate results
        logger.info("- Aggregating results round {}".format(frag_iter))
        with mp.Pool(processes=n_processes, 
                     initializer = init_postprocess, 
                     initargs=(ligand, fragment)) as pool:
            task_data = list(task_subset.apply(lambda row: (frag_iter,
                                                        DINC_CORE_PARAMS.replica_num,
                                                        row.receptor,
                                                        root_dir
                                                        ), axis=1))
            pool.starmap(dinc_complete_single_dock_round, task_data)

        # update status
        tasks_df["status"] = tasks_df.apply(lambda x: "docked" if x.fragment ==frag_iter
                                                     else x.status, axis=1)
        tasks_df[["receptor_name", "fragment", "replica", "ligand", "status"]].to_csv(root_dir / "progress.csv")
    
    
    end_time = time.time()
    # STEP 3 - Final processing of the results
    logger.info("- Processing final results")
    results = final_processing_results(
        task_directories=list(tasks_df["outdir"].unique()),
        receptor_names=list(tasks_df["receptor_name"].unique()),
        fragment_cnt=len(fragment.fragments),
        replica_cnt=DINC_CORE_PARAMS.replica_num,
        original_ligand=ligand,
        output_dir=dinc_run_info.analysis
    )
    
    if results[0] is not None:
        if DINC_CORE_PARAMS.job_type == DINC_JOB_TYPE.REDOCK:
            final_res_txt = TEMPLATE_RES_REDOCK
            energy_conf_txt = results[0]
            clust_conf_txt = results[1]
            rmsd_conf_txt = results[2]
            final_res_txt = final_res_txt.format(LIGAND = ligand_file,
                                                RECEPTORS = "\n".join(receptor_files),
                                                ENERGY_CONF_TXT = energy_conf_txt,
                                                CLUST_CONF_TXT = clust_conf_txt,
                                                RMSD_CONF_TXT = rmsd_conf_txt,
                                                RUNTIME = end_time - start_time)
        else:
            final_res_txt = TEMPLATE_RES_CROSSDOCK
            energy_conf_txt = results[0]
            clust_conf_txt = results[1]
            final_res_txt = final_res_txt.format(LIGAND = ligand_file,
                                                RECEPTORS = "\n".join(receptor_files),
                                                ENERGY_CONF_TXT = energy_conf_txt,
                                                CLUST_CONF_TXT = clust_conf_txt,
                                                RUNTIME = end_time - start_time)
    else:
        final_res_txt = TEMPLATE_NO_RES
        final_res_txt = final_res_txt.format(LIGAND = ligand_file,
                                            RECEPTORS = "\n".join(receptor_files),
                                            RUNTIME = end_time - start_time)

    with open(dinc_run_info.analysis / "dincens_results.txt", "w") as f:
        f.write(final_res_txt)
    logger.info("-------------------------------------")
    logger.info("DINC-Ensemble: Finished running DINC-Ensemble!")
    logger.info("Total running time: {}s".format(end_time - start_time))
    logger.info("DINC-Ensemble: See the results here: {}".format(dinc_run_info.analysis))
    logger.info("-------------------------------------")
    
# this function is for aggregating results after docking runs
# has to be here because of share ligand/fragment
def dinc_complete_single_dock_round(frag_idx: int,
                                    replica_cnt: int,
                                    receptor: str, 
                                    root_dir: str) -> str:
    
    # if we are in continue mode, only need to do all of this if collection does not exist
    task_dir = Path(root_dir) / "{}_out".format(Path(receptor).stem)
    output_collection_file = task_dir / "collection_frag_{}.csv".format(frag_idx)
    if DINC_CORE_PARAMS.continue_run and output_collection_file.exists():
        return output_collection_file
    
    # STEP 1 - Initialize all the inputs
    original_ligand = shared_ligand
    fragment = deepcopy(shared_frag)
    ref_ligand = original_ligand.molkit_molecule
    output_pdbqts = ["{}/frag_{}_rep_{}_out.pdbqt".format(
        task_dir, frag_idx, r
    ) for r in range(replica_cnt)]

    # STEP 2 - load all conformations; extract energies
    all_confs = []
    replicas = []
    models = []
    energies = []
    for out_pdbqt in output_pdbqts:
        if Path(out_pdbqt).exists():
            conformations = extract_vina_conformations(out_pdbqt,
                                                    original_ligand.molkit_molecule)
            all_confs.extend(conformations)
            replicas.extend([out_pdbqt for i in conformations])
            models.extend([i for i, c in enumerate(conformations)])
            energies.extend([c.binding_energy for c in conformations])
    df_results = pd.DataFrame({"energy": energies,
                            "model":models,
                            "replica":replicas,
                            "confs":all_confs})
    df_results = df_results.sort_values("energy").reset_index()
    
    # STEP 3 calculate the rmsds
    if df_results.shape[0] > 0:
        if DINC_CORE_PARAMS.dock_type == DINC_JOB_TYPE.CROSSDOCK:
            ref_ligand = df_results["confs"].iloc[0].molecule
    df_results["rmsd"] = df_results["confs"].apply(lambda x: calculate_rmsd(x, ref_ligand))
    # STEP 4 cluster conformations and extract cluster info 
    cluster_conformations(all_confs)
    df_results["clust_nrg_rank"] = df_results["confs"].apply(lambda x: x.clust_nrg_rank)
    df_results["clust_size_rank"] = df_results["confs"].apply(lambda x: x.clust_size_rank)
    
    # STEP 5 prepare fragment for next iterations
    if frag_idx + 1 < len(fragment.fragments):
        if len(df_results) > 0:
            # if there were outputs, use them to seed
            clus_df = df_results.sort_values(["clust_nrg_rank", "clust_size_rank"]).groupby("clust_size_rank").head(1).reset_index()
            for rep in range(replica_cnt):
                if len(clus_df) > rep:
                    selected_res = clus_df.iloc[rep]
                else:
                    selected_res = clus_df.iloc[0]
                fragment.init_conformation(frag_idx, coords = selected_res.confs.coords)
                cur_frag = fragment.fragments[frag_idx]
                fragment.expand_conformation(cur_frag.conf, frag_idx)
                out_pdbqt_file = task_dir / ("frag_{}_rep_{}.pdbqt".format(frag_idx+1,
                                                                            rep))
                with open(out_pdbqt_file, "w") as f:
                    pdbqt_val = fragment.fragments_dincmol[frag_idx+1].molkit_molecule.pdbqt_str
                    f.write(pdbqt_val)
        else:
        # if there was no outputs use the initial fragment
            for rep in range(replica_cnt):
                out_pdbqt_file = task_dir / ("frag_{}_rep_{}.pdbqt".format(frag_idx+1,
                                                                            rep))
                with open(out_pdbqt_file, "w") as f:
                    pdbqt_val = fragment.fragments_dincmol[frag_idx+1].molkit_molecule.pdbqt_str
                    f.write(pdbqt_val)
            
    # STEP 6 save metadata to file and return it
    df_results = df_results[["energy", "rmsd", "clust_size_rank", "clust_nrg_rank", "replica", "model"]]
    df_results.to_csv(output_collection_file)
    return output_collection_file

def final_processing_results(task_directories, 
                             receptor_names,
                             fragment_cnt, 
                             replica_cnt,
                             original_ligand,
                             output_dir):
    # load all the results from the task output directories to a single 
    # get all task outputs

    all_results_df = None
    last_frag_idx = fragment_cnt - 1
    # STEP 1 - load all final conformations from tasks
    all_confs = []
    receptors = []
    replicas = []
    models = []
    energies = []
    for i, task_dir in enumerate(task_directories):
        output_pdbqts = ["{}/frag_{}_rep_{}_out.pdbqt".format(
            task_dir, last_frag_idx, r
        ) for r in range(replica_cnt)]
        # STEP 1.1 - load all conformations per task; extract energies
        for out_pdbqt in output_pdbqts:
            if Path(out_pdbqt).exists():
                conformations = extract_vina_conformations(out_pdbqt,
                                                        original_ligand.molkit_molecule)
                all_confs.extend(conformations)
                replicas.extend([out_pdbqt for i in conformations])
                models.extend([i for i, c in enumerate(conformations)])
                energies.extend([c.binding_energy for c in conformations])
                receptors.extend([receptor_names[i] for c in conformations])

    # STEP 2 - gather all this data and cluster
    df_results = pd.DataFrame({"energy": energies,
                            "model":models,
                            "receptor":receptors,
                            "replica":replicas,
                            "confs":all_confs})
    cluster_conformations(all_confs)
    df_results["clust_nrg_rank"] = df_results["confs"].apply(lambda x: x.clust_nrg_rank)
    df_results["clust_size_rank"] = df_results["confs"].apply(lambda x: x.clust_size_rank)
    df_results = df_results.sort_values("energy").reset_index(drop=True)
    df_results = df_results.drop_duplicates(["clust_size_rank"]).reset_index(drop=True)
   
    # STEP 3 calculate the rmsds
    ref_ligand = original_ligand.molkit_molecule
    if df_results.shape[0] > 0:
        if DINC_CORE_PARAMS.dock_type == DINC_JOB_TYPE.CROSSDOCK:
            ref_ligand = df_results["confs"].iloc[0].molecule
        df_results["rmsd"] = df_results["confs"].apply(lambda x: calculate_rmsd(x, ref_ligand))
        # STEP 4 extract best poses and save them 
        n_out = min(DINC_CORE_PARAMS.n_out, len(df_results))
        out_results = df_results[:n_out].reset_index()
        energy_conf_txt = ""
        for i, res in out_results.iterrows():
            conf = res["confs"]
            ligand = DINCMolecule(conf.mol, prepare=False)
            outfile = str(output_dir / Path("top_energy_{}.pdb".format(i)))
            write_ligand(ligand, outfile)
            energy_conf_txt+="         {}            {} (kcal/mol)       {} (Å)       {}\n".format("top_energy_{}.pdb".format(i), 
                                                                                       round(res["energy"], 2),
                                                                                       round(res["rmsd"], 2),
                                                                                       res["receptor"])
        
        # STEP 5 extract output info 
        tmp_results = df_results.sort_values(["clust_size_rank", "energy"])
        out_results = tmp_results[:n_out].reset_index()
        clust_conf_txt = ""
        for i, res in out_results.iterrows():
            conf = res["confs"]
            ligand = DINCMolecule(conf.mol, prepare=False)
            write_ligand(ligand, str(output_dir / Path("top_cluster_{}.pdb".format(i))))
            clust_conf_txt+="         {}            {} (kcal/mol)       {} (Å)       {}\n".format("top_cluster_{}.pdb".format(i), 
                                                                                       round(res["energy"], 2),
                                                                                       round(res["rmsd"], 2),
                                                                                       res["receptor"])
        
        rmsd_conf_txt = None
        if DINC_CORE_PARAMS.job_type == DINC_JOB_TYPE.REDOCK:
            tmp_results = df_results.sort_values(["rmsd", "energy"])
            out_results = tmp_results[:n_out].reset_index()
            rmsd_conf_txt = ""
            for i, res in out_results.iterrows():
                conf = res["confs"]
                ligand = DINCMolecule(conf.mol, prepare=False)
                write_ligand(ligand, str(output_dir / Path("top_rmsd_{}.pdb".format(i))))
                rmsd_conf_txt+="         {}            {} (kcal/mol)       {} (Å)       {}\n".format("top_rmsd_{}.pdb".format(i), 
                                                                                        round(res["energy"], 2),
                                                                                        round(res["rmsd"], 2),
                                                                                        res["receptor"])
        df_results_out = df_results[["energy", "rmsd", "clust_size_rank", "receptor"]]
        df_results_out["energy"] = df_results_out["energy"].apply(lambda x: round(x, 2))
        df_results_out["rmsd"] = df_results_out["rmsd"].apply(lambda x: round(x, 2))
        df_results_out.columns = ["Energy (kcal/mol)", "RMSD", "Cluster size rank", "Receptor Name"]
        df_results_out.to_csv(str(output_dir / Path("results.csv")))
        df_results.to_csv(str(output_dir / Path("all_info_results.csv")))
        return (energy_conf_txt, clust_conf_txt, rmsd_conf_txt)
    else:
        logger.info("No poses generated!!!")
        return (None, None, None)


# this we need to be able to share some data
def init_postprocess(ligand, frag):
    global shared_ligand
    global shared_frag
    shared_ligand = ligand
    shared_frag = frag