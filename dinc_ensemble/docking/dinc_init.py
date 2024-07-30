'''
Here are some functionalities needed to set up the basic docking!
Given as input a list of ligands and a list of receptors.
As well as DINC_PARAMS that are represented as global and can be changed at some entry points such as scripts.

1 - load ligands
2 - prepare ligands
3 - fragment ligands
----
4 - load + prepare receptors
----
5 - prepare the binding boxes
----
6 - see all the combinations that will be scheduled for docking
7 - compute grids / maps where possible for docking
8 - schedule docking
'''
from typing import List
import pandas as pd
from pathlib import Path
import json
from dataclasses import asdict

from .dinc_job_elem import DINCRunInfo, DINCJobInfo, DINCThreadElem
from ..parameters import DINC_CORE_PARAMS, DINC_RECEPTOR_PARAMS, DINC_ANALYSIS_PARAMS, DINC_FRAG_PARAMS, DINC_DOCK_TYPE
from .utils.utils_files import prepare_run_directory, prepare_dinc_thread_elems
from .utils.utils_prep_receptor import align_receptors, prepare_bbox

def init_dinc_ensemble_threads(ligand_file: str,
                       receptor_files: List[str]):
    
    # STEP 0 - copy all files to the correct locations
    dinc_run_info:DINCRunInfo = prepare_run_directory(ligand_file, 
                                                    receptor_files,
                                                    DINC_CORE_PARAMS.output_dir)

    # STEP 1 - align receptors if needed
    if DINC_RECEPTOR_PARAMS.align_receptors and len(dinc_run_info.receptors) > 1:
        aligned_rec = align_receptors(dinc_run_info.receptors, dinc_run_info.ensemble)
        dinc_run_info.receptors = aligned_rec

    # STEP 2 - get the binding box
    box_center, box_dims = prepare_bbox(dinc_run_info.ligand, 
                 dinc_run_info.receptors,
                 DINC_RECEPTOR_PARAMS)
    
    # STEP 3 - load and prepare information for all the jobs
    # DINCRunInfo - info for the full DINC run (1 ligand, multiple receptors)
    # DINCJobInfo - info for one DINC job (1 ligand, 1 receptor)
    # DINCThreadData - info for one DINC job (1 ligand, 1 receptor, 1 replica, 1 fragment at a time)
    # DINCThreadElem - (DINCJobPaths, DINCThreadData)
    dinc_run_jobs: List[DINCJobInfo] = dinc_run_info.create_jobs()
    dinc_thread_elems: List[DINCThreadElem] = prepare_dinc_thread_elems(dinc_run_info,
                                                                        dinc_run_jobs,
                                                                        rep_n=DINC_CORE_PARAMS.replica_num,
                                                                        dock_type=DINC_CORE_PARAMS.dock_type)
    dinc_run_summary = pd.concat([e.info for e in dinc_thread_elems]).reset_index(drop=True)

    # STEP 4 - for tracking progress save all the parameters and progress in a file
    dinc_run_summary["progress"] = dinc_run_summary["fragment_id"]
    progress_file = dinc_run_info.root / Path("progress.csv")

    if not progress_file.exists():
        dinc_run_summary.to_csv(progress_file)
        with open(dinc_run_info.root / Path("core_params.json"), "w") as f:
            json.dump(asdict(DINC_CORE_PARAMS), f)
        with open(dinc_run_info.root / Path("rec_params.json"), "w") as f:
            json.dump(asdict(DINC_RECEPTOR_PARAMS), f)
        with open(dinc_run_info.root / Path("analysis_params.json"), "w") as f:
            json.dump(asdict(DINC_ANALYSIS_PARAMS), f)
        if DINC_CORE_PARAMS.dock_type == DINC_DOCK_TYPE.INCREMENTAL:
            with open(dinc_run_info.root / Path("frag_params.json"), "w") as f:
                json.dump(asdict(DINC_FRAG_PARAMS), f)
    return dinc_run_summary, dinc_thread_elems, dinc_run_info






         

