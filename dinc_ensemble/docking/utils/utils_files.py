
from ..dinc_job_elem import DINCRunInfo, DINCJobInfo, DINCThreadData, DINCThreadElem
from ... import load_ligand, load_receptor,\
                            DINC_CORE_PARAMS, \
                        DINC_FRAG_PARAMS
from ...ligand import DINCFragment
from ...parameters.core import DINC_DOCK_TYPE
from ...parameters.fragment import DincFragParams

from pathlib import Path
from os import path
from typing import List
from shutil import copy as shcopy
from copy import deepcopy
import logging
logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)


def prepare_run_directory(ligand_file: str, 
                           receptor_files: List[str],
                           out_dir: str) -> DINCRunInfo:
    out_dir_path = Path(out_dir)
    
    logger.info("-------------------------------------")
    logger.info("DINC-Ensemble: Preparing the file structure for jobs")
    logger.info("-------------------------------------")

    out_dir_path_ensemble = out_dir_path / "ensemble"
    out_dir_path_ensemble.mkdir(exist_ok=True, parents=True)
    # copy receptors
    rec_new_paths = []
    for rec_file in receptor_files:
        new_rec_file = check_and_copy_file(rec_file, out_dir_path_ensemble)
        rec_new_paths.append(new_rec_file)
        
    # copy ligand
    out_dir_path_ligand= out_dir_path / "ligand"

    logger.info("Copying ligand to {}".format(out_dir_path_ligand))
    out_dir_path_ligand.mkdir(exist_ok=True, parents=True)
    lig_new_path = check_and_copy_file(ligand_file, out_dir_path_ligand)
    

    out_dir_path_analysis = out_dir_path / "analysis"
    out_dir_path_analysis.mkdir(exist_ok=True, parents=True)

    logger.info("Run root directory: {}".format(out_dir_path))
    logger.info("Run ligand directory: {}".format(out_dir_path_ligand))
    logger.info("Ensemble/Receptor directory: {}".format(out_dir_path_ensemble))
    logger.info("Run analysis directory: {}".format(out_dir_path_analysis))
    
    return DINCRunInfo(out_dir_path, lig_new_path,
                        rec_new_paths, out_dir_path_ensemble, out_dir_path_analysis)


def check_and_copy_file(old_file: str,
                        dst_dir: Path) -> Path:
    old_file_path = Path(old_file)
    if not old_file_path.exists():
        raise FileNotFoundError("DINCEnsemble: Receptor file {} not found.".format(old_file))
    new_file = dst_dir / old_file_path.name
    shcopy(old_file, new_file)
    return new_file

def prepare_dinc_thread_elems(dinc_run_info: DINCRunInfo,
                              dinc_run_jobs: List[DINCJobInfo],
                              rep_n: int = DINC_CORE_PARAMS.replica_num,
                              dock_type: DINC_DOCK_TYPE = DINC_CORE_PARAMS.dock_type,
                              frag_params: DincFragParams = DINC_FRAG_PARAMS
                              ) -> List[DINCThreadElem]:
    dinc_run_thr_elems: List[DINCThreadElem] = []

    logger.info("-------------------------------------")
    logger.info("DINC-Ensemble: Loading thread data!")
    logger.info("-------------------------------------")
    
    ligand = load_ligand(str(dinc_run_info.ligand))
    logger.info("-------------------------------------")
    logger.info("DINC-Ensemble: Loaded ligand")
    logger.info("-------------------------------------")
    fragment = None
    if dock_type == DINC_DOCK_TYPE.INCREMENTAL:
        fragment = DINCFragment(ligand, DINC_FRAG_PARAMS)
        fragment._split_to_fragments_()
        # this expansion will inactivate certain bonds that should be inactive
        for i, sfrag in enumerate(fragment.split_frags[:-1]):
            sfrag._freeze_prev_bonds(fragment.split_frags[i+1])
        output_dir = str(dinc_run_info.ligand.parent)
        fragment._write_pdbqt_frags_(out_dir=output_dir)
        fragment._write_svg_frags_(out_dir=output_dir)
        info_table_fname = ligand.molkit_molecule.name + "_frag_info.html"
        frags_info_df = fragment._to_df_frags_info_()
        frags_info_df["frag_svg_path"] = frags_info_df["frag_pdbqt"].apply(lambda x: str(x).split(".")[0]+".svg") # type: ignore
        frags_info_df["frag_svg"] = frags_info_df["frag_svg_path"].apply(lambda x: '''<img src=\"./{}\"/>'''.format(x)) # type: ignore
        frags_info_df = frags_info_df[["frag_id", # type: ignore
                                        "frag_dof",
                                        "frag_pdbqt",
                                        "frag_svg"]]
        fragment.info_df = frags_info_df
        frag_info_html = frags_info_df.to_html(render_links=True,escape=False)
        with open(path.join(output_dir, info_table_fname), "w") as f:
            f.write(frag_info_html)

        logger.info("-------------------------------------")
        logger.info("DINC-Ensemble: Loaded fragment")
        logger.info("-------------------------------------")

    receptors = []
    for receptor_path in dinc_run_info.receptors:
        receptor = load_receptor(str(receptor_path))

    logger.info("-------------------------------------")
    logger.info("DINC-Ensemble: Loaded receptors")
    logger.info("-------------------------------------")
        
    for i, job in enumerate(dinc_run_jobs):
        # load thread data - one job data per replica
        dinc_thread_data = []
        for replica in range(rep_n):
            data = DINCThreadData(deepcopy(ligand), 
                                  deepcopy(fragment), 
                                  deepcopy(receptor), 
                                  replica)
            dinc_thread_data.append(data)
        elems = []
        for data in dinc_thread_data:
            elem = DINCThreadElem(data, job)
            elems.append(elem)
        dinc_run_thr_elems.extend(elems) 
    return  dinc_run_thr_elems