
from ... import load_ligand, load_receptor,\
                            DINC_CORE_PARAMS, \
                        DINC_FRAG_PARAMS
from ...ligand import DINCFragment
from ...parameters.core import DINC_DOCK_TYPE
from ...parameters.fragment import DINC_FRAGMENT_MODE
from dinc_ensemble import DINC_CORE_PARAMS, \
                          DINC_RECEPTOR_PARAMS, \
                          VINA_ENGINE_PARAMS
from .utils_prep_receptor import align_receptors, prepare_bbox

from pathlib import Path
from shutil import copy as shcopy
from dataclasses import asdict, dataclass
import json
from typing import List
import os

import logging
logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)


@dataclass
class DINCRunInfo:
    # DINC Run: one full run of DINC
    # 1 ligand, N receptors
    root: Path
    ligand: Path
    receptors: List[Path]
    ensemble: Path
    bbox_file: Path
    analysis: Path


def dinc_prepare_inputs(ligand_file, receptor_files):
    # STEP 0 - copy all files to the correct locations
    # ligand info is in: dinc_run_info.ligand
    # receptors info is in: dinc_run_info.receptors
    
    logger.info("-------------------------------------")
    logger.info("STEP 1 - prepare files and inputs!")
    logger.info("-------------------------------------")
    dinc_run_info:DINCRunInfo = prepare_run_directory(ligand_file, 
                                                    receptor_files,
                                                    DINC_CORE_PARAMS.output_dir)
    # STEP 1 - align receptors if needed
    # this may update receptors info is: dinc_run_info.receptors
    if DINC_RECEPTOR_PARAMS.align_receptors and len(dinc_run_info.receptors) > 1:
        
        logger.info("- Aligning receptors")
        aligned_rec = align_receptors(dinc_run_info.receptors, dinc_run_info.ensemble)
        dinc_run_info.receptors = aligned_rec   
        
    # STEP 2 - get the binding box
    # this info is in: dinc_run_info.bbox_file
    logger.info("- Calculating binding box")
    bbox_file = prepare_bbox(dinc_run_info.ligand, 
                 dinc_run_info.receptors,
                 dinc_run_info.bbox_file,
                 DINC_RECEPTOR_PARAMS)
    
    # STEP 3 - load ligand
    # this info is from: dinc_run_info.bbox_file
    logger.info("- Loading ligand")
    
    ligand = load_ligand(str(dinc_run_info.ligand))
    
    # STEP 4 - generate fragments, freeze bonds
    # pdbqt strings will be written to : dinc_run_info.ligand.parent
    logger.info("- Generate fragments")
    # if doing classic docking, all bonds are active
    frag_params = DINC_FRAG_PARAMS
    print(frag_params)
    if DINC_CORE_PARAMS.dock_type == DINC_DOCK_TYPE.CLASSIC:
        frag_params.frag_mode = DINC_FRAGMENT_MODE.MANUAL
        frag_params.frag_size = len(ligand.bonds)
    fragment = DINCFragment(ligand, frag_params)
    output_dir = str(dinc_run_info.ligand.parent)
    frag_out_df = fragment.write_pdbqt_frags(out_dir=output_dir)
    frag_files = list(frag_out_df["frag_pdbqt_file"])
    
    # STEP 5 - load receptors
    # receptors loaded from: dinc_run_info.receptors
    logger.info("- Load receptors")
    receptors = []
    receptor_pdbqt_paths = []
    for receptor_path in dinc_run_info.receptors:
        dince_rec = load_receptor(str(receptor_path))
        # save the pdbqt version
        rec_name = receptor_path.stem
        dst_rec = dinc_run_info.ensemble / (rec_name+".pdbqt")
        with open(dst_rec, "w") as f:
            f.write(dince_rec.molkit_receptor.pdbqt_str)
        receptor_pdbqt_paths.append(dst_rec)
        receptors.append(dince_rec)
    dinc_run_info.receptors = receptor_pdbqt_paths
    logger.info("Finished preparing files and inputs!")
    logger.info("-------------------------------------")
    return dinc_run_info, ligand, fragment, frag_files, receptors



def prepare_run_directory(ligand_file: str, 
                           receptor_files: List[str],
                           out_dir: str) -> DINCRunInfo:
    out_dir_path = Path(out_dir)
    
    logger.info("Preparing the file structure for jobs:")

    out_dir_path_ensemble = out_dir_path / "ensemble"
    out_dir_path_ensemble.mkdir(exist_ok=True, parents=True)
    # copy receptors
    rec_new_paths = []
    for rec_file in receptor_files:
        new_rec_file = check_and_copy_file(rec_file, out_dir_path_ensemble)
        rec_new_paths.append(new_rec_file)
        
    # copy ligand
    out_dir_path_ligand= out_dir_path / "ligand"

    logger.info("- Copying ligand to {}".format(out_dir_path_ligand))
    out_dir_path_ligand.mkdir(exist_ok=True, parents=True)
    lig_new_path = check_and_copy_file(ligand_file, out_dir_path_ligand)
    

    out_dir_path_analysis = out_dir_path / "analysis"
    out_dir_path_analysis.mkdir(exist_ok=True, parents=True)

    logger.info("- Run root directory: {}".format(out_dir_path))
    logger.info("- Run ligand directory: {}".format(out_dir_path_ligand))
    logger.info("- Ensemble/Receptor directory: {}".format(out_dir_path_ensemble))
    logger.info("- Run analysis directory: {}".format(out_dir_path_analysis))
    
    # save the docking parameters to json files
    with open(out_dir_path / Path("core_params.json"), "w") as f:
        json.dump(asdict(DINC_CORE_PARAMS), f)
    with open(out_dir_path / Path("rec_params.json"), "w") as f:
        json.dump(asdict(DINC_RECEPTOR_PARAMS), f)
    with open(out_dir_path / Path("vina_params.json"), "w") as f:
        json.dump(asdict(VINA_ENGINE_PARAMS), f)
    if DINC_CORE_PARAMS.dock_type == DINC_DOCK_TYPE.INCREMENTAL:
        with open(out_dir_path / Path("frag_params.json"), "w") as f:
            json.dump(asdict(DINC_FRAG_PARAMS), f)

    return DINCRunInfo(out_dir_path, lig_new_path,
                        rec_new_paths, out_dir_path_ensemble, out_dir_path_ensemble / "bbox.txt",
                        out_dir_path_analysis)


def check_and_copy_file(old_file: str,
                        dst_dir: Path) -> Path:
    old_file_path = Path(old_file)
    if not old_file_path.exists():
        raise FileNotFoundError("DINCEnsemble: Receptor file {} not found.".format(old_file))
    new_file = dst_dir / old_file_path.name
    shcopy(old_file, new_file)
    return new_file

    
def create_task_dir(receptor, root_dir):
    rec_name = Path(receptor).stem
    task_dir = root_dir / "{}_out/".format(rec_name)
    if not task_dir.exists():
        task_dir.mkdir(exist_ok=True, parents=True)
    return task_dir       

def generate_ligand_task_fname(row, frag_files):
    root = row.outdir
    dst_frag = root / "frag_{}_rep_{}.pdbqt".format(row.fragment, 
                                                    row.replica)
    if row.fragment == 0:
        source_frag = frag_files[0]
        os.system("cp {} {}".format(source_frag, dst_frag))
    return dst_frag 
