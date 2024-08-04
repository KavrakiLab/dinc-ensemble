
from dinc_ensemble.parameters.analysis import *
from dinc_ensemble.parameters.fragment import *
from dinc_ensemble.parameters.core import *
from dinc_ensemble.parameters import *
from dinc_ensemble.receptor import *
from dinc_ensemble.parameters.dock_engine_vina import *

def init_all_dince_params(**kwargs):

    if "output_dir" in kwargs:
        DINC_CORE_PARAMS.output_dir = kwargs["output_dir"]
    if "job_type" in kwargs:
        DINC_CORE_PARAMS.job_type = kwargs["job_type"]
    if "dock_type" in kwargs:
        DINC_CORE_PARAMS.dock_type = kwargs["dock_type"]
    if "dock_engine" in kwargs:
        DINC_CORE_PARAMS.dock_engine = kwargs["dock_engine"]
    if "replica_num" in kwargs:
        DINC_CORE_PARAMS.replica_num = kwargs["replica_num"]
    if "output_dir" in kwargs:
        DINC_CORE_PARAMS.output_dir = str(kwargs["output_dir"])
    if "n_out" in kwargs:
        DINC_CORE_PARAMS.n_out = kwargs["n_out"]
    if "continue" in kwargs:
        DINC_CORE_PARAMS.continue_run = kwargs["continue"]
    if "cpu_count" in kwargs:
        DINC_CORE_PARAMS.cpu_count = int(kwargs["cpu_count"])
    if "verbose" in kwargs:
        DINC_CORE_PARAMS.verbose = kwargs["verbose"]

        
    if "bbox_center_type" in kwargs:
        DINC_RECEPTOR_PARAMS.bbox_center_type = kwargs["bbox_center_type"]
    if "bbox_center_x" in kwargs:
        DINC_RECEPTOR_PARAMS.bbox_center_x = kwargs["bbox_center_x"]
    if "bbox_center_y" in kwargs:
        DINC_RECEPTOR_PARAMS.bbox_center_y = kwargs["bbox_center_y"]
    if "bbox_center_z" in kwargs:
        DINC_RECEPTOR_PARAMS.bbox_center_z = kwargs["bbox_center_z"]
    if "bbox_dim_type" in kwargs:
        DINC_RECEPTOR_PARAMS.bbox_dim_type = kwargs["bbox_dim_type"]
    if "bbox_dim_x" in kwargs:
        DINC_RECEPTOR_PARAMS.bbox_dim_x = kwargs["bbox_dim_x"]
    if "bbox_dim_y" in kwargs:
        DINC_RECEPTOR_PARAMS.bbox_dim_y = kwargs["bbox_dim_y"]
    if "bbox_dim_z" in kwargs:
        DINC_RECEPTOR_PARAMS.bbox_dim_z = kwargs["bbox_dim_z"]
    if "align_receptors" in kwargs:
        DINC_RECEPTOR_PARAMS.align_receptors = kwargs["align_receptors"]
    if "ref_receptor" in kwargs:
        DINC_RECEPTOR_PARAMS.ref_receptor = kwargs["ref_receptor"]

    if "vina_score_f" in kwargs:
        VINA_ENGINE_PARAMS.score_f = kwargs["vina_score_f"]
    if "vina_exhaustive" in kwargs:
        VINA_ENGINE_PARAMS.exhaustive = kwargs["vina_exhaustive"]
    if "vina_n_poses" in kwargs:
        VINA_ENGINE_PARAMS.n_poses = kwargs["vina_n_poses"]
    if "vina_cpu_count" in kwargs:
        VINA_ENGINE_PARAMS.cpu_count = kwargs["vina_cpu_count"]
    if "vina_seed" in kwargs:
        VINA_ENGINE_PARAMS.seed = kwargs["vina_seed"]
    if "vina_min_rmsd" in kwargs:
        VINA_ENGINE_PARAMS.min_rmsd = kwargs["vina_min_rmsd"]
    if "vina_max_evals" in kwargs:
        VINA_ENGINE_PARAMS.max_evals = kwargs["vina_max_evals"]
    if "vina_rand_steps" in kwargs:
        VINA_ENGINE_PARAMS.rand_steps = kwargs["vina_rand_steps"]

    if "frag_mode" in kwargs:
        DINC_FRAG_PARAMS.frag_mode = kwargs["frag_mode"]
    if "frag_size" in kwargs:
        DINC_FRAG_PARAMS.frag_size = kwargs["frag_size"]
    if "frag_new" in kwargs:
        DINC_FRAG_PARAMS.frag_new = kwargs["frag_new"]
    if "root_type" in kwargs:
        DINC_FRAG_PARAMS.root_type = kwargs["root_type"]
    if "root_auto" in kwargs:
        DINC_FRAG_PARAMS.root_auto = kwargs["root_auto"]
    if "root_name" in kwargs:
        DINC_FRAG_PARAMS.root_name = kwargs["root_name"]