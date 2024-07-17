
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
    if "score_f" in kwargs:
        
        VINA_ENGINE_PARAMS.score_f = kwargs["score_f"]
    if "exhaustive" in kwargs:
        VINA_ENGINE_PARAMS.exhaustive = kwargs["exhaustive"]
    if "n_poses" in kwargs:
        VINA_ENGINE_PARAMS.n_poses = kwargs["n_poses"]
    if "cpu_count" in kwargs:
        VINA_ENGINE_PARAMS.cpu_count = kwargs["cpu_count"]
    if "seed" in kwargs:
        VINA_ENGINE_PARAMS.seed = kwargs["seed"]
    if "min_rmsd" in kwargs:
        VINA_ENGINE_PARAMS.min_rmsd = kwargs["min_rmsd"]
    if "max_evals" in kwargs:
        VINA_ENGINE_PARAMS.max_evals = kwargs["max_evals"]
    if "rand_steps" in kwargs:
        VINA_ENGINE_PARAMS.rand_steps = kwargs["rand_steps"]