import typer
import click
from enum import EnumType
from typing import List
from os import path
from pathlib import Path
import json

from dinc_ensemble.parameters.analysis import *
from dinc_ensemble.parameters.fragment import *
from dinc_ensemble.parameters.core import *
from dinc_ensemble.parameters import *
from dinc_ensemble.parameters.dock_engine_vina import *
from dinc_ensemble.parameters.receptor import DEFAULT_BBOX_CENTER_X, DEFAULT_BBOX_CENTER_Y, DEFAULT_BBOX_CENTER_Z, \
                                                BBOX_DIM_TYPE, DEFAULT_BBOX_DIM_TYPE, \
                                                DEFAULT_BBOX_DIM_X, DEFAULT_BBOX_DIM_Y, DEFAULT_BBOX_DIM_Z, \
                                                DEFAULT_BBOX_CENTER_TYPE, \
                                                DEFAULT_ALIGN_RECEPTORS, DEFAULT_REF_RECEPTOR_IND, DincReceptorParams, \
                                                BBOX_CENTER_TYPE

from dinc_ensemble import load_ligand
from dinc_ensemble import dinc_full_run
from dinc_ensemble.ligand import DINCFragment
from typing_extensions import Annotated

import logging
logger = logging.getLogger('dinc_ensemble.ligand')


from .utils import init_all_dince_params


app = typer.Typer(pretty_exceptions_show_locals=False)

analyze_docstrings = DincAnalysisParams.__doc_arg__ if DincAnalysisParams.__doc_arg__ is not None else {}
@app.command(name="analyze")
def analyze(
    n_out: int = typer.Option(
        default=DEFAULT_DINC_N_OUT, 
        help=analyze_docstrings["n_out"]
                            ),
    plot_score_vs_rmsd: DINC_PLOT_SCORE_RMSD = typer.Option(
        default=DEFAULT_DINC_PLOT_SCORE_RMSD, 
        help=analyze_docstrings["plot_score_vs_rmsd"]
        ),
    dinc_rmsd: DINC_RMSD = typer.Option(
        default=DEFAULT_DINC_RMSD, 
        help=analyze_docstrings["dinc_rmsd"]
        )
):
    
    DINC_ANALYSIS_PARAMS.n_out = n_out
    DINC_ANALYSIS_PARAMS.plot_score_vs_rmsd = plot_score_vs_rmsd
    DINC_ANALYSIS_PARAMS.dinc_rmsd = dinc_rmsd
    logger.info("Hello world")
    logger.info("These are the parameters: {}".format(DINC_ANALYSIS_PARAMS))
    # TODO:implement this step
    

fragment_docstrings = DincFragParams.__doc_arg__ if DincFragParams.__doc_arg__ is not None else {}
@app.command(name="fragment")
def fragment(
    ligand: Path = typer.Argument(
        help = "Path to the input ligand to be fragmented.\n \
                Possible extensions (.mol2, .sdf, .fasta, .pdb) \
                Preferred file type: mol2."
    ),
    output_dir: Path = typer.Argument(
        default= ".",
        help = "Directory where to save the outputs. \
                    Outputs: (1) fragment pdbqt files,\
(2) fragment SVG files,\
(3) list of fragments in HTML format \
(current directory)"
    ),
    frag_mode: DINC_FRAGMENT_MODE  = typer.Option(
        default = DEFAULT_DINC_FRAGMENT_MODE,
        help = fragment_docstrings["frag_mode"]),
    frag_size: int = typer.Option(
        default = DEFAULT_DINC_FRAG_SIZE,
        help = fragment_docstrings["frag_size"]),
    frag_new: int = typer.Option(
        default = DEFAULT_DINC_FRAG_NEW,
        help = fragment_docstrings["frag_new"]),
    root_type: DINC_ROOT_TYPE = typer.Option(
        default = DEFAULT_DINC_ROOT_TYPE,
        help = fragment_docstrings["root_type"]),
    root_auto: DINC_ROOT_AUTO = typer.Option(
        default = DEFAULT_DINC_ROOT_AUTO,
        help = fragment_docstrings["root_auto"]),
    root_name: str = typer.Option(
        default = None,
        help = fragment_docstrings["root_name"])
):  
    DINC_FRAG_PARAMS.frag_mode = frag_mode
    DINC_FRAG_PARAMS.frag_size = frag_size
    DINC_FRAG_PARAMS.frag_new = frag_new
    DINC_FRAG_PARAMS.root_type = root_type
    DINC_FRAG_PARAMS.root_auto = root_auto
    DINC_FRAG_PARAMS.root_name = root_name
    logger.info("DINC-Ensemble Fragment")
    logger.info("These are the parameters: {}".format(DINC_FRAG_PARAMS))
    logger.info("Loading DINC-Ensemble ligand")
    lig = load_ligand(str(ligand))
    frag = DINCFragment(lig, 
                        DincFragParams(
                            frag_mode=DINC_FRAG_PARAMS.frag_mode,
                        frag_size=DINC_FRAG_PARAMS.frag_size,
                        frag_new=DINC_FRAG_PARAMS.frag_new,
                        root_type=DINC_FRAG_PARAMS.root_type,
                        root_auto=DINC_FRAG_PARAMS.root_auto,
                        root_name=DINC_FRAG_PARAMS.root_name))

    # this expansion will inactivate certain bonds that should be inactive
    frag.freeze_bonds()

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    frag.write_pdbqt_frags(out_dir=output_dir)
    #frag._write_svg_frags_(out_dir=output_dir)
    #info_table_fname = lig.molkit_molecule.name + "_frag_info.html"
    #frags_info_df = frag._to_df_frags_info_()
    #frags_info_df["frag_svg_path"] = frags_info_df["frag_pdbqt"].apply(lambda x: str(x).split(".")[0]+".svg") # type: ignore
    #frags_info_df["frag_svg"] = frags_info_df["frag_svg_path"].apply(lambda x: '''<img src=\"./{}\"/>'''.format(x)) # type: ignore
    #frags_info_df = frags_info_df[["frag_id", # type: ignore
    #                                "frag_dof",
    #                                "frag_pdbqt",
    #                                "frag_svg"]]
    #frag_info_html = frags_info_df.to_html(render_links=True,escape=False)
    #with open(path.join(output_dir, info_table_fname), "w") as f:
    #    f.write(frag_info_html)


core_docstrings = DincCoreParams.__doc_arg__ if DincCoreParams.__doc_arg__ is not None else {}
bbox_docstrings = DincReceptorParams.__doc_arg__ if DincReceptorParams.__doc_arg__ is not None else {}
vina_docstrings = VinaEngineParams.__doc_arg__ if VinaEngineParams.__doc_arg__ is not None else {}
@app.command(name="dock")
def dock(
    # DOCKING CORE
    ligand_file: Annotated[Path, typer.Argument(help="Input ligand path.")], 
    receptor_files: Annotated[List[Path], typer.Argument(help="Input receptor paths.")], 
    
    output_dir: Annotated[Path, typer.Argument(help="Output directory for DINC-Ensemble run.")],
    
    parameters_json: Path = typer.Option(
        default=None,
        help="All parameters provided in a json format (default is None). \n \
            If provided it will be used to initialize DINC-Ensemble paramterers. \n \
            It overrides other parameters provided in CLI command. "
    ),

    # CORE OPTIONS
    job_type: DINC_JOB_TYPE = typer.Option(
        default=DEFAULT_DINC_JOB_TYPE,
        help=core_docstrings["job_type"]
    ),
    dock_type: DINC_DOCK_TYPE = typer.Option(
        default=DEFAULT_DINC_DOCK_TYPE,
        help=core_docstrings["dock_type"]
    ),
    dock_engine: DINC_DOCK_ENGINE = typer.Option(
        default=DEFAULT_DOCK_ENGINE,
        help=core_docstrings["dock_engine"]
    ),
    replica_num: int = typer.Option(
        default=DEFAULT_DINC_NUM_REPLICAS,
        help=core_docstrings["replica_num"]
    ),
    n_out: int = typer.Option(
        default=DEFAULT_N_OUT,
        help=core_docstrings["n_out"]
    ),    
    cpu_count: str = typer.Option(
        default = DEFAULT_CPU_CNT,
        help = core_docstrings["cpu_count"]
    ),
    continue_run: bool = typer.Option(
        default = DEFAULT_CONTINUE,
        help = core_docstrings["continue"]
    ),
    verbose: str = typer.Option(
        default = DEFAULT_CONTINUE,
        help = core_docstrings["verbose"]
    ),
    # RECEPTOR OPTIONS
    bbox_center_type: BBOX_CENTER_TYPE = typer.Option(
        default=DEFAULT_BBOX_CENTER_TYPE,
        help=bbox_docstrings["bbox_center_type"]
    ),
    bbox_center_x: int = typer.Option(
        default=DEFAULT_BBOX_CENTER_X,
        help=bbox_docstrings["bbox_center_x"]
    ),
    bbox_center_y: int = typer.Option(
        default=DEFAULT_BBOX_CENTER_Y,
        help=bbox_docstrings["bbox_center_y"]
    ),
    bbox_center_z: int = typer.Option(
        default=DEFAULT_BBOX_CENTER_Z,
        help=bbox_docstrings["bbox_center_z"]
    ),
    bbox_dim_type: BBOX_DIM_TYPE = typer.Option(
        default=DEFAULT_BBOX_DIM_TYPE,
        help=bbox_docstrings["bbox_dim_type"]
        ),         
    bbox_dim_x: int = typer.Option(
        default=DEFAULT_BBOX_DIM_X,
        help=bbox_docstrings["bbox_dim_x"]
        ),
    bbox_dim_y: int = typer.Option(
        default=DEFAULT_BBOX_DIM_Y,
        help=bbox_docstrings["bbox_dim_y"]
        ),
    bbox_dim_z: int = typer.Option(
        default=DEFAULT_BBOX_DIM_Z,
        help=bbox_docstrings["bbox_dim_z"]
        ),
    align_receptors: bool =typer.Option(
        default= DEFAULT_ALIGN_RECEPTORS,
        help=bbox_docstrings["align_receptors"]
        ),
    ref_receptor: int = typer.Option(
        default=DEFAULT_REF_RECEPTOR_IND,
        help=bbox_docstrings["ref_receptor"]
        ),
    # VINA ENGINE OPTIONS
    vina_score_f: SCORE_F = typer.Option(
        default=DEFAULT_SCORE_F,
        help=vina_docstrings["score_f"]
        ),
    vina_exhaustive: int = typer.Option(
        default=DEFAULT_VINA_EXHAUSTIVE,
        help=vina_docstrings["exhaustive"]
        ),
    vina_n_poses: int = typer.Option(
        default=DEFAULT_VINA_N_POSES,
        help=vina_docstrings["n_poses"]
        ),
    vina_cpu_count: int = typer.Option(
        default=DEFAULT_VINA_CPU_CNT,
        help=vina_docstrings["cpu_count"]
        ),
    vina_seed: int = typer.Option(
        default=DEFAULT_VINA_SEED,
        help=vina_docstrings["seed"]
        ),
    vina_min_rmsd: float = typer.Option(
        default=DEFAULT_VINA_MIN_RMSD,
        help=vina_docstrings["min_rmsd"]
        ),
    vina_energy_range: float = typer.Option(
        default=DEFAULT_VINA_ENERGY_RANGE,
        help=vina_docstrings["energy_range"]
        ),
    vina_max_evals: int = typer.Option(
        default=DEFAULT_VINA_MAX_EVALS,
        help=vina_docstrings["max_evals"]
        ),
    vina_rand_steps: int = typer.Option(
        default=DEFAULT_RAND_STEPS,
        help=vina_docstrings["rand_steps"]
        ),
    # FRAGMENT OPTIONS
    frag_mode: DINC_FRAGMENT_MODE  = typer.Option(
        default = DEFAULT_DINC_FRAGMENT_MODE,
        help = fragment_docstrings["frag_mode"]),
    frag_size: int = typer.Option(
        default = DEFAULT_DINC_FRAG_SIZE,
        help = fragment_docstrings["frag_size"]),
    frag_new: int = typer.Option(
        default = DEFAULT_DINC_FRAG_NEW,
        help = fragment_docstrings["frag_new"]),
    root_type: DINC_ROOT_TYPE = typer.Option(
        default = DEFAULT_DINC_ROOT_TYPE,
        help = fragment_docstrings["root_type"]),
    root_auto: DINC_ROOT_AUTO = typer.Option(
        default = DEFAULT_DINC_ROOT_AUTO,
        help = fragment_docstrings["root_auto"]),
    root_name: DINC_ROOT_AUTO = typer.Option(
        default = None,
        help = fragment_docstrings["root_name"])
    ):

    input_params = {
        "output_dir" :output_dir,
        "job_type" :job_type,
        "dock_type" :dock_type,
        "dock_engine" :dock_engine,
        "replica_num" :replica_num,
        "n_out" :n_out,
        "continue": continue_run,
        "cpu_count": cpu_count,
        "verbose": verbose,

        "bbox_center_type" :bbox_center_type,
        "bbox_center_x" :bbox_center_x,
        "bbox_center_y" :bbox_center_y,
        "bbox_center_z" :bbox_center_z,
        "bbox_dim_type" :bbox_dim_type,
        "bbox_dim_x" :bbox_dim_x,
        "bbox_dim_y" :bbox_dim_y,
        "bbox_dim_z" :bbox_dim_z,
        "align_receptors" :align_receptors,
        "ref_receptor" :ref_receptor,

        "vina_score_f" :vina_score_f,
        "vina_exhaustive" :vina_exhaustive,
        "vina_n_poses" :vina_n_poses,
        "vina_cpu_count" :vina_cpu_count,
        "vina_seed" :vina_seed,
        "vina_min_rmsd" :vina_min_rmsd,
        "vina_energy_range" :vina_energy_range,
        "vina_max_evals" :vina_max_evals,
        "vina_rand_steps" :vina_rand_steps,

        "frag_mode" :frag_mode,
        "frag_size" :frag_size,
        "frag_new" :frag_new,
        "root_type" :root_type,
        "root_auto" :root_auto,
        "root_name" :root_name
    }

    init_all_dince_params(**input_params)

    if parameters_json:
        if parameters_json.suffix != ".json":
            logger.error("DINC-Ensemble: parameters option must be a json file.")
            exit()
        else:
            with open(parameters_json, "r") as fp:
                parameters_kwars = json.load(fp)
                if "align_receptors" in parameters_kwars:
                    if parameters_kwars["align_receptors"].lower() in ['true', '1', 't', 'y', 'yes']:
                        parameters_kwars["align_receptors"] = True
                    else:
                        parameters_kwars["align_receptors"] = False
                    if parameters_kwars["continue"].lower() in ['true', '1', 't', 'y', 'yes']:
                        parameters_kwars["continue"] = True
                    else:
                        parameters_kwars["continue"] = False
                    
                init_all_dince_params(**parameters_kwars)


    dinc_full_run(str(ligand_file),[ str(r) for r in receptor_files])
    logger.info("Thank you for using DINC-Ensemble!")

if __name__ == "__main__":
    app()