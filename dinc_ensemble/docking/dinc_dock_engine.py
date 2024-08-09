import os
import logging
from pathlib import Path
from dinc_ensemble import VINA_ENGINE_PARAMS, DINC_CORE_PARAMS


logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)

def dinc_run_single_vina(ligand_path_pdbqt: str,
                        receptor_path: str, 
                        output_file: str, 
                        randomize: bool,
                        box_path: str,  
                        exhaustiveness: int = VINA_ENGINE_PARAMS.exhaustive,
                        n_poses: int = VINA_ENGINE_PARAMS.n_poses,
                        max_evals: int = VINA_ENGINE_PARAMS.max_evals,
                        min_rmsd: int = VINA_ENGINE_PARAMS.min_rmsd, 
                        energy_range: int = VINA_ENGINE_PARAMS.energy_range,
                        seed: int = VINA_ENGINE_PARAMS.seed,
                        continue_run: bool = DINC_CORE_PARAMS.continue_run,
                        cpu_count: int = VINA_ENGINE_PARAMS.cpu_count):
    #logger.info("Starting vina run for: {}".format(ligand_path_pdbqt))
    logger.debug("Docking with Vina")
    logger.debug("------------------")
    logger.debug("Ligand: {}".format(ligand_path_pdbqt))
    logger.debug("Receptor: {}".format(receptor_path))
    logger.debug("Output file: {}".format(output_file))
    logger.debug("Randomize: {}".format(randomize))
    logger.debug("Bbox path: {}".format(box_path))
    logger.debug("------------------")
    
    # randomize if needed
    if randomize:
        ligand_path_pdbqt_str = str(ligand_path_pdbqt)
        rand_fname = ligand_path_pdbqt_str[:ligand_path_pdbqt_str.rfind(".")]+"_rand.pdbqt"
        print(rand_fname)
        #if not Path(rand_fname).exists():
        os.system("vina --ligand {lig} --receptor {rec} \
            --randomize_only --out {out_file} --config {box} --verbosity 0".format(
            lig=ligand_path_pdbqt,
            rec=receptor_path,
            out_file=rand_fname,
            box=box_path
        ))
        ligand_path_pdbqt = rand_fname
    # dock
    if not (continue_run and Path(output_file).exists()):
        os.system("vina --ligand {lig} --receptor {rec} \
                --exhaustiveness={ex} --config {box} \
                --num_modes {n_poses} --out {out} \
                --max_evals {max_e} --min_rmsd {min_r} \
                --energy_range {nrg_rng} --seed {seed} \
                --cpu {cpu} --verbosity 0".format(lig=ligand_path_pdbqt,
                                    rec=receptor_path,
                                    box=box_path,
                                    ex=exhaustiveness,
                                    max_e=max_evals,
                                    min_r=min_rmsd,
                                    nrg_rng=energy_range,
                                    seed=seed,
                                    out=output_file,
                                    n_poses=n_poses,
                                    cpu=cpu_count))
    