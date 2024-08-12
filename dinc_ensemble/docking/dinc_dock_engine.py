import os
import glob
import logging
from pathlib import Path
import pandas as pd
from dinc_ensemble.parameters import VINA_ENGINE_PARAMS, DINC_CORE_PARAMS


logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)

def dinc_probe_vina(probes_path_pdbqt: str,
                        receptor_path: str, 
                        output_file: str, 
                        box_path: str,  
                        exhaustiveness: int = VINA_ENGINE_PARAMS.exhaustive,
                        n_poses: int = VINA_ENGINE_PARAMS.n_poses,
                        max_evals: int = VINA_ENGINE_PARAMS.max_evals,
                        min_rmsd: int = VINA_ENGINE_PARAMS.min_rmsd, 
                        energy_range: int = VINA_ENGINE_PARAMS.energy_range,
                        seed: int = VINA_ENGINE_PARAMS.seed,
                        cpu_count: int = VINA_ENGINE_PARAMS.cpu_count):
    # ligand_path_pdbqt here must be the leaved devided by space
    lig_paths = probes_path_pdbqt.split()
    lig_out_paths = [lp[:lp.rfind(".")]+"_out.pdbqt" for lp in lig_paths]
    if not Path(output_file).exists():
        # run vina multi-ligand
        os.system("vina --ligand {lig} --receptor {rec} \
                --exhaustiveness={ex} --config {box} \
                --num_modes {n_poses} --out {out} \
                --max_evals {max_e} --min_rmsd {min_r} \
                --energy_range {nrg_rng} --seed {seed} \
                --cpu {cpu} --verbosity 2".format(lig=probes_path_pdbqt,
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
    results = None
    # process the file to score the probes
    # 1 - split all models
    os.system("vina_split  --input {} --ligand {} ".format(output_file, "tmp_lig"))
    model_files = glob.glob("tmp_lig*.pdbqt")
    
    # 2 - extract individual fragments from there
    leaves = []
    cur_leaf = []
    lines = None
    for i, file in enumerate(model_files): 
        with open(file, "r") as f:
            lines = f.readlines()
        for line in lines:
            cur_leaf.append(line)
            if "TORSDOF" in line:
                leaves.append(cur_leaf)
                cur_leaf = []
    # 3 - write individual fragments to file
    root_dir = Path(output_file).parent
    split_dir = root_dir / "probe_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    outfiles = []
    for i, leaf in enumerate(leaves):
        leaf_index = i%len(lig_paths)
        model_index = i//len(lig_paths)
        outfile = str(Path(lig_out_paths[leaf_index]).stem)
        outfile = outfile+"_m{}.pdbqt".format(model_index)
        outfile = split_dir / outfile
        with open(outfile, "w") as f:
            f.writelines(leaf)
        outfiles.append(outfile)
    # 4 - remove all the model files
    os.system("rm {}".format(" ".join(model_files)))
    # 5 - score the fragments
    binding_affinities = {}
    for outfile in outfiles:
        outfile_str = str(outfile)
        log_fname = outfile_str[:outfile_str.rfind(".")]+".txt"
        os.system("vina --score_only --ligand {} --receptor {} --config {} >> {}".format(outfile, 
                                                                             receptor_path, 
                                                                             box_path,
                                                                             log_fname))
            
        with open(log_fname, "r") as log:
            lines = log.readlines()
            if "Estimated Free Energy of Binding   :" in "\n".join(lines):
                ba = float(
                            next(l.split()[6] for l in lines if "Estimated Free Energy of Binding   :" in l)
                        )
                binding_affinities[outfile_str] = float(ba)
        os.system("rm {}".format(log_fname))
        #call("rm {}".format(log))
    # 6 - get the best probe!
    best_probe = min(binding_affinities, key=binding_affinities.get)
    best_leaf = int(best_probe.split("_")[-3])
    # 7 - suggest root atoms within this probe
    logger.info("Best energy achieved for leaf {}! \
                Suggest choosing a root atom within this leaf node!".format(best_leaf))
    ba_df = pd.DataFrame({"file": binding_affinities.keys(),
                 "energy": binding_affinities.values()})
    return ba_df, best_leaf
    
    
    

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
        #print(rand_fname)
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
    