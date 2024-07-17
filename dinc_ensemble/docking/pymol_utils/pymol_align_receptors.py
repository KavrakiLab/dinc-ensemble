import pymol
from typing import List
from pathlib import Path

def align_receptors_pymol(receptor_ref: Path, 
                    all_receptors: List[Path], 
                    output_dir: Path,
                    save_pse: bool = False, # save a pymol scene?
                    ) -> List[Path]:
    # load and align all inputs
    pymol.finish_launching(['pymol', '-cq'])
    pymol.cmd.load(receptor_ref, 'receptor_ref')
    for i, rec in enumerate(all_receptors):
        current_receptor = "receptor_{}".format(i)
        pymol.cmd.load(rec, current_receptor)
        if rec != receptor_ref:
            pymol.cmd.super(current_receptor, 'receptor_ref')
    # save in the output_dir
    new_receptors = []
    for i, rec in enumerate(all_receptors):
        current_receptor = "receptor_{}".format(i)
        rec_fname = rec.name[:rec.name.rfind(".")]+"_aligned.pdb"
        out_file = output_dir / rec_fname
        pymol.cmd.save(out_file, current_receptor)
        new_receptors.append(out_file)
    # save the aligned receptor scene
    output_scene = output_dir / "aligned_receptors.pse"
    pymol.cmd.hide('all')
    pymol.cmd.show('cartoon', 'all')
    pymol.cmd.save(output_scene)
    #dump_rep("receptor_ref")
    return new_receptors
    