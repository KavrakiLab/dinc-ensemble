
from ...parameters import DINC_RECEPTOR_PARAMS
from ...parameters.receptor import *
from typing import List
from ..pymol_utils.pymol_align_receptors import align_receptors_pymol
from MolKit import Read as MolKitRead
from pathlib import Path
verbose = True



def align_receptors(receptors: List[Path], 
                    ensemble_dir: Path):
    # align the receptors
    if DINC_RECEPTOR_PARAMS.align_receptors and len(receptors) > 1:
        if verbose: 
            print("-------------------------------------")
            print("DINC-Ensemble: Aligning receptors!")
            print("-------------------------------------")
        rec_ref_ind = DINC_RECEPTOR_PARAMS.ref_receptor
        if rec_ref_ind < 0 or rec_ref_ind > len(receptors):
            raise ValueError("DINCEnsemble: Reference receptor index wrong: {}".format(rec_ref_ind))
        rec_ref = receptors[rec_ref_ind]
        if verbose: 
            print("Aligning to: {}".format(rec_ref))
        new_receptor_files = align_receptors_pymol(rec_ref, 
                              receptors, 
                              ensemble_dir)
        if verbose: 
            print("Saved aligned receptors to: {}".format(ensemble_dir))
        
        receptors = new_receptor_files
    return receptors
        


def prepare_bbox(ligand_file: Path, 
                 receptor_files: List[Path],
                 bbox_parameters: DincReceptorParams):
    
    if verbose: 
        print("-------------------------------------")
        print("DINC-Ensemble: Preparing binding box!")
        print("-------------------------------------")
    # define the box center
    # check the type of box
    box_center = [bbox_parameters.bbox_center_x,
                  bbox_parameters.bbox_center_y,
                  bbox_parameters.bbox_center_z]
    if bbox_parameters.bbox_center_type == BBOX_CENTER_TYPE.LIGC:
        if verbose: print("Setting binding box center to ligand")
        ligand = MolKitRead(str(ligand_file))[0]
        box_center = ligand.getCenter()
        bbox_parameters.bbox_center_x = box_center[0]
        bbox_parameters.bbox_center_y = box_center[1]
        bbox_parameters.bbox_center_z = box_center[2]
    if bbox_parameters.bbox_center_type == BBOX_CENTER_TYPE.PROTC:
        if verbose: print("Setting binding box center to receptor")
        receptor_index = bbox_parameters.ref_receptor
        receptor = MolKitRead(str(receptor_files[receptor_index]))[0]
        box_center = receptor.getCenter()
        bbox_parameters.bbox_center_x = box_center[0]
        bbox_parameters.bbox_center_y = box_center[1]
        bbox_parameters.bbox_center_z = box_center[2]

    box_dims = [bbox_parameters.bbox_dim_x,
                bbox_parameters.bbox_dim_y,
                bbox_parameters.bbox_dim_z]
    
    if bbox_parameters.bbox_dim_type == BBOX_DIM_TYPE.LIG:
        if verbose: print("Setting binding box dimensions to ligand")
        ligand = MolKitRead(str(ligand_file))[0]
        min_c = [min([a.coords[i] for a in ligand.allAtoms]) for i in range(3)]
        max_c = [max([a.coords[i] for a in ligand.allAtoms]) for i in range(3)]
        box_dims = [(max_c[i] - min_c[i] + bbox_parameters.bbox_padding) for i in range(3)]
        bbox_parameters.bbox_dim_x = box_dims[0]
        bbox_parameters.bbox_dim_y = box_dims[1]
        bbox_parameters.bbox_dim_z = box_dims[2]
    print(box_center, box_dims)
    return box_center, box_dims




    


