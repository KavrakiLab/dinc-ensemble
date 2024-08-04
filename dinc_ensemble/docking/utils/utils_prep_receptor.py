
from ...parameters import DINC_RECEPTOR_PARAMS
from ...parameters.receptor import *
from typing import List
from ..pymol_utils.pymol_align_receptors import align_receptors_pymol
from MolKit import Read as MolKitRead
from pathlib import Path
import logging
logger = logging.getLogger('dinc_ensemble.docking.run')
logger.setLevel(logging.DEBUG)



def align_receptors(receptors: List[Path], 
                    ensemble_dir: Path):
    # align the receptors
    if DINC_RECEPTOR_PARAMS.align_receptors and len(receptors) > 1:

        logger.info("Aligning receptors")
        rec_ref_ind = DINC_RECEPTOR_PARAMS.ref_receptor
        if rec_ref_ind < 0 or rec_ref_ind > len(receptors):
            raise ValueError("DINCEnsemble: Reference receptor index wrong: {}".format(rec_ref_ind))
        rec_ref = receptors[rec_ref_ind]

        logger.info("- Aligning to: {}".format(rec_ref))
        new_receptor_files = align_receptors_pymol(rec_ref, 
                              receptors, 
                              ensemble_dir)
        logger.info("- Saved aligned receptors to: {}".format(ensemble_dir))
        
        receptors = new_receptor_files
    return receptors
        


def prepare_bbox(ligand_file: Path, 
                 receptor_files: List[Path],
                 bbox_file_loc: Path,
                 bbox_parameters: DincReceptorParams):
    

    logger.info("Preparing binding box!")
    # define the box center
    # check the type of box
    box_center = [bbox_parameters.bbox_center_x,
                  bbox_parameters.bbox_center_y,
                  bbox_parameters.bbox_center_z]
    if bbox_parameters.bbox_center_type == BBOX_CENTER_TYPE.LIGC:
        logger.info("- Setting binding box center to ligand")
        ligand = MolKitRead(str(ligand_file))[0]
        box_center = ligand.getCenter()
        bbox_parameters.bbox_center_x = box_center[0]
        bbox_parameters.bbox_center_y = box_center[1]
        bbox_parameters.bbox_center_z = box_center[2]
    if bbox_parameters.bbox_center_type == BBOX_CENTER_TYPE.PROTC:
        logger.info("- Setting binding box center to receptor")
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
        logger.info("- Setting binding box dimensions to ligand")
        ligand = MolKitRead(str(ligand_file))[0]
        min_c = [min([a.coords[i] for a in ligand.allAtoms]) for i in range(3)]
        max_c = [max([a.coords[i] for a in ligand.allAtoms]) for i in range(3)]
        box_dims = [(max_c[i] - min_c[i] + bbox_parameters.bbox_padding) for i in range(3)]
        bbox_parameters.bbox_dim_x = box_dims[0]
        bbox_parameters.bbox_dim_y = box_dims[1]
        bbox_parameters.bbox_dim_z = box_dims[2]
    logger.info("{} {}".format(box_center, box_dims))
    output_txt = "\
                \ncenter_x = {}\
                \ncenter_y = {}\
                \ncenter_z = {}\
                \nsize_x = {}\
                \nsize_y = {}\
                \nsize_z = {}\
                ".format(box_center[0],
                        box_center[1],
                        box_center[2],
                        box_dims[0],
                        box_dims[1],
                        box_dims[2])
    bbox_file = bbox_file_loc
    with open(bbox_file, 'w') as file:
        file.write(output_txt)
    return bbox_file




    


