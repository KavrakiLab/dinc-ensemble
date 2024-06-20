from numpy import array, sqrt, sum


# Return the all-heavy-atom RMSD between the given conformation and the given molecule.
#   params: parameters of the docking job
#   dir_path: directory where the docking job is running
#
def calculate_rmsd(conf, ref_mol, dir_path):
    ## non symmetry-corrected rmsd
    # find automorphism involving non-H atoms
    A1 = []
    A2 = []
    for a1 in conf.mol.allAtoms.get(lambda a: a.element != "H"):
        for a2 in ref_mol.allAtoms:
            if a1.name == a2.name:
                A2.append(a2)
                A1.append(a1)
    rmsd = None
    #if DincAnalysisParams == DINC_RMSD.NORMAL:
    delta = array([a.coords for a in A1]) - array([a.coords for a in A2])
    rmsd = sqrt(sum(sum((delta * delta).T)) / len(delta))

    ## symmetry-corrected rmsd
    #if DincAnalysisParams == DINC_RMSD.SYMMETRIC:
        # TODO: this relies on pybel maybe change dependency?
    return rmsd