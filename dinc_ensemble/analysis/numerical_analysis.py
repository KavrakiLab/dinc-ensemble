import tarfile

from copy import deepcopy
from os import path, listdir
from collections import defaultdict

# Import matplotlib and fix display issue in Tkinter
#from matplotlib import use as matplotUse
#matplotUse("Agg")

from MolKit import Read as MolKitRead
from AutoDockTools.MoleculePreparation import AD4FlexibleReceptorPreparation
from AutoDockTools.cluster import Clusterer

#from dinc_ensemble.utils import calculate_rmsd
#from dinc_ensemble.ligand.ligand_prepare import write_conformation
#from dinc_ensemble.docking.old_postprocessing import process_docking_output


# Analyze the docking results:
#  - rank all generated conformations by increasing binding energy
#  - calculate RMSD between all conformations and a reference that depends on the job type
#  - cluster conformations based on their energy and respective RMSD
#  - save statistics and most representative conformations to file.
# Return the lowest-energy conformation produced by DINC.
#   dock_files: output files from the docking tool, which contain the ligand's conformations
#   ligand: ligand molecule processed by DINC
#   bonded_atoms: dictionary listing, for each ligand's atom, the atoms it shares a bond with
#   back_trace: dictionary allowing to trace back the original names of the ligand's atoms
#   atom_order: order of atoms in input ligand
#   receptor_name: name of the receptor (as specified by the user's input)
#   receptor_file: pdbqt file describing the receptor
#   params: parameters of the docking job
#   time: running time of the incremental docking process
#   dir_path: directory containing the docking files
#
def analyze_results(
    dock_files,
    ligand,
    bonded_atoms,
    back_trace,
    atom_order,
    receptor_name,
    receptor_file,
    params,
    time,
    dir_path,
):
    # read the docking results of the ligand and sort its conformations by increasing energy
    conformations = process_docking_output(
        dock_files, ligand, bonded_atoms, receptor_file, params, dir_path
    )

    # save the lowest-energy binding modes into separate files
    for n, conf in enumerate(conformations):
        conf.nrg_rank = n + 1
    best_energies = (
        conformations
        if params["num_output"] == "all"
        else conformations[: params["num_output"]]
    )
    for n, conf in enumerate(best_energies):
        file_name = path.join(dir_path, "dincresults_energy_%s.pdb" % str(n + 1))
        write_conformation(conf, file_name, atom_order, back_trace)
        if params["flex"]:
            AD4FlexibleReceptorPreparation(
                MolKitRead(path.join(dir_path, ligand.name + "_prot.pdbqt"))[0],
                residues=deepcopy(conf).flex_res,
                flexres_filename=path.join(
                    dir_path, "dincresults_energy_%s_flex.pdb" % str(n + 1)
                ),
                rigid_filename=path.join(
                    dir_path, "dincresults_energy_%s_rigid.pdb" % str(n + 1)
                ),
            )

    # rank conformations by increasing RMSD;
    # save the lowest-RMSD ones to file, in the case of re-docking jobs;
    # note that RMSD values are calculated with respect to
    #   the original ligand conformation, for re-docking jobs, or
    #   the lowest-energy conformation produced by DINC, for cross-docking jobs
    if params["job_type"] == "redocking":
        ref_mol = ligand
    elif params["job_type"] == "crossdocking":
        ref_mol = best_energies[0].mol
        ref_mol.allAtoms.updateCoords(best_energies[0].coords, 0)
    for conf in conformations:
        conf.minrmsd = calculate_rmsd(conf, ref_mol, params, dir_path)
    conformations.sort(key=lambda c: c.minrmsd)
    for n, conf in enumerate(conformations):
        conf.rmsd_rank = n + 1
    if params["job_type"] == "redocking" and params["num_output"] != "all":
        best_RMSDs = conformations[: params["num_output"]]
        for n, conf in enumerate(best_RMSDs):
            file_name = path.join(dir_path, "dincresults_rmsd_%s.pdb" % str(n + 1))
            write_conformation(conf, file_name, atom_order, back_trace)
            if params["flex"]:
                AD4FlexibleReceptorPreparation(
                    MolKitRead(path.join(dir_path, ligand.name + "_prot.pdbqt"))[0],
                    residues=deepcopy(conf).flex_res,
                    flexres_filename=path.join(
                        dir_path, "dincresults_rmsd_%s_flex.pdb" % str(n + 1)
                    ),
                    rigid_filename=path.join(
                        dir_path, "dincresults_rmsd_%s_rigid.pdb" % str(n + 1)
                    ),
                )

    # cluster the conformations and save the representatives of the largest clusters to file
    # (a cluster's representative is the lowest-energy conformation within this cluster)
    # NB: the AutoDock clustering method is as follows:
    #   1) we sort all the conformations by increasing energy;
    #   2) we visit every conformation in order, adding it to the current cluster
    #      if it is within the RMSD tolerance, or seeding a new cluster otherwise
    # (as a consequence, conformations are already sorted by energy within clusters)
    if params["num_output"] != "all":
        rmsd_tolerance = 3.0
        clusterer = Clusterer(conformations)
        clusterer.make_clustering(rmsd_tolerance)
        all_clusters = clusterer.clustering_dict[rmsd_tolerance]
        all_clusters.sort(key=lambda c: len(c), reverse=True)
        for k, cluster in enumerate(all_clusters):
            for n, conf in enumerate(cluster):
                conf.clust_size_rank = k + 1
                conf.clust_nrg_rank = n + 1
        best_clusters = [c[0] for c in list(all_clusters)[: params["num_output"] :]]
        for n, conf in enumerate(best_clusters):
            file_name = path.join(dir_path, "dincresults_cluster_%s.pdb" % str(n + 1))
            write_conformation(conf, file_name, atom_order, back_trace)
            if params["flex"]:
                AD4FlexibleReceptorPreparation(
                    MolKitRead(path.join(dir_path, ligand.name + "_prot.pdbqt"))[0],
                    residues=deepcopy(conf).flex_res,
                    flexres_filename=path.join(
                        dir_path, "dincresults_cluster_%s_flex.pdb" % str(n + 1)
                    ),
                    rigid_filename=path.join(
                        dir_path, "dincresults_cluster_%s_rigid.pdb" % str(n + 1)
                    ),
                )
    else:
        for conf in conformations:
            conf.clust_size_rank = 1
            conf.clust_nrg_rank = conf.nrg_rank

    # create the .dat file containing all the stats about the conformations produced by DINC
    conformations.sort(key=lambda c: c.nrg_rank)
    with open(path.join(dir_path, ligand.name + "_stats.dat"), "w") as f:
        f.write("TORSDOF: " + str(conformations[0].mol.TORSDOF) + "\n")
        f.write(
            "Id | Energy | Nrg_rank | RMSD | RMSD_rank | Clust_rank | Clust_nrg_rank\n"
        )
        for i, c in enumerate(conformations):
            f.write(
                "%d\t%.2f\t\t%d\t\t%.2f\t\t%d\t\t\t%d\t\t\t%d\n"
                % (
                    i + 1,
                    c.binding_energy,
                    c.nrg_rank,
                    c.minrmsd,
                    c.rmsd_rank,
                    c.clust_size_rank,
                    c.clust_nrg_rank,
                )
            )

    # get the box center type
    if params["box_center_type"] == "ligc":
        box_center_type = "ligand center"
    elif params["box_center_type"] == "protc":
        box_center_type = "protein center"
    else:
        box_center_type = "user-specified"

    # round the box center coordinates and the box dimensions to 1 decimal place
    rounded_box_center = [round(float(c), 1) for c in params["box_center"]]
    rounded_box_dimensions = [round(float(c), 1) for c in params["box"]]

    # create DINC's output file
    f = open(path.join(dir_path, "dincresults.txt"), "w")
    f.write("Docking results from DINC\n\n")
    f.write("Input ligand: " + ligand.name + "\n")
    f.write("Input receptor: " + path.splitext(path.basename(receptor_name))[0] + "\n")
    f.write("Job type: " + params["job_type"] + "\n")
    f.write("Binding box center type: " + box_center_type + "\n")
    f.write("Box center coordinates: " + str(tuple(rounded_box_center)) + " angstrom\n")
    f.write("Binding box type: " + params["box_type"] + "\n")
    f.write(
        "Box dimensions (l, w, h): "
        + str(tuple(rounded_box_dimensions))
        + " angstrom\n"
    )
    f.write("Docking method: " + params["sampling"] + "\n")
    if params["scoring"] != params["sampling"]:
        f.write("Scoring method: " + params["scoring"] + "\n")
    if params["rescoring"] != params["sampling"]:
        f.write("Rescoring method: " + params["rescoring"] + "\n")
    f.write("Prepare ligand: " + str(params["prepare_ligand"]) + "\n")
    f.write("\n*********************************************************\n")

    array_line = "--------------------------------------------------\n"
    array_header = (
        array_line + "\t File name \t\t Energy (kcal/mol) \t RMSD (A) \n" + array_line
    )
    f.write("\n+ Conformations sorted by increasing binding energy:\n")
    f.write(array_header)
    for n, c in enumerate(best_energies):
        f.write(
            "dincresults_energy_%s\t\t%.2f\t\t%.2f\n"
            % (str(n + 1), c.binding_energy, c.minrmsd)
        )
    f.write(array_line)

    if params["num_output"] != "all":
        f.write("\n+ Conformations sorted by decreasing cluster size\n")
        f.write("(lowest-energy conformation from each cluster):\n")
        f.write(array_header)
        for n, c in enumerate(best_clusters):
            f.write(
                "dincresults_cluster_%s\t\t%.2f\t\t%.2f\n"
                % (str(n + 1), c.binding_energy, c.minrmsd)
            )
        f.write(array_line)

    if params["job_type"] == "redocking" and params["num_output"] != "all":
        f.write("\n+ Conformations sorted by increasing RMSD:\n")
        f.write(array_header)
        for n, c in enumerate(best_RMSDs):
            f.write(
                "dincresults_rmsd_%s\t\t\t%.2f\t\t%.2f\n"
                % (str(n + 1), c.binding_energy, c.minrmsd)
            )
        f.write(array_line)

    f.write("\nNote: RMSD values are computed with respect to the\n")
    if params["job_type"] == "redocking":
        f.write("      input conformation of the ligand\n")
    elif params["job_type"] == "crossdocking":
        f.write("      lowest-energy conformation produced by DINC\n")

    f.write("\nTotal runtime = " + time)
    f.write("\n\nThank you for using DINC\n")
    f.close()

    # create an archive containing all the results
    results = [
        f
        for f in listdir(dir_path)
        if f.startswith("dincresults_energy")
        or f.startswith("dincresults_rmsd")
        or (f.startswith("dincresults_cluster") and params["num_output"] != "all")
    ]
    results.append("dincresults.txt")
    with tarfile.open(path.join(dir_path, "dincresults.tar.gz"), "w|gz") as tar:
        for f in results:
            tar.add(path.join(dir_path, f), arcname=f)

    return best_energies[0]
