from numpy import linspace
from math import floor, ceil
from os import path, listdir
from collections import defaultdict

# Import matplotlib and fix display issue in Tkinter
#from matplotlib import use as matplotUse
#matplotUse("Agg")
from matplotlib import pyplot
from matplotlib.cm import rainbow as matplotRainbow
from .rmsd import calculate_rmsd

#from dinc_ensemble.utils import calculate_rmsd
#from dinc_ensemble.docking.old_postprocessing import process_docking_output


# Construct the Energy vs RMSD plot.
#   ligand: ligand molecule processed by DINC
#   atm_bonds: dictionary listing, for each ligand's atom, the atoms it shares a bond with
#   top_conf: lowest-energy conformation produced by DINC for this ligand
#   receptor: name of the file containing the receptor's description
#   params: parameters of the docking job
#   dir_path: directory containing the docking files and where the plot should be created
#
def plot_score_vs_rmsd(ligand, atm_bonds, top_conf, receptor, params, dir_path):
    # data dictionary: fragment number --> list of (RMSD, energy) pairs
    data = defaultdict(list)

    # find all the files containing the intermediate docking results in the appropriate directory
    files = []
    if params["sampling"] == "AD4":
        files = [f for f in listdir(dir_path) if f.endswith(".dlg")]
    elif params["sampling"] == "Vina":
        files = [f for f in listdir(dir_path) if f.endswith("_out.pdbqt")]
    for f in files:
        file_name = path.join(dir_path, path.basename(f))
        # get the conformations stored in the docking file and compute their associated
        # (RMSD, energy) pairs
        rmsd_energy_pairs = []
        confs = process_docking_output(
            [file_name], ligand, atm_bonds, receptor, params, dir_path, True, False
        )
        if not confs:
            continue

        # the reference for RMSD calculations is the initial ligand, for redocking jobs,
        # or lowest-energy conformation produced by DINC, for crossdocking jobs
        if params["job_type"] == "redocking":
            ref_mol = ligand
        elif params["job_type"] == "crossdocking":
            ref_mol = top_conf.mol
            ref_mol.allAtoms.updateCoords(top_conf.coords, 0)
        for c in confs:
            minrmsd = calculate_rmsd(c, ref_mol, params, dir_path)
            rmsd_energy_pairs.append((minrmsd, c.binding_energy))
        # use the fragment number (in the name of the docking file) as an index in the dictionary
        data[int(f.split("_frag_")[-1].split("_")[0])].extend(rmsd_energy_pairs)

    # write the file containing the (RMSD, energy) pairs
    with open(path.join(dir_path, "score_vs_rmsd.dat"), "w") as file:
        for k in list(data.keys()):
            file.write("Fragment " + str(k) + " (RMSD, energy)\n")
            data[k].sort(key=lambda x: x[1])  # sort the dictionary entries by energy
            for pair in data[k]:
                file.write(str(pair) + "\n")
            file.write("\n")

    # the chosen (RMSD, energy) pairs correspond to the lowest-energy conformations selected by
    # DINC;
    # all the other (RMSD, energy) pairs are stored together;
    # max_rmsd, min_nrg, and max_nrg are used to capture all the scatter plot data (min_rmsd is
    # always 0)
    chosen_rmsds = []
    chosen_energies = []
    other_rmsds = []
    other_energies = []
    for k in list(data.keys()):
        (rmsd, energy) = list(map(list, list(zip(*data[k]))))
        chosen_rmsds.append(rmsd[: params["num_threads"]])
        chosen_energies.append(energy[: params["num_threads"]])
        other_rmsds.append(rmsd[params["num_threads"] :])
        other_energies.append(energy[params["num_threads"] :])
        max_rmsd = max(rmsd) if k == 0 else max(max_rmsd, max(rmsd))
        max_nrg = max(energy) if k == 0 else max(max_nrg, max(energy))
        min_nrg = min(energy) if k == 0 else min(min_nrg, min(energy))

    # (RMSD, energy) pairs are plotted using a color gradient to distinguish the fragments
    pyplot.figure()
    colors = iter(matplotRainbow(linspace(0, 1, len(data))))
    for k in list(data.keys()):
        color = next(colors)
        # "other" points are plotted first, and have a bit of transparency
        pyplot.scatter(
            other_rmsds[k],
            other_energies[k],
            color=color,
            label="Fragment " + str(k),
            alpha=0.8,
            marker="o",
        )
        # lowest-energy conformations are plotted on top of the others
        pyplot.scatter(
            chosen_rmsds[k],
            chosen_energies[k],
            color=color,
            label="Lowest-energy Frag. " + str(k),
            marker="s",
            edgecolor="black",
        )

    # create and save the plot
    pyplot.title("Energy vs RMSD plot")
    pyplot.xlabel("RMSD")
    pyplot.ylabel("Binding Energy")
    pyplot.xlim(0, ceil(max_rmsd))
    pyplot.ylim(floor(min_nrg), ceil(max_nrg))
    pyplot.grid(True)
    pyplot.axvline(x=2.0, color="green", ls="dashed")
    pyplot.legend(bbox_to_anchor=(1.01, 1.0), loc=2, borderaxespad=0.0)
    pyplot.savefig(path.join(dir_path, "score_vs_rmsd.png"), bbox_inches="tight")
    pyplot.close()
