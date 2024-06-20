from numpy import array
from math import ceil
from os import path
from copy import deepcopy
from subprocess import call
import fileinput

from mglutil.math.statetocoords import StateToCoords
from AutoDockTools.Conformation import Conformation
from AutoDockTools.MoleculePreparation import AD4FlexibleReceptorPreparation
from MolKit import Read as MolKitRead

from dinc_ensemble.utils import terminate  # or ..utils?
from dinc_ensemble.ligand.ligand_prepare import write_conformation


# Re-score the given conformations using the given scoring function.
# Note that their "binding_energy" attribute is modified in the process.
#   ligand: name of the ligand itself
#   receptor: name of the pdbqt file containing the receptor's description
#   params: parameters of the docking job
#   dir_path: directory containing the docking files
#
def rescore(conformations, function, ligand, receptor, params, dir_path):
    print("Scoring conformations with", function)

    if function == "AD4":
        if params["job_type"] != "scoring":
            # adjust binding box in case some atoms fall outside of it
            min_c = [
                min(
                    [
                        min([a.coords[i] for a in conf.mol.allAtoms])
                        for conf in conformations
                    ]
                )
                for i in range(3)
            ]
            max_c = [
                max(
                    [
                        max([a.coords[i] for a in conf.mol.allAtoms])
                        for conf in conformations
                    ]
                )
                for i in range(3)
            ]
            min_box_coords = array(params["box_center"]) - array(params["box"]) / 2.0
            max_box_coords = array(params["box_center"]) + array(params["box"]) / 2.0
            min_coords = [min(min_c[i], min_box_coords[i]) for i in range(3)]
            max_coords = [max(max_c[i], max_box_coords[i]) for i in range(3)]
            box = [ceil(max_coords[i] - min_coords[i]) for i in range(3)]
            box_center = [
                round((min_coords[i] + max_coords[i]) / 2.0, 1) for i in range(3)
            ]
        else:
            box = params["box"]
            box_center = params["box_center"]

        number_grid_points = [min(int(x / 0.375) + 1, 126) for x in box]
        autogrid4_param = path.join(dir_path, "autogrid4_param.gpf")
        write_conformation(conformations[0], path.join(dir_path, "temp.pdbqt"))

        # prepare parameter files
        if params["flex"]:
            tmp = deepcopy(conformations[0])
            AD4FlexibleReceptorPreparation(
                MolKitRead(path.join(dir_path, ligand + "_prot.pdbqt"))[0],
                residues=tmp.flex_res,
                flexres_filename=path.join(dir_path, "temp_flex.pdbqt"),
                rigid_filename=path.join(dir_path, "temp_rigid.pdbqt"),
            )
            call(
                "prepare_gpf4.py -l temp.pdbqt"
                + " -p npts="
                + str([(x + x % 2) for x in number_grid_points])[1:-1].replace(" ", "")
                + " -p gridcenter="
                + str(box_center)[1:-1].replace(" ", "")
                + " -r temp_rigid.pdbqt -x temp_flex.pdbqt -o "
                + autogrid4_param,
                shell=True,
                cwd=dir_path,
            )
        else:
            call(
                "prepare_gpf4.py -l temp.pdbqt"
                + " -p npts="
                + str([(x + x % 2) for x in number_grid_points])[1:-1].replace(" ", "")
                + " -p gridcenter="
                + str(box_center)[1:-1].replace(" ", "")
                + " -r "
                + receptor
                + " -o "
                + autogrid4_param,
                shell=True,
                cwd=dir_path,
            )

        call(
            "autogrid4 -p "
            + autogrid4_param
            + " -l "
            + path.join(dir_path, "autogrid.log"),
            shell=True,
            cwd=dir_path,
        )

        # run autodock to rescore
        for conf in conformations:
            write_conformation(conf, path.join(dir_path, "temp.pdbqt"))

            # prepare parameter files
            if params["flex"]:
                AD4FlexibleReceptorPreparation(
                    MolKitRead(path.join(dir_path, ligand + "_prot.pdbqt"))[0],
                    residues=deepcopy(conf).flex_res,
                    flexres_filename=path.join(dir_path, "temp_flex.pdbqt"),
                    rigid_filename=path.join(dir_path, "temp_rigid.pdbqt"),
                )
                call(
                    "prepare_dpf42.py -l temp.pdbqt -r temp_rigid.pdbqt "
                    + "-x temp_flex.pdbqt -o temp.dpf",
                    shell=True,
                    cwd=dir_path,
                )
            else:
                call(
                    "prepare_dpf42.py -l temp.pdbqt -r %s -o temp.dpf" % receptor,
                    shell=True,
                    cwd=dir_path,
                )
            for line in fileinput.input(path.join(dir_path, "temp.dpf"), inplace=True):
                if line.startswith("tran0"):
                    print("epdb")
                    break
                print(line, end="")
            fileinput.close()

            call("autodock4 -p temp.dpf -l temp.dlg", shell=True, cwd=dir_path)
            with open(path.join(dir_path, "temp.dlg"), "r") as dlg:
                conf.binding_energy = float(
                    next(l.split()[8] for l in dlg if "Binding" in l)
                )

    elif function == "Vina":
        for conf in conformations:
            write_conformation(conf, path.join(dir_path, "temp.pdbqt"))
            with open(path.join(dir_path, "temp.log"), "w+") as log:
                if params["flex"]:
                    AD4FlexibleReceptorPreparation(
                        MolKitRead(path.join(dir_path, ligand + "_prot.pdbqt"))[0],
                        residues=deepcopy(conf).flex_res,
                        flexres_filename=path.join(dir_path, "temp_flex.pdbqt"),
                        rigid_filename=path.join(dir_path, "temp_rigid.pdbqt"),
                    )
                    call(
                        "vina --score_only --ligand temp.pdbqt "
                        + "--receptor temp_rigid.pdbqt --flex temp_flex.pdbqt",
                        stdout=log,
                        shell=True,
                        cwd=dir_path,
                    )
                else:
                    call(
                        "vina --score_only --ligand temp.pdbqt --receptor %s"
                        % receptor,
                        stdout=log,
                        shell=True,
                        cwd=dir_path,
                    )
                log.seek(0)
                conf.binding_energy = float(
                    next(l.split()[1] for l in log if "Affinity:" in l)
                )

    elif function == "Smina":
        for conf in conformations:
            write_conformation(conf, path.join(dir_path, "temp.pdbqt"))
            with open(path.join(dir_path, "temp.log"), "w+") as log:
                if params["flex"]:
                    AD4FlexibleReceptorPreparation(
                        MolKitRead(path.join(dir_path, ligand + "_prot.pdbqt"))[0],
                        residues=deepcopy(conf).flex_res,
                        flexres_filename=path.join(dir_path, "temp_flex.pdbqt"),
                        rigid_filename=path.join(dir_path, "temp_rigid.pdbqt"),
                    )
                    # call("smina.static --score_only --ligand temp.pdbqt " + \
                    #     "--receptor temp_rigid.pdbqt --flex temp_flex.pdbqt",
                    #     stdout = log, shell = True, cwd = dir_path)
                    call(
                        "smina --score_only --ligand temp.pdbqt "
                        + "--receptor temp_rigid.pdbqt --flex temp_flex.pdbqt",
                        stdout=log,
                        shell=True,
                        cwd=dir_path,
                    )
                else:
                    # call("smina.static --score_only --ligand temp.pdbqt --receptor %s" % receptor,
                    #     stdout = log, shell = True, cwd = dir_path)
                    call(
                        "smina --score_only --ligand temp.pdbqt --receptor %s"
                        % receptor,
                        stdout=log,
                        shell=True,
                        cwd=dir_path,
                    )
                log.seek(0)
                conf.binding_energy = float(
                    next(line.split()[1] for line in log if "Affinity:" in line)
                )

    elif function == "Vinardo":
        for conf in conformations:
            write_conformation(conf, path.join(dir_path, "temp.pdbqt"))
            with open(path.join(dir_path, "temp.log"), "w+") as log:
                if params["flex"]:
                    AD4FlexibleReceptorPreparation(
                        MolKitRead(path.join(dir_path, ligand + "_prot.pdbqt"))[0],
                        residues=deepcopy(conf).flex_res,
                        flexres_filename=path.join(dir_path, "temp_flex.pdbqt"),
                        rigid_filename=path.join(dir_path, "temp_rigid.pdbqt"),
                    )
                    # call("smina.static --score_only --scoring vinardo --ligand temp.pdbqt " + \
                    #     "--receptor temp_rigid.pdbqt --flex temp_flex.pdbqt",
                    #     stdout = log, shell = True, cwd = dir_path)
                    call(
                        "smina --score_only --scoring vinardo --ligand temp.pdbqt "
                        + "--receptor temp_rigid.pdbqt --flex temp_flex.pdbqt",
                        stdout=log,
                        shell=True,
                        cwd=dir_path,
                    )
                else:
                    # call("smina.static --score_only --scoring vinardo --ligand temp.pdbqt " + \
                    #     "--receptor %s" % receptor, stdout = log, shell = True, cwd = dir_path)
                    call(
                        "smina --score_only --scoring vinardo --ligand temp.pdbqt "
                        + "--receptor %s" % receptor,
                        stdout=log,
                        shell=True,
                        cwd=dir_path,
                    )
                log.seek(0)
                conf.binding_energy = float(
                    next(line.split()[1] for line in log if "Affinity:" in line)
                )

    else:
        print("DincError: The re-scoring function you selected is not supported.")
        terminate(dir_path)


# Load the ligand with MolKitRead and prepare the coordinates for rescoring
# Rescore the conformation
# Record the rescoring results in dincresults.txt
#   ligand:
#   options:
#   receptor_file:
#   params:
#   dir_path:
def run_rescoring(ligand, options, receptor_file, params, dir_path):
    l = MolKitRead(path.splitext(options.ligand)[0] + ".pdbqt")[0]
    l.stoc = StateToCoords(l, [0, 0, 0], 0)
    c = Conformation(l, [0, 0, 0], [0, 0, 0], [1, 0, 0, 0], [0] * ligand.TORSDOF)
    c.getCoords()
    rescore([c], params["scoring"], ligand.name, receptor_file, params, dir_path)
    f = open(path.join(dir_path, "dincresults.txt"), "w")
    f.write("Scoring results from DINC\n\n")
    f.write("Input ligand: " + ligand.name + "\n")
    f.write(
        "Input receptor: " + path.splitext(path.basename(options.receptor))[0] + "\n"
    )
    f.write("Scoring method: " + params["scoring"] + "\n")
    f.write("Binding energy: " + str(c.binding_energy))
    f.write("\n\nThank you for using DINC\n")
    f.close()
