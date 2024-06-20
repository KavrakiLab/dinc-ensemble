from collections import defaultdict
from os import path
from subprocess import call
from numpy import array, absolute
import shutil
import fileinput


from MolKit import Read as MolKitRead
from AutoDockTools.MoleculePreparation import AD4FlexibleReceptorPreparation

from dinc_ensemble.utils import terminate


# Prepare the receptor and the binding box for docking (and update the docking parameters).
# Return the name of the pdbqt file containing the receptor's description.
#   receptor_file: name of the file containing the protein receptor
#   ligand: processed ligand molecule
#   params (modified): parameters of the docking job
#   dir_path: directory where the docking job is running
#
def prepare_receptor_and_box(receptor_file, ligand, params, dir_path):
    print("Read receptor file:", path.abspath(receptor_file))
    receptor = None
    try:
        receptor = MolKitRead(receptor_file)[0]
    except Exception:
        print(
            "DincError: Could not parse this receptor file. Are you sure it is correctly formatted?"
        )
        terminate(dir_path)

    receptor_pdbqt = path.join(dir_path, ligand.name + "_prot.pdbqt")
    if receptor_file.endswith(".pdbqt"):
        shutil.copy(receptor_file, receptor_pdbqt)
    else:
        # use the "prepare_receptor4.py" script from the Utilities24 of AutoDockTools:
        # + explicit parameter: -r <receptor_filename> (supported file types: pdb, mol2, etc)
        # + explicit parameter: -o <pdbqt_filename> (output file)
        # + explicit parameter: -A checkhydrogens  -->  add hydrogens only if there are none already
        # + implicit parameter: add Gasteiger charges
        # + implicit parameter: -U nphs_lps_waters_nonstdres
        #   - nonstdres: remove chains composed entirely of non-standard amino acids
        #   - waters: remove water molecules
        #   - lps: merge charges and remove lone pairs
        #   - nphs: merge charges and remove non-polar hydrogens
        print("Prepare receptor for docking")
        call(
            "prepare_receptor4.py -r "
            + receptor_file
            + " -A checkhydrogens -o "
            + receptor_pdbqt,
            shell=True,
            cwd=dir_path,
        )

    if params["flex"]:
        (
            rigid_file,
            flex_file,
        ) = set_flex_res(receptor_pdbqt, params, dir_path)
    else:
        rigid_file = receptor_pdbqt
        flex_file = None

    print("Binding box center:", end=" ")
    if params["box_center_type"] == "ligc":
        params["box_center"] = ligand.getCenter()
        print("ligand center =", end=" ")
    elif params["box_center_type"] == "protc":
        params["box_center"] = receptor.getCenter()
        print("protein center =", end=" ")
    else:
        print("user-specified -->", end=" ")
    print(params["box_center"], "angstrom")

    print("Binding box dimensions:", end=" ")
    if not params["box"]:
        params["box_type"] = "ligand-based"
        print("ligand-based -->", end=" ")
        min_c = [min([a.coords[i] for a in ligand.allAtoms]) for i in range(3)]
        max_c = [max([a.coords[i] for a in ligand.allAtoms]) for i in range(3)]
        params["box"] = [(max_c[i] - min_c[i] + 10) for i in range(3)]
    else:
        params["box_type"] = "user-specified"
        print("user-specified -->", end=" ")
    print(params["box"], "angstrom")

    # check that the binding box and the receptor intersect
    receptor_in_box = False
    for a in receptor.allAtoms:
        if (
            max(
                absolute(array(a.coords) - array(params["box_center"]))
                - array(params["box"]) / 2
            )
            < 0
        ):
            receptor_in_box = True
            break
    if not receptor_in_box:
        print(
            "DincError: The binding box you provided does not intersect with the receptor."
        )
        if params["box_center_type"] == "ligc":
            print(
                "DincError: It is centered on the ligand, which is too far from the receptor."
            )
        print("DincError: box center =", params["box_center"], "angstrom")
        print("DincError: box dimensions =", params["box"], "angstrom")
        terminate(dir_path)

    # when using AutoDock4, create the file containing the grid parameters
    if params["sampling"] == "AD4":
        number_grid_points = [min(int(x / 0.375) + 1, 126) for x in params["box"]]
        autogrid4_param = path.join(dir_path, "autogrid4_param.gpf")
        if params["flex"]:
            call(
                "prepare_gpf4.py -l "
                + path.join(dir_path, ligand.name + ".pdbqt")
                + " -p npts="
                + str([(x + x % 2) for x in number_grid_points])[1:-1].replace(" ", "")
                + " -p gridcenter="
                + str(params["box_center"])[1:-1].replace(" ", "")
                + " -r "
                + rigid_file
                + " -x "
                + flex_file
                + " -o "
                + autogrid4_param,
                shell=True,
                cwd=dir_path,
            )
        else:
            call(
                "prepare_gpf4.py -l "
                + path.join(dir_path, ligand.name + ".pdbqt")
                + " -p npts="
                + str([(x + x % 2) for x in number_grid_points])[1:-1].replace(" ", "")
                + " -p gridcenter="
                + str(params["box_center"])[1:-1].replace(" ", "")
                + " -r "
                + rigid_file
                + " -o "
                + autogrid4_param,
                shell=True,
                cwd=dir_path,
            )
        print("Call autogrid4 and create log file: autogrid.log")
        call(
            "autogrid4 -p "
            + autogrid4_param
            + " -l "
            + path.join(dir_path, "autogrid.log"),
            shell=True,
            cwd=dir_path,
        )

    return rigid_file, flex_file


# Set relevant side-chains to flexible and create the necessary receptor files for flexible docking.
# Return the names of the files containing the flexible and rigid parts of the receptor,
#      and the dictionary listing which atoms are bonded to which in flexible side-chains.
#   receptor_pdbqt: name of the pdbqt file containing the prepared protein receptor
#   params: parameters of the docking job (including the list of flexible side-chains)
#   dir_path: directory where the docking job is running
#
def set_flex_res(receptor_pdbqt, params, dir_path):
    receptor = MolKitRead(receptor_pdbqt)[0]
    flex_res = []
    if params["flex_res"] == None:
        res_codes = []
    else:
        res_codes = params["flex_res"].split(",")

    # assuming each code is chain:residue
    flex_bonded_atoms = {}
    for res in res_codes:
        [chain_code, res_code] = res.split(":")
        chain = receptor.chains[receptor.chains.id.index(chain_code)]
        residue = chain.residues[chain.residues.number.index(res_code)]
        residue.buildBondsByDistance()
        flex_res.append(residue)
        bonded_atoms = defaultdict(list)
        for atom in residue.atoms:
            bonded_atoms[atom.name] = [
                b.neighborAtom(atom).name
                for b in atom.bonds
                if b.neighborAtom(atom).name not in ["C", "O", "N"]
            ]
        flex_bonded_atoms[res] = bonded_atoms

    AD4FlexibleReceptorPreparation(
        receptor,
        residues=flex_res,
        flexres_filename=path.join(dir_path, receptor.name + "_flex.pdbqt"),
        rigid_filename=path.join(dir_path, receptor.name + "_rigid.pdbqt"),
    )

    return receptor.name + "_rigid.pdbqt", receptor.name + "_flex.pdbqt"


# Remove flexible residues from the given .dlg file, when docking with AD4,
# so that they are not treated as part of the ligand.
# Return the new file name with the flexible residues removed.
#   params: parameters of the docking job
#
def remove_flex_from_output(docking_file, params):
    if params["sampling"] == "AD4":
        # this filename can't end with .dlg
        new_file = docking_file + "_removed_flex"

        f = open(new_file, "w")

        found_res = False
        for line in fileinput.input(docking_file):
            if line.startswith("DOCKED: TER"):
                f.write(
                    line,
                )
                found_res = False
                continue

            if found_res:
                continue

            if line.startswith("DOCKED: BEGIN_RES"):
                found_res = True
                continue

            f.write(
                line,
            )

        fileinput.close()
        return new_file
