from os import path, remove
from math import isnan
import fileinput
from numpy.linalg import norm
from numpy import array
from collections import defaultdict
from copy import deepcopy
import traceback

from openbabel import pybel
from MolKit import Read as MolKitRead, WritePDB
from MolKit.chargeCalculator import GasteigerChargeCalculator
from MolKit.hydrogenBuilder import HydrogenBuilder
from AutoDockTools.atomTypeTools import (
    AutoDock4_AtomTyper,
    LonepairMerger,
    NonpolarHydrogenMerger,
)

from dinc_ensemble.utils import terminate  # or ..utils
from .ligand_fragmentation import make_tor_tree


# Generate a peptide structure in the form of a .pdb file from a fasta sequence.
# Return the name of the .pdb file.
#   file_name: name of the input file containing the fasta sequence
#   dir_path: directory where the docking job is running
#
def seq_to_peptide(file_name, dir_path):
    from Bio.PDB.Atom import Atom
    from Bio.PDB.PDBIO import PDBIO
    from PeptideBuilder.Geometry import geometry
    from PeptideBuilder.PeptideBuilder import (
        calculateCoordinates,
        make_extended_structure,
    )

    # Iterate over the lines of the input file and process them
    # only saving the first peptide in the file
    stop = False
    with open(path.join(dir_path, file_name)) as in_file:
        bio_pdb = PDBIO()
        peptide_name = ""
        for line in in_file:
            # a line starting with > contains the peptide name
            # maybe indicating the first sequence is over
            if line.startswith(">"):
                if stop:
                    break
                stop = True
                peptide_name = line[1:21]
            # a line not starting with > contains the amino acid sequence
            else:
                AA_chain = line[:-1]
                structure = make_extended_structure(AA_chain)
                # add the missing OXT atom to the last residue
                res = list(structure.get_residues())[-1]
                geo = geometry(AA_chain[-1])
                oxt = calculateCoordinates(
                    res["N"],
                    res["CA"],
                    res["C"],
                    geo.C_O_length,
                    geo.CA_C_O_angle + 120,
                    380.0,
                )
                res.add(Atom("OXT", oxt, 0.0, 1.0, " ", " OXT", 0, "O"))
                bio_pdb.set_structure(structure)
                bio_pdb.save(path.join(dir_path, peptide_name[:-1] + ".pdb"))

    return peptide_name[:-1] + ".pdb"


# Read the given ligand input file (in fasta, pdb or mol2 format), do the required conversions,
# and perform some preprocessing on the ligand, based on the given parameters:
#  - check ligand size, check atom types and coordinates (always),
#  - look for steric clashes and non-bonded atoms, rename atoms (always),
#  - add hydrogens and charges, and merge atoms (if requested by the user).
# Return the ligand molecule, a dictionary matching each atom to its bonded atoms,
#   an array to keep track of the original order of atoms,
#   and a dictionary to trace back the original names of atoms.
#
def prepare_ligand(file_name, params):
    print("Read ligand file:", path.abspath(file_name))
    dir_path = path.dirname(path.abspath(file_name))

    # convert fasta sequence to peptide pdb file, if needed
    if file_name.endswith(".fasta"):
        file_name = seq_to_peptide(file_name, dir_path)
        print("Converted fasta file to", path.abspath(file_name))

    # convert pdb file to mol2 file, if needed
    if file_name.endswith(".pdb"):
        # remove lines that do not contain typical PDB entries (including empty lines)
        for line in fileinput.input(path.join(dir_path, file_name), inplace=True):
            if (
                line.startswith("ATOM")
                or line.startswith("HETATM")
                or line.startswith("CONECT")
            ):
                print(line, end="")
        fileinput.close()
        try:
            molecule = next(pybel.readfile("pdb", path.join(dir_path, file_name)))
            # add missing hydrogen atoms, if needed
            if params["prepare_ligand"]:
                molecule.OBMol.AddHydrogens()
            file_name = file_name[:-3] + "mol2"
            molecule.write("mol2", file_name, overwrite=True)
        except Exception:
            print(
                "DincError: Could not read this ligand pdb file. Is it correctly formatted?"
            )
            terminate(dir_path)

    # read the mol2 file and build the ligand molecule described in this file
    try:
        ligands = MolKitRead(file_name)
        ligand = ligands[0]
        heavy_atoms = ligand.allAtoms.get(lambda a: a.element != "H")

        # check the ligand's size
        if len(heavy_atoms) > 1000:
            print("DincError: Unfortunately, DINC cannot process such a large ligand.")
            terminate(dir_path)

        # when using AutoDock 4, check that the types of the ligand's atoms are valid
        if (
            params["sampling"] == "AD4"
            or params["scoring"] == "AD4"
            or params["rescoring"] == "AD4"
        ):
            for a in ligand.allAtoms:
                if a.element not in ["Br", "C", "Cl", "F", "H", "N", "O", "P", "S"]:
                    print(
                        "DincError: DINC cannot process this ligand using AutoDock 4."
                    )
                    print("DincError: Unknown atom", a.name, a.element)
                    if "H" in a.name:
                        print(
                            "DincError: Try again after removing the ligand's hydrogen atoms."
                        )
                    terminate(dir_path)

        # check that the coordinates of the ligand's atoms are valid
        if any(
            isnan(coord) for coord in [c for a in ligand.allAtoms for c in a.coords]
        ):
            print("DincError: Some ligand atoms have invalid coordinates!")
            terminate(dir_path)
        if all(coord == 0.0 for coord in [c for a in heavy_atoms for c in a.coords]):
            print(
                "DincError: All the coordinates of this ligand's atoms are equal to zero!"
            )
            terminate(dir_path)

        # check that the ligand does not contain any non-bonded heavy atom
        for atom in heavy_atoms:
            if not [b for b in atom.bonds if b.neighborAtom(atom).element != "H"]:
                print(
                    "DincError: This ligand contains non-bonded atoms. Please check your file."
                )
                terminate(dir_path)

        # check that the ligand does not contain any steric clash
        atomList = list(heavy_atoms)
        while atomList:
            atom = atomList.pop(0)
            for at in atomList:
                if at not in [b.neighborAtom(atom) for b in atom.bonds]:
                    dist = norm(array(atom.coords) - array(at.coords))
                    if dist < 1.2 * (atom.bondOrderRadius + at.bondOrderRadius):
                        print(
                            "DincError: This ligand contains steric clashes. Please check your file."
                        )
                        if all(
                            coord == 0.0 for coord in [a.coords[2] for a in heavy_atoms]
                        ):
                            print(
                                "DincError: This ligand has a flat and not a proper 3D structure."
                            )
                        terminate(dir_path)

        # if it was requested by the user, chemical preparation of the ligand is performed:
        # 1) add hydrogen atoms (only if no hydrogen is there); 2) remove lone pairs;
        # 3) add Gasteiger charges; 4) merge charges and remove non-polar hydrogens
        if params["prepare_ligand"]:
            if not ligand.allAtoms.get(lambda a: a.element == "H"):
                try:
                    HydrogenBuilder().addHydrogens(ligand)
                except Exception:
                    print("DincError: Unfortunately, DINC cannot process this ligand.")
                    print(
                        "DincError: An error occurred when trying to add hydrogen atoms."
                    )
                    terminate(dir_path)
            LonepairMerger().mergeLPS(ligand.allAtoms)
            GasteigerChargeCalculator().addCharges(ligand.allAtoms)
            NonpolarHydrogenMerger().mergeNPHS(ligand.allAtoms)

        # set AutoDock's element types for all the ligand's atoms
        try:
            AutoDock4_AtomTyper().setAutoDockElements(ligand, reassign=True)
        except AssertionError:
            print("DincError: Unfortunately, DINC cannot process this ligand.")
            print(
                "DincError: An error occurred when assigning AutoDock elements to atoms."
            )
            terminate(dir_path)

        # deal with docking cases where fragment decomposition is defined automatically
        # NB: the smallest frag_size is 6, and the largest frag_new is 3
        if params["job_type"] != "scoring" and params["frag_mode"] == "automatic":
            # print(ligand)
            ligand_copy = deepcopy(ligand)
            make_tor_tree(ligand_copy, heavy_atoms[0].name)
            params["frag_new"] = min(max(ligand_copy.torscount - 6, 0), 3)
            params["frag_size"] = ligand_copy.torscount - params["frag_new"]

        # make the information about the ligand's atoms homogeneous
        for a in ligand.allAtoms:
            a.parent.parent.id = "X"  # create a proper chain id
            if len(a.parent.type) > 3:
                a.parent.type = a.parent.type[:3]  # correct the residue type
            if "occupancy" in a.__dict__:
                del a.occupancy  # delete occupancy
            if "temperatureFactor" in a.__dict__:
                del a.temperatureFactor  # delete temperature

        # rename the ligand's atoms to ensure name uniqueness
        element_id = defaultdict(int)
        name_back_trace = defaultdict(str)
        atom_order = []
        for a in ligand.allAtoms:
            element_id[a.element] += 1
            new_name = a.element + str(element_id[a.element])
            name_back_trace[new_name] = a.name
            atom_order.append(new_name)
            a.name = new_name

        # create a dictionary listing, for each ligand's atom, the atoms with which it shares a bond
        bonded_atoms = defaultdict(list)
        for atom in ligand.allAtoms:
            bonded_atoms[atom.name] = [b.neighborAtom(atom).name for b in atom.bonds]

        return (ligand, bonded_atoms, name_back_trace, atom_order)

    except Exception as e:
        print(
            "DincError: Could not read this ligand mol2 file. Is it correctly formatted?"
        )
        print(e)
        traceback.print_exc()
        terminate(dir_path)


# Write the given conformation into a file having the given name, optionally using
# the given back_trace to replace the names of the ligand's atoms by the original ones.
#   atom_order: order of atoms in input ligand
#
def write_conformation(conf, file_name, atom_order=[], back_trace=None):
    if back_trace:
        conformation = deepcopy(conf)
        for atom in conformation.mol.allAtoms:
            atom.segID = ""
        conformation.mol.allAtoms.updateCoords(conf.coords, 0)
        WritePDB(file_name + "pre", conformation.mol)
        # sort the pdb file and replace atom names with original names
        # this may not be the most efficient way right now but it sure seems to work
        unsorted_lines = open(file_name + "pre", "r").readlines()
        with open(file_name, "w") as f:
            f.write(unsorted_lines[0])
            for a in atom_order:
                newa = back_trace[a]
                # maintain the correct pdb column format
                if len(a) == 4:
                    start = 0
                else:
                    start = 1
                if len(newa) == 4:
                    newstart = 0
                else:
                    newstart = 1
                for line in unsorted_lines:
                    if line[12 + start : 12 + start + len(a) + 1] == a + " ":
                        linelist = list(line)
                        replacement = list("    ")
                        replacement[newstart : newstart + len(newa)] = newa
                        linelist[12:16] = replacement
                        f.write(
                            "".join(linelist),
                        )
                        break
            f.write(unsorted_lines[-1])
        remove(file_name + "pre")

    else:
        for i in range(len(conf.mol.parser.allLines)):
            if conf.mol.parser.allLines[i][-1:] != "\n":
                conf.mol.parser.allLines[i] += "\n"
            if conf.mol.parser.allLines[i].startswith("BEGIN_RES"):
                conf.mol.parser.allLines = conf.mol.parser.allLines[:i]
                break
        conf.mol.parser.write_with_new_coords(conf.coords, file_name)
