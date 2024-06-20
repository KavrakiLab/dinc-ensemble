from __future__ import annotations

from Bio.PDB.Atom import Atom
from Bio.PDB.PDBIO import PDBIO
from PeptideBuilder.Geometry import geometry
from PeptideBuilder.PeptideBuilder import calculateCoordinates, make_extended_structure

from os import path
from datetime import date

from openbabel import pybel

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.molecule import DINCMolecule

# Generate a peptide structure in the form of a .pdb file from a fasta sequence.
# Return the name of the .pdb file.
#   file_name: name of the input file containing the fasta sequence
#
def seq_to_peptide(file_name):

    """
    Convert a sequence of amino acids given in FASTA format into a PDB file with the given peptide.

    This function reads a file specified by the file_name parameter, which contains a sequence of amino acids in FASTA format. It processes the file and converts the sequence into a PDB file representing the peptide.

    Parameters
    ----------
    file_name : str
        The path to the input file containing the amino acid sequence in FASTA format.

    Returns
    -------
    str
        The path to the generated PDB file.

    Notes
    -----
    This function follows the following steps to convert the sequence into a PDB file:
    1. Iterate over the lines of the input file and process them, only saving the first peptide in the file.
    2. Identify the peptide name from a line starting with ">".
    3. Extract the amino acid sequence from a line not starting with ">".
    4. Create an extended structure from the amino acid sequence.
    5. Add the missing OXT atom to the last residue.
    6. Set the structure in the PDBIO object.
    7. Save the structure as a PDB file with the peptide name as the filename.

    The function uses the PDBIO class to handle the creation and saving of the PDB file.

    """
    # TODO: refine this function and test it properly
    # Iterate over the lines of the input file and process them
    # only saving the first peptide in the file
    stop = False
    with open(path.join(file_name)) as in_file:
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
                if '\n' in line:
                    AA_chain = line[:-1]
                else:
                    AA_chain = line

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
                bio_pdb.save(path.join(peptide_name[:-1] + ".pdb"))
    return peptide_name[:-1] + ".pdb"

def pybel_convert(filename1, format1, filename2, format2, overwrite = True) -> None:
    
    pybel_mol = next(pybel.readfile(format1, filename1))
    pybel_mol.write(format2, filename2, overwrite=overwrite)



    
