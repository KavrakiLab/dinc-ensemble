"""
Unit tests for the speed of ligand core module.
"""

import pytest
import os
from dinc_ensemble import load_ligand
from dinc_ensemble.ligand.core import DINCMolecule
import timeit
import pandas as pd

def test_load_ligand_existing_pdb_file(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.fasta"
    molecule = load_ligand(filename)
    # number of atoms here ~128
    print(molecule.n_atoms)
    # here we have a list of atom dataclasses
    atom_datac = list(molecule._atoms)
    # here we have a list of dictionaries
    atom_dict = [atom.__dict__ for atom in molecule._atoms]
    # here we time both versions 
    time_datac = timeit.timeit(lambda: pd.DataFrame(atom_datac), number=1)
    time_dict = timeit.timeit(lambda: pd.DataFrame(atom_dict), number=1)
    print("Time to convert a list of dataclasses into a dataframe: {}".format(time_datac))
    print("Time to convert a list of dictionaries into a dataframe: {}".format(time_dict))
    assert time_datac > time_dict