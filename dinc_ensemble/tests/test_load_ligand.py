"""
Unit tests for the ligand loading module.
"""

import pytest
import os
from dinc_ensemble import load_ligand
from dinc_ensemble.ligand.core import DINCMolecule

def test_load_ligand_existing_pdb_file(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.pdb"
    molecule = load_ligand(filename)
    assert isinstance(molecule, DINCMolecule)

def test_load_ligand_existing_fasta_file(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.fasta"
    molecule = load_ligand(filename)
    assert isinstance(molecule, DINCMolecule)

def test_load_ligand_existing_mol2_file(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.mol2"
    molecule = load_ligand(filename)
    assert isinstance(molecule, DINCMolecule)

def test_load_ligand_nonexistent_file():
    # Test loading a non-existent ligand file
    filename = "nonexistent.pdb"
    with pytest.raises(FileNotFoundError):
        load_ligand(filename)

def test_load_ligand_unknown_extension():
    # Test loading a ligand file with an unknown extension
    filename = "ligand.xyz"
    with pytest.raises(IOError):
        load_ligand(filename)
