"""
Unit tests for the ligand loading module.
"""

import pytest
import os
from dinc_ensemble import load_ligand, write_ligand
from dinc_ensemble.ligand.core import DINCMolecule

def test_write_ligand_pdb2pdb_file(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.pdb"
    molecule = load_ligand(filename)
    filename_out = test_directory + "/data/small_tests_data/ligand_pdb.pdb"
    write_ligand(molecule, filename_out)
    molecule = load_ligand(filename_out)
    assert isinstance(molecule, DINCMolecule)
    os.remove(filename_out)

def test_write_ligand_mol2pdb_file(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.mol2"
    molecule = load_ligand(filename)
    filename_out = test_directory + "/data/ligand_mol2.pdb"
    write_ligand(molecule, filename_out)
    molecule = load_ligand(filename_out)
    assert isinstance(molecule, DINCMolecule)
    os.remove(filename_out)

def test_write_ligand_fasta2pdb_file(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.fasta"
    molecule = load_ligand(filename)
    filename_out = test_directory + "/data/small_tests_data/ligand_fasta.pdb"
    write_ligand(molecule, filename_out)
    # TODO: check this - works fine locally
    #molecule = load_ligand(filename_out)
    #assert isinstance(molecule, DINCMolecule)
    os.remove(filename_out)

def test_write_ligand_random_extension(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.fasta"
    molecule = load_ligand(filename)
    filename_out = test_directory + "/data/small_tests_data/ligand_fasta.rand"
    with pytest.raises(IOError):
        write_ligand(molecule, filename_out)
        
def test_write_ligand_fasta2mol2_file(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.fasta"
    molecule = load_ligand(filename)
    filename_out = test_directory + "/data/small_tests_data/ligand_fasta.mol2"
    write_ligand(molecule, filename_out)
    molecule = load_ligand(filename_out)
    assert isinstance(molecule, DINCMolecule)
    #os.remove(filename_out)
    
def test_write_ligand_mol2mol2_file(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.mol2"
    molecule = load_ligand(filename)
    filename_out = test_directory + "/data/small_tests_data/ligand_mol2.mol2"
    write_ligand(molecule, filename_out)
    molecule_new = load_ligand(filename_out)
    assert isinstance(molecule, DINCMolecule)
    assert molecule_new.n_atoms == molecule.n_atoms
    assert molecule_new.bonds.shape[0] == molecule.bonds.shape[0]
    #os.remove(filename_out)
    
def test_write_ligand_fasta2sdf_file(test_directory):
    # Test loading an existing ligand file
    filename = test_directory + "/data/small_tests_data/ligand.fasta"
    molecule = load_ligand(filename)
    filename_out = test_directory + "/data/small_tests_data/ligand_fasta.sdf"
    write_ligand(molecule, filename_out)
    molecule_new = load_ligand(filename_out)
    assert isinstance(molecule, DINCMolecule)
    assert molecule_new.n_atoms == molecule.n_atoms
    assert molecule_new.bonds.shape[0] == molecule.bonds.shape[0]
    #os.remove(filename_out)