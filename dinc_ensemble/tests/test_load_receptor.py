import pytest
from dinc_ensemble import load_receptor
#from MolKit.protein import Protein as MolKitProtein
from dinc_ensemble.receptor.core import DINCReceptor

def test_load_receptor_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_receptor("nonexistent_file.pdb")

def test_load_receptor_no_loader_found():
    with pytest.raises(IOError):
        load_receptor("receptor.xyz")

def test_load_receptor_pdb(test_directory):
    filename = test_directory + "/data/small_tests_data/receptor.pdb"
    receptor = load_receptor(filename)
    assert isinstance(receptor, DINCReceptor)

def test_load_receptor_pdbqt(test_directory):
    filename = test_directory + "/data/small_tests_data/receptor.pdbqt"
    receptor = load_receptor(filename)
    assert isinstance(receptor, DINCReceptor)