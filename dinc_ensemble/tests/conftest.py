import pytest
import os
from glob import glob
from dinc_ensemble import load_ligand, load_receptor

@pytest.fixture(scope="session", autouse=True)
def test_directory():
    return os.path.dirname(__file__)

@pytest.fixture(scope="session", autouse=True)
def cli_main(test_directory):
    data_path = os.path.join(test_directory, 
                             "../scripts/dinc_ensemble_run.py")
    return data_path

@pytest.fixture(scope="session", autouse=True)
def data_directory(test_directory):
    data_path = os.path.join(test_directory, "data")
    return data_path

@pytest.fixture(scope="session", autouse=True)
def ligand(data_directory):
    print("Initializing ligand")
    ligand_path = os.path.join(data_directory, "./small_tests_data/ligand.mol2")
    return load_ligand(ligand_path)

@pytest.fixture(scope="session", autouse=True)
def ligands_0dof(data_directory):
    print("Initializing ligands 1d0f")
    lig_path = os.path.join(data_directory, "./pdbbind_test_ligands/*dof_0.mol2")
    mol2_files = glob(lig_path)
    ligs = []
    for file in mol2_files:
        ligs.append(load_ligand(file))
    return ligs

@pytest.fixture(scope="session", autouse=True)
def ligands_1dof(data_directory):
    print("Initializing ligands 1d0f")
    lig_path = os.path.join(data_directory, "./pdbbind_test_ligands/*dof_1.mol2")
    mol2_files = glob(lig_path)
    ligs = []
    for file in mol2_files:
        ligs.append(load_ligand(file))
    return ligs

@pytest.fixture(scope="session", autouse=True)
def ligands_5dof(data_directory):
    print("Initializing ligands 1d0f")
    lig_path = os.path.join(data_directory, "./pdbbind_test_ligands/*dof_5.mol2")
    mol2_files = glob(lig_path)
    ligs = []
    for file in mol2_files:
        ligs.append(load_ligand(file))
    return ligs

@pytest.fixture(scope="session", autouse=True)
def ligands_10dof(data_directory):
    print("Initializing ligands 1d0f")
    lig_path = os.path.join(data_directory, "./pdbbind_test_ligands/*dof_10.mol2")
    mol2_files = glob(lig_path)
    ligs = []
    for file in mol2_files:
        ligs.append(load_ligand(file))
    return ligs

@pytest.fixture(scope="session", autouse=True)
def ligands_20dof(data_directory):
    print("Initializing ligands 1d0f")
    lig_path = os.path.join(data_directory, "./pdbbind_test_ligands/*dof_20.mol2")
    mol2_files = glob(lig_path)
    ligs = []
    for file in mol2_files:
        ligs.append(load_ligand(file))
    return ligs

@pytest.fixture(scope="session", autouse=True)
def receptor(data_directory):
    print("Initializing receptor")
    ligand_path = os.path.join(data_directory, "./small_tests_data/receptor.pdb")
    return load_receptor(ligand_path)

