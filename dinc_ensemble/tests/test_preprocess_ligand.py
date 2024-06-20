import pytest
from dinc_ensemble import PREPROCESS_MOLECULE_OPTIONS, preprocess_molecule
from itertools import combinations

def test_preprocess_options(ligand):
    print("Available options for preprocessing")
    print(PREPROCESS_MOLECULE_OPTIONS)
    # check the number of preprocessing options registered is correct
    preprocess_options_number = len(PREPROCESS_MOLECULE_OPTIONS) 
    assert preprocess_options_number == 5
    # try out all different preprocessing option combinations, make sure that they all run
    all_preprocess_options = []
    for i in range(1, preprocess_options_number+1):
        all_preprocess_options.extend(list(combinations(PREPROCESS_MOLECULE_OPTIONS, i)))
    print(all_preprocess_options)
    # no parameters
    preprocess_molecule(ligand)
    # all parameters options
    for preprocess_option in all_preprocess_options:
        option_dict = {x:1 for x in preprocess_option}
        print(option_dict)
        preprocess_molecule(ligand, **option_dict)
    print(ligand)
    

def test_preprocess_wrong_option(ligand):
    with pytest.raises(KeyError):
        preprocess_molecule(ligand, random_option = 1)

    