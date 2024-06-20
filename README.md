# dinc-ensemble: docking incrementally to an ensemble of receptors 

`dinc-ensemble` is a toolkit for molecular docking. It provides a meta-docking incremental docking approach with an ensemble of input receptors. Currently implemented docking engine is Vina, but can be extended to other docking approaches.

### Folder structure

- `dinc_ensemble`
The python package with all the modules.

- `installation`
Different install options and files including conda/mamba install, pulling or building a docker image.

-  `tutorial-notebooks`
Jupyter notebook tutorials showcasing dinc-ensemble use as a python package and different modules. 

-  `sample-data`
Contains some minimal-example data that can be used to run and test `dinc-ensemble`.

### Install with mamba

- Conda install
1. Pull the github repo.

2. Mamba install (start from this directory).

```bash
mamba env create --file ./installation/environment.yml
conda activate dinc-ensemble
pip install -e . 
```

3. Test the installation.

- Run tests
```bash
pytest dinc_ensemble
```
- Run the CLI
```bash
dinc-ensemble --help
```
Check more details for installing dinc-ensemble [here](https://github.com/KavrakiLab/dinc-ensemble/tree/main/installation).

### Run dinc-ensemble CLI

- Try running the command-line interface. 
`dinc-ensemble dock {LIGAND_PATH} [{RECEPTOR_PATH}] {OUTPUT_DIR}`
- 1OHR xample:
`dinc-ensemble dock ./sample-data/small_tests_data/1ohr_ligand.mol2 ./sample-data/small_tests_data/1ohr_receptor.pdb ./tmp_test_out --replica-num 1`


