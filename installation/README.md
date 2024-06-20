# Installation instructions

Couple of installation options:
- Install a [mamba/conda](#mamba-install--some-additional-steps) environment.
- [Pull a docker image](#pull-the-docker-image)
- [Build a docker image](#building-the-docker-image)

## Mamba install (+ some additional steps)

0. Enter this folder + make sure you have conda/mamba installed.
1. Create the mamba/conda environment. Run: 
```bash
mamba env create --file ./environment.yml` #(or `conda env create --file ./environment.yml`)
```
2. Activate the environment. Run:
```bash
conda activate dinc-ensemble
```

3. Install dinc-ensemble package. Inside `dinc-ensemble/dinc-ensemble-bin` folder run:
```bash
# install the package (if installed with -e, it is editable)
pip install -e . 
```

4. Test the installation. 

```bash
pytest ./dinc-ensemble
```


## Pull the Docker image

0. Install docker.
1. Run `docker pull ac121/dinc-ensemble:latest`
2. Run the pulled docker image `docker run -e ENV_NAME=DINC-Ensemble -it ac121/dinc-ensemble:latest` 
3. Clone the github repo  `git clone https://github.com/KavrakiLab/dinc-ensemble.git`
4. Install dinc-ensemble (inside `dinc-ensemble/dinc-ensemble-bin` ) `pip install .`
5. Test the installation  (inside `dinc-ensemble/dinc-ensemble-bin` ) `pytest .`

## Building the Docker image

0. Install docker.
1. Enter this folder.
2. Run `docker build . -t dinc-ensemble`
3. Run the built docker image `docker run -e ENV_NAME=dinc-ensemble -it dinc-ensemble` 
4. Clone the github repo  `git clone https://github.com/KavrakiLab/dinc-ensemble.git`
5. Install dinc-ensemble (inside `dinc-ensemble/dinc-ensemble-bin` ) `pip install .`
6. Test the installation  (inside `dinc-ensemble/dinc-ensemble-bin` ) `pytest .`


