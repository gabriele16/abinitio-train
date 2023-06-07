# abinitio-train
An end-to-end workflow to train neural network potentials from ab initio molecular dynamics simulations and run molecular dynamics simulations.
This workflow is for now meant to be used with CP2K and Allegro/NequIP and it is mainly intended for private use, at least for now.

## Installation

abinitio-train can be installed from source by:
```
git clone git@github.com:gabriele16/abinitio-train.git
cd abinitio-train
pip install .
```

The following packages, listed in `requirements.txt`, will be installed, in addition to the development versions nequip == 0.6.0 and allegro == 0.2.0:
* numpy
* pandas
* ase
* torch==1.13
* wandb==0.14.2
* mdanalysis==2.4.3
* nglview==3.0.4
* jedi==0.18.2
* gdown==4.7.1

## Usage

The workflow is run by executing `abinitio-train-workflow`. To check the possible options you can run `abinitio-train-workflow -h`.

Inside the directory `./train_workflow` there are some example bash scripts to run either the entire workflow, from pre-processing data, to training to running MD with CP2K, or single steps:
* `run_workflow.sh` runs the whole workflow, trains a NN potential with `nequip`, deploys the model to get a serialized .PTH model file and then runs MD with CP2K
* `run_deploy.sh` only executes the deployment of the model
* `run_cp2k.sh` runs both deployment and MD with CP2K

