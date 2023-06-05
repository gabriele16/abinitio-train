import sys
import os
import warnings
import subprocess
import matplotlib.pyplot as plt
import argparse
from argparse import RawTextHelpFormatter
import torch
import numpy as np
import pandas as pd
from ase.io import read, write
from ase import Atoms
from ase.io.trajectory import Trajectory
import nequip
from nequip.utils import Config
import allegro
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
sys.path.append('../')
from utils.utils import *

description = """pipeline to train NequIP or Allegro potential and run MD with CP2K
"""

def parse_arguments():

   arg_parser = argparse.ArgumentParser(
       description=description, formatter_class=RawTextHelpFormatter)
   arg_parser.add_argument('--train',
                           help='perform training',
                           action="store_true")
   arg_parser.add_argument('--no-train', dest='train',
                           action="store_false")
   arg_parser.add_argument('--data_load',
                           help='load data',
                           action="store_true")
   arg_parser.add_argument('--no-data_load',
                           dest='data_load',
                           action="store_false")
   arg_parser.add_argument('--run_md',
                           help="run md with cp2k",
                           action="store_true")
   arg_parser.add_argument('--no-run_md',
                           dest='run_md',
                           action="store_false")
   arg_parser.add_argument('--analysis',
                           help='perform analysis with mdanalysis',
                           action="store_true")
   arg_parser.add_argument('--no-analysis',
                           dest='analysis',
                           action="store_false")
   arg_parser.add_argument('--method',
                           default='nequip',
                           choices = ["nequip","allegro"])
   arg_parser.add_argument('--data_dir',
                           default = "./",
                           help="directory where training set is")
   arg_parser.add_argument('--resultsdir',
                           default = "results",
                           help="directory where training is performed")
   arg_parser.add_argument('--forces',
                           required = False,
                           default = "traj-frc-1.xyz",
                           help="not compulsory traj-frc-1.xyz' is default")
   arg_parser.add_argument('--positions',
                           required = False,
                           default = "traj-pos-1.xyz",
                           help="not compulsory, use 'traj-pos-1.xyz' as default")
   arg_parser.add_argument('--dataset',
                           required = False,
                           default = "dataset.extxyz",
                           help="use 'dataset.extxyz' as default")
   arg_parser.add_argument('--system_name',
                           required = True,
                           default = "system_name",
                           help="compulsory, use 'system_name' as default")
   arg_parser.add_argument('--cutoff',
                           required = False,
                           default = 5.0,
                           help="cutoff value")
   arg_parser.add_argument('--polynomial_cutoff_p',
                           required = False,
                           default = 48,
                           help="polynomial cutoff p value")
   arg_parser.add_argument('--l_max',
                           required = False,
                           default = 2,
                           help="l_max")
   arg_parser.add_argument('--num_layers',
                           required = False,
                           default = 2,
                           help="number of layers")
   arg_parser.add_argument('--num_features',
                           required = False,
                           default = 32,
                           help="number of features")
   arg_parser.add_argument('--max_epochs',
                           required = False,
                           default = 10000,
                           help="maximum number of epochs")
   arg_parser.add_argument('--n_train',
                           required = False,
                           default = 1000,
                           help="size of training set")
   arg_parser.add_argument('--n_val',
                           required = False,
                           default = 100,
                           help="size of validation set")
   arg_parser.add_argument('--default_dtype',
                           required = False,
                           default = "float64",
                           choices = ["float64", "float32"],
                           help="use single or double precision")

   return arg_parser.parse_args()

def print_run_info():

   print("Check Pytorch and whether it works correctly on the GPU")
   print("*****************************")
   print("torch version: ", torch.__version__)
   print("*****************************")
   print("cuda is available: ", torch.cuda.is_available())
   print("*****************************")
   if torch.cuda.is_available():
      print("cuda version:")
      subprocess.call("nvcc --version",shell=True)
      print("*****************************")
      print("check which GPU is being used:")
      subprocess.call("nvidia-smi",shell=True)
      print("*****************************")
      print("check path to nvcc:")
      subprocess.call("which nvcc", shell=True)
      print("*****************************")
   print("check which NequIP version is being used: ", nequip.__version__)
   print("*****************************")
   print("check which Allegro version is being used: ", allegro.__version__)

def main():

   args = parse_arguments()
   method = args.method
   data_dir = args.data_dir
   resultsdir = args.resultsdir
   data_pos = args.positions
   data_frc = args.forces
   dataset = args.dataset
   system_name = args.system_name
   #options of the model
   cutoff_value = args.cutoff
   polynomial_cutoff_p_value = args.polynomial_cutoff_p
   num_layers_value = args.num_layers
   num_features_value = args.num_features
   l_max_value = args.l_max
   default_dtype_value = args.default_dtype
   #training options
   n_train_value = args.n_train
   n_val_value = args.n_val
   max_epochs_value = args.max_epochs

   print_run_info()
   
   np.random.seed(0)
   if torch.cuda.is_available():
     torch.cuda.manual_seed(0)
   else:
     torch.manual_seed(0)
   
   if args.train and method == "allegro":
      dataset = data_dir+'/'+dataset
      conf = sort(read(dataset))
      symbols_list = re.findall(r'[a-zA-Z]', str(conf.symbols))
      allegro_input = generate_allegro_input(resultsdir=resultsdir, system_name=system_name, dataset_file_name = dataset,
              cutoff=cutoff_value, polynomial_cutoff_p=polynomial_cutoff_p_value, default_dtype = default_dtype_value,
              num_layers = num_layers_value, n_train = n_train_value, n_val = n_val_value, max_epochs = max_epochs_value,
              chemical_symbols=symbols_list)
      with open(f"{system_name}.yaml", "w") as f:
         f.write(allegro_input)
   elif args.train and method == "nequip":
      dataset = data_dir+'/'+dataset
      conf = sort(read(dataset))
      symbols_list = re.findall(r'[a-zA-Z]', str(conf.symbols))
      nequip_input = generate_nequip_input(resultsdir=resultsdir, system_name=system_name, dataset_file_name = dataset,
              cutoff=cutoff_value, polynomial_cutoff_p=polynomial_cutoff_p_value, default_dtype = default_dtype_value,
              num_layers = num_layers_value, num_features = num_features_value, n_train = n_train_value, n_val = n_val_value,
              max_epochs = max_epochs_value, chemical_symbols=symbols_list)
      with open(f"{system_name}.yaml", "w") as f:
         f.write(nequip_input)
   elif args.train:
       raise ValueError("Error: Training is only supported for method 'nequip' or 'allegro.")
   
   config = Config.from_file(f'{system_name}.yaml')
   
   if args.train:
       print("##################")
       print("Train model")
       subprocess.call("rm -rf results", shell=True)
       subprocess.call(f"nequip-train {system_name}.yaml", shell=True)
       print("##################")
       print("Training complete")
       

if __name__ == "__main__":
   
   main()

####"""### Train with Allegro or NequIP"""
####
####!rm -rf ./results
####
####Model='Allegro'
####
####if Model=='Allegro':
####  !nequip-train allegro/configs/allegro-water-gra.yaml  --equivariance-test
####elif Model=='NequIP':
####  !nequip-train nequip_train/water-gra.yaml --equivariance-test
####else:
####  print("Model has to be either Allegro or NequIP")
####
####!nequip-train --help
####
####"""## Evaluate the test error
######### We get rather small errors in the forces of ~50 meV/A
####"""
####
####! nequip-evaluate --train-dir results/water-gra-film/water-gra-film --batch-size 10
####
####"""### Deploy the model
####
####We now convert the model to a potential file. This makes it independent of NequIP and we can load it in CP2K to run MD.
####"""
####
####from datetime import datetime
####import subprocess
####
##### datetime object containing current date and time
####depl_time = datetime.now().strftime("%d%m%Y-%H%M")
####
####if Model == "Allegro":
####  !nequip-deploy build --train-dir results/water-gra-film/water-gra-film water-gra-film-deploy-alle.pth
####  cmd = ["cp", "water-gra-film-deploy-alle.pth", "water-gra-film-deploy-alle-"+depl_time+".pth"] 
####elif Model == "NequIP":
####   !nequip-deploy build --train-dir results/water-gra-film/water-gra-film water-gra-film-deploy-neq.pth 
####   cmd = ["cp", "water-gra-film-deploy-neq.pth", "water-gra-film-deploy-neq-"+depl_time+".pth"] 
####
####subprocess.Popen(cmd)
####
####model_is_pretrained = False
####
####if model_is_pretrained == False:
####  ! cp *2023*.pth /content/drive/MyDrive/models_and_datasets/models/.
####else:
####  ! cp /content/drive/MyDrive/models_and_datasets/models/*.pth .
####
