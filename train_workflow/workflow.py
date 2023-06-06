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
from datetime import datetime
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
   arg_parser.add_argument('--deploy',
                           help='deploy model',
                           action="store_true")
   arg_parser.add_argument('--no-deploy', dest='deploy',
                           action="store_false")   
   arg_parser.add_argument('--data_load',
                           help='load data',
                           action="store_true")
   arg_parser.add_argument('--no-data_load',
                           dest='data_load',
                           action="store_false")
   arg_parser.add_argument('--mask_labels',
                           help='mask labels',
                           action="store_true")
   arg_parser.add_argument('--no-mask_labels',
                           dest='mask_labels',
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
   arg_parser.add_argument('--cp2k_coord_file_name',
                           required = False,
                           default = "None",
                           help="provide a CP2K configuration file in xyz or extxyz format")
   arg_parser.add_argument('--cell',
                           required = False,
                           default = [None,None,None],
                           type=float,
                           nargs=3, 
                           help='three floats containing the cell vectors in Angstrom')   
   arg_parser.add_argument('--system_name',
                           required = False,
                           default = "system_name",
                           help="compulsory, use 'system_name' as default")
   arg_parser.add_argument('--model_name',
                           required = False,
                           default = "model.pth",
                           help="Name of the serialized model use 'model.pth' as default")   
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
   arg_parser.add_argument('--n_steps',
                           required = False,
                           default = 1000,
                           help="number of steps to run MD with CP2K")   
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
   arg_parser.add_argument('--unit_coords',
                           required = False,
                           default = "angstrom",
                           help="units of coordinates of the allegro/nequip model for CP2K")
   arg_parser.add_argument('--unit_energy',
                           required = False,
                           default = "Hartree",
                           help="units of energy of the allegro/nequip model for CP2K")
   arg_parser.add_argument('--unit_forces',
                           required = False,
                           default = "Hartree*Bohr^-1",
                           help="units of forces of the allegro/nequip model for CP2K")
   arg_parser.add_argument('--cp2k_exe',
                           required = False,
                           default = "cp2k.ssmp",
                           help="path to CP2K executable")
   
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
   mask_labels = args.mask_labels
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
   #cp2k options
   coord_file_name = args.cp2k_coord_file_name
   cell_value = args.cell
   n_steps_value = args.n_steps
   model_name = args.model_name
   unit_energy = args.unit_energy
   unit_coords = args.unit_coords
   unit_forces =  args.unit_forces
   cp2k_exe = args.cp2k_exe

   print_run_info()
   
   np.random.seed(0)
   if torch.cuda.is_available():
     torch.cuda.manual_seed(0)
   else:
     torch.manual_seed(0)
   
   if args.train and method == "allegro":
      dataset = data_dir+'/'+dataset
      conf = sort(read(dataset), index = "-1")
      symbols_list = re.findall(r'[a-zA-Z]', str(conf.symbols))
      allegro_input = generate_allegro_input(resultsdir=resultsdir, system_name=system_name, dataset_file_name = dataset,
              cutoff=cutoff_value, polynomial_cutoff_p=polynomial_cutoff_p_value, default_dtype = default_dtype_value,
              num_layers = num_layers_value, n_train = n_train_value, n_val = n_val_value, max_epochs = max_epochs_value,
              chemical_symbols=symbols_list, mask_labels = mask_labels)
      with open(f"{system_name}.yaml", "w") as f:
         f.write(allegro_input)
   elif args.train and method == "nequip":
      dataset = data_dir+'/'+dataset
      conf = sort(read(dataset))
      symbols_list = re.findall(r'[a-zA-Z]', str(conf.symbols))
      nequip_input = generate_nequip_input(resultsdir=resultsdir, system_name=system_name, dataset_file_name = dataset,
              cutoff=cutoff_value, polynomial_cutoff_p=polynomial_cutoff_p_value, default_dtype = default_dtype_value,
              num_layers = num_layers_value, num_features = num_features_value, n_train = n_train_value, n_val = n_val_value,
              max_epochs = max_epochs_value, chemical_symbols=symbols_list, mask_labels = mask_labels)
      with open(f"{system_name}.yaml", "w") as f:
         f.write(nequip_input)
   elif args.train:
       raise ValueError("Error: Training is only supported for method 'nequip' or 'allegro.")

   if args.train or args.deploy:

      config = Config.from_file(f'{system_name}.yaml')
   
   if args.train:
       print("##################")
       print("Train model")
       subprocess.call("mv results results_bak", shell=True)
       subprocess.call(f"nequip-train {system_name}.yaml", shell=True)
       print("##################")
       print("Training complete")

   if args.deploy:
       print("##################")
       print("Deploy model:")
       depl_time = datetime.now().strftime("%d%m%Y_%H%M")
       print(f"nequip-deploy build --train-dir {resultsdir}/{system_name} {system_name}_deploy_{depl_time}.pth")
       subprocess.call(f"nequip-deploy build --train-dir {resultsdir}/{system_name} {system_name}_deploy_{depl_time}.pth", shell=True)
       print("Model deployed")
       print("##################")      

   if args.run_md:
       print("##################")
       print("Run MD")
       subprocess.call("mv cp2k_run cp2k_run_bak", shell=True)
       subprocess.call("mkdir cp2k_run", shell=True)
       if args.deploy:
           model_name = f"{system_name}_deploy_{depl_time}.pth"
       if not os.path.exists(model_name):
          raise FileNotFoundError(f"{model_name} is either not found or not in the current running directory.")
       subprocess.call(f"cp {model_name} cp2k_run/.", shell = True)

       if ".extxyz" in coord_file_name:

          conf = sort(read(coord_file_name, index = '-1'))
          write("temp.extxyz", conf)
          symbols_list = re.findall(r'[a-zA-Z]', str(conf.symbols))
          atomic_nums = [ase.data.atomic_numbers[sym] for sym in symbols_list]
          symbols_list = [e[1] for e in sorted(zip(atomic_nums, symbols_list))]

          sort_xyz_file("temp.extxyz", f"cp2k_run/{coord_file_name}")
          subprocess.call("rm temp.extxyz", shell =True)

          if None in cell_value:
             cell_value = [conf.cell[i,i] for i in range(len(conf.cell))]

       if ".xyz" in coord_file_name:

          conf = sort(read(coord_file_name, index = '-1'))

          write("temp.extxyz", conf)
          symbols_list = re.findall(r'[a-zA-Z]', str(conf.symbols))
          atomic_nums = [ase.data.atomic_numbers[sym] for sym in symbols_list]
          symbols_list = [e[1] for e in sorted(zip(atomic_nums, symbols_list))]

          sort_xyz_file("temp.extxyz", f"cp2k_run/{coord_file_name}")
          subprocess.call("rm temp.extxyz", shell =True)

       cp2k_input_md = generate_cp2k_input_md(system_name=system_name, coord_file_name = coord_file_name, method_name = args.method.upper(),
                                             model_name  = model_name, n_steps = n_steps_value, cell = cell_value, 
                                             unit_coords = unit_coords, unit_energy = unit_energy, unit_forces = unit_forces, 
                                             chemical_symbols=symbols_list)
       with open("cp2k_run/neq_alle_md.inp", "w") as f:
          f.write(cp2k_input_md)
       
       subprocess.call(f"cd cp2k_run/ && {cp2k_exe} -i neq_alle_md.inp ",shell=True)
       print("MD completed")
       print("##################")
       
       
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