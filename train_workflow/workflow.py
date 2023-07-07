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
   arg_parser.add_argument('--data_process',
                           help='preprocess data',
                           action="store_true")
   arg_parser.add_argument('--no-data_process',
                           dest='data_prprocess',
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
   arg_parser.add_argument('--restart_cp2k',
                           help="restart md with cp2k",
                           action="store_true")
   arg_parser.add_argument('--no-restart_cp2k',
                           dest='restart_cp2k',
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
   arg_parser.add_argument('--cell_a',
                           required = False,
                           default = [None,None,None],
                           type=float,
                           nargs=3, 
                           help='three floats containing the A cell vector in Angstrom')   
   arg_parser.add_argument('--cell_b',
                           required = False,
                           default = [None,None,None],
                           type=float,
                           nargs=3,
                           help='three floats containing the B cell vector in Angstrom')
   arg_parser.add_argument('--cell_c',
                           required = False,
                           default = [None,None,None],
                           type=float,
                           nargs=3,
                           help='three floats containing the C cell vector in Angstrom')   
   arg_parser.add_argument('--interval',
                           default = 1,
                           type = int, 
                           help="slice of trajectory when processing the positions and forces, use same as ase")
   arg_parser.add_argument('--system_name',
                           required = False,
                           default = "system_name",
                           help="use 'system_name' as default")
   arg_parser.add_argument('--forces_loss',
                           required = False,
                           default = "MSELoss",
                           help="use e.g. MSELoss or PerSpeciesMSELoss")   
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
                           default = 6,
                           help="polynomial cutoff p value")
   arg_parser.add_argument('--l_max',
                           required = False,
                           default = 2,
                           help="l_max")
   arg_parser.add_argument('--num_layers',
                           required = False,
                           default = 2,
                           help="number of layers")
   arg_parser.add_argument('--num_tensor_features',
                           required = False,
                           default = 8,
                           help="number of tensor features")
   arg_parser.add_argument('--parity',
                           required = False,
                           default = "o3_full",
                           choices = ["o3_full", "o3_restricted", "so3"],
                           help="which symmetry to use for parity, choose between o3_full, o3_restricted, so3")
   arg_parser.add_argument('--hyperparams_size',
                           required = False,
                           default = "custom",
                           choices = ["custom" , "small", "medium", "large", "small_tens", "medium_tens"],
                           help="flag that determines the overall size of the model, for allegro only, for now")   
   arg_parser.add_argument('--two_body_mlp',
                           required = False,
                           nargs = "+",
                           type = int, 
                           default = [32, 64, 128],
                           help="hidden layer dimensions of the 2-body embedding MLP for allegro")   
   arg_parser.add_argument('--latent_mlp',
                           required = False,
                           nargs = "+",
                           type = int,
                           default = [128],
                           help="hidden layer dimensions of the latent MLP for allegro, the second mlp in the allegro layer")
   arg_parser.add_argument('--output_mlp',
                           required = False,
                           nargs = "+",
                           type = int,
                           default = [32],
                           help="hidden layer dimensions of the last (output) mlp to compute the edge energy") 
   arg_parser.add_argument('--max_epochs',
                           required = False,
                           default = 10000,
                           help="maximum number of epochs")
   arg_parser.add_argument('--n_steps',
                           required = False,
                           default = 1000,
                           help="number of steps to run MD with CP2K")  
   arg_parser.add_argument('--temperature',
                           required = False,
                           default = 300,
                           help="temperature of MD with CP2K")   
   arg_parser.add_argument('--n_train',
                           required = False,
                           default = 1000,
                           help="size of training set")
   arg_parser.add_argument('--n_val',
                           required = False,
                           default = 100,
                           help="size of validation set")
   arg_parser.add_argument('--batch_size',
                           required = False,
                           default = 1,
                           help="batch size")  
   arg_parser.add_argument('--validation_loss_delta',
                           required = False,
                           default = 0.002,
                           help="threshold on the residual of the validation loss beween epochs for early stopping")   
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
   forces_loss = args.forces_loss
   #options of the model
   cutoff_value = args.cutoff
   polynomial_cutoff_p_value = args.polynomial_cutoff_p
   num_layers_value = args.num_layers
   num_tensor_features_value = args.num_tensor_features
   two_body_mlp_value = args.two_body_mlp
   latent_mlp_value = args.latent_mlp
   output_mlp_value = args.output_mlp  
   hyperparams_size = args.hyperparams_size
   parity_value = args.parity
   l_max_value = args.l_max
   default_dtype_value = args.default_dtype
   #training options
   n_train_value = args.n_train
   batch_size_value = args.batch_size
   validation_loss_delta = args.validation_loss_delta
   n_val_value = args.n_val
   max_epochs_value = args.max_epochs
   #cp2k options
   coord_file_name = args.cp2k_coord_file_name
   cell_value_a = args.cell_a
   cell_value_b = args.cell_b
   cell_value_c = args.cell_c
   interval = args.interval
   n_steps_value = args.n_steps
   temperature_value = args.temperature
   restart_cp2k_value = args.restart_cp2k
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

   if args.data_process:

       print("*****************************")
       print("Processing positions and forces files")     
       positions = data_dir + '/' + data_pos
       forces = data_dir + '/' + data_frc
       file_exists(positions)
       file_exists(forces)

       cell_vec_a = np.array([float(c) for c in cell_value_a ])
       cell_vec_b = np.array([float(c) for c in cell_value_b ])
       cell_vec_c = np.array([float(c) for c in cell_value_c ])
       cell_mat = np.concatenate( (cell_vec_a, cell_vec_b, cell_vec_c), axis = 0).reshape(3,3)

       combine_trajectory( positions, forces, data_dir+'/'+dataset, cell_mat, 
                           interval = interval, mask_labels = mask_labels, dim = 0)
   
   if args.train and method == "allegro":
      dataset = data_dir+'/'+dataset
      conf = sort(read(dataset, index = "-1"))
      symbols_list = list(set(conf.get_chemical_symbols()))

      print(hyperparams_size)

      l_max_value, num_layers_value, num_tensor_features, two_body_mlp_value, latent_mlp_value, output_mlp_value, parity_value = set_hyperparams_size(hyperparams_size,
              l_max_value, num_layers_value, num_tensor_features_value, two_body_mlp_value, latent_mlp_value, output_mlp_value, parity_value)

      allegro_input = generate_allegro_input(resultsdir=resultsdir, system_name=system_name, dataset_file_name = dataset,
              cutoff=cutoff_value, polynomial_cutoff_p=polynomial_cutoff_p_value, default_dtype = default_dtype_value,
              num_layers = num_layers_value, num_tensor_features = num_tensor_features_value, two_body_mlp = two_body_mlp_value,
              latent_mlp = latent_mlp_value, output_mlp = output_mlp_value,parity = parity_value, l_max = l_max_value,
              n_train = n_train_value, n_val = n_val_value, max_epochs = max_epochs_value,  batch_size = batch_size_value, 
              chemical_symbols=symbols_list, mask_labels = mask_labels, forces_loss = forces_loss, validation_loss_delta = validation_loss_delta)
      with open(f"{system_name}.yaml", "w") as f:
         f.write(allegro_input)
      print("*****************************")
      print("Train an allegro model")
   elif args.train and method == "nequip":
      dataset = data_dir+'/'+dataset
      conf = sort(read(dataset))
      symbols_list = list(set(conf.get_chemical_symbols()))
      nequip_input = generate_nequip_input(resultsdir=resultsdir, system_name=system_name, dataset_file_name = dataset,
              cutoff=cutoff_value, polynomial_cutoff_p=polynomial_cutoff_p_value, default_dtype = default_dtype_value,
              num_layers = num_layers_value, num_features = num_features_value, parity = parity_value, l_max = l_max_value, 
              n_train = n_train_value, n_val = n_val_value, max_epochs = max_epochs_value, batch_size = batch_size_value, 
              chemical_symbols=symbols_list, mask_labels = mask_labels, forces_loss = forces_loss, 
              validation_loss_delta = validation_loss_delta)
      with open(f"{system_name}.yaml", "w") as f:
         f.write(nequip_input)
      print("*****************************")
      print("Train a nequip model")
   elif args.train:
       raise ValueError("Error: Training is only supported for method 'nequip' or 'allegro.")

   if args.train or args.deploy:

      config = Config.from_file(f'{system_name}.yaml')
   
   if args.train:
       print("##################")
       print("Train model")
       if os.path.isdir(f"{resultsdir}"):
          if os.path.isdir(f"{resultsdir}_bak"): 
              subprocess.call(f"rm -r {resultsdir}_bak", shell=True)
          subprocess.call(f"mv {resultsdir} {resultsdir}_bak", shell=True)
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
       if os.path.isdir("cp2k_run"):
           if os.path.isdir("cp2k_run_bak"):
               subprocess.call(f"rm -r cp2k_run_bak", shell=True)
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
          symbols_list = list(set(conf.get_chemical_symbols()))
          atomic_nums = [ase.data.atomic_numbers[sym] for sym in symbols_list]
          symbols_list = [e[1] for e in sorted(zip(atomic_nums, symbols_list))]

          sort_xyz_file("temp.extxyz", f"cp2k_run/{coord_file_name}")
          subprocess.call("rm temp.extxyz", shell =True)

          if None in cell_value_a:
             cell_value_a = np.array([conf.cell[0,i] for i in range(len(conf.cell))])
             cell_value_b = np.array([conf.cell[1,i] for i in range(len(conf.cell))])
             cell_value_c = np.array([conf.cell[2,i] for i in range(len(conf.cell))])

       if ".xyz" in coord_file_name:

          conf = sort(read(coord_file_name, index = '-1'))

          write("temp.extxyz", conf)
          symbols_list = list(set(conf.get_chemical_symbols()))
          atomic_nums = [ase.data.atomic_numbers[sym] for sym in symbols_list]
          symbols_list = [e[1] for e in sorted(zip(atomic_nums, symbols_list))]

          sort_xyz_file("temp.extxyz", f"cp2k_run/{coord_file_name}")
          subprocess.call("rm temp.extxyz", shell =True)

       cp2k_input_md = generate_cp2k_input_md(system_name=system_name, coord_file_name = coord_file_name, method_name = args.method.upper(),
                                             model_name  = model_name, n_steps = n_steps_value, cell_a = cell_value_a, cell_b = cell_value_b,
                                             cell_c = cell_value_c, temperature = temperature_value, restart_cp2k = restart_cp2k_value,
                                             unit_coords = unit_coords, unit_energy = unit_energy, unit_forces = unit_forces, 
                                             chemical_symbols=symbols_list)
       with open("cp2k_run/neq_alle_md.inp", "w") as f:
          f.write(cp2k_input_md)
       
       subprocess.call(f"cd cp2k_run/ && {cp2k_exe} -i neq_alle_md.inp > out.out",shell=True)
       print("MD completed")
       print("##################")

   if args.analysis:

       print("##################")
       print("Perform analysis of the rdfs")

       cell_vec_a = np.array([float(c) for c in cell_value_a ])
       cell_vec_b = np.array([float(c) for c in cell_value_b ])
       cell_vec_c = np.array([float(c) for c in cell_value_c ])
       cell_mat = np.concatenate( (cell_vec_a, cell_vec_b, cell_vec_c), axis = 0).reshape(3,3)

       positions = data_dir + '/' + data_pos

       coordinates = mda.coordinates.XYZ.XYZReader(positions)
       topology_coordinates = mda.topology.XYZParser.XYZParser(positions)
       cell_box = mda.lib.mdamath.triclinic_box(cell_vec_a, cell_vec_b, cell_vec_c)
       coordinates_universe = mda.Universe(positions, topology_format = "XYZ", dt = .001)
  
       coordinates_universe.dimensions = cell_box

       conf_ase = sort(read(positions, index = '-1'))
       symbols_list = list(set(conf_ase.get_chemical_symbols()))
       atomic_nums = [ase.data.atomic_numbers[sym] for sym in symbols_list]
       symbols_list = [e[1] for e in sorted(zip(atomic_nums, symbols_list))]

       element_pairs = get_pairs(symbols_list)

       for el_i, el_j in element_pairs:

         select_i = coordinates_universe.select_atoms(f'name {el_i}')
         select_j = coordinates_universe.select_atoms(f'name {el_j}')

         print(f"Compute rdf for pair {el_i} {el_j}")       

         rdf_ij = rdf.InterRDF(select_i, select_j, nbins = int(np.rint( np.min(cell_box[:3]) /2./0.05 )),
                 range=(0.00001, np.min(cell_box[:3]) /2.))

         if (interval == 0 or interval == 1):
            rdf_ij.run()
         else:
            rdf_ij.run(step = interval) 
   
         # Plot RDF
         plt.plot(rdf_ij.bins, rdf_ij.rdf)
         plt.xlabel('Radius (angstrom)')
         plt.ylabel(f'g(r) {el_i}-{el_j}')
     
         # Save RDF data to a file
         data_filename = f'rdf_data_{el_i}_{el_j}.dat'
         np.savetxt(data_filename, np.column_stack((rdf_ij.bins, rdf_ij.rdf)), header='Radius (angstrom)\tg(r)')
     
         # Save plot as PNG file
         plot_filename = f'rdf_plot_{el_i}_{el_j}.png'
         plt.savefig(plot_filename)
         
         # Clear the plot for the next pair
         plt.clf()         
               
if __name__ == "__main__":
   
   main()
