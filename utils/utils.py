import os
import re
import subprocess
from ase.build import sort
import numpy as np
import site
import textwrap
from ase.io import read, write
import ase.data
import MDAnalysis as mda
import itertools


def file_exists(file_path):
    if os.path.exists(file_path):
        return True
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def MD_reader_xyz(f, data_dir, no_skip=0):
    filename = os.path.join(data_dir, f)
    fo = open(filename, "r")
    natoms_str = fo.read().rsplit(" i = ")[0]
    natoms = int(natoms_str.split("\n")[0])
    fo.close()
    fo = open(filename, "r")
    samples = fo.read().split(natoms_str)[1:]
    steps = []
    xyz = []
    temperatures = []
    energies = []
    for sample in samples[::no_skip]:
        entries = sample.split("\n")[:-1]
        energies.append(float(entries[0].split("=")[-1]))
        temp = np.array([list(map(float, lv.split()[1:])) for lv in entries[1:]])
        xyz.append(temp[:, :])
    return natoms_str, np.array(xyz), np.array(energies)

def get_pairs(lst):
    pairs = []
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            pairs.append((lst[i], lst[j]))
    return pairs

def read_cell(f, data_dir):
    filename = os.path.join(data_dir, f)
    fo = open(filename, "r")
    cell_list_abc = fo.read().split("\n")[:-1]
    cell_vec_abc = np.array(
        [list(map(float, lv.split())) for lv in cell_list_abc]
    ).squeeze()
    return cell_vec_abc

def read_n_atoms_xyz(positions_filename):
    with open(positions_filename, "r") as f:
       first_line = f.readline().strip()
       match = re.match(r"(\d+)", first_line)
       if match:
           num_atoms = int(match.group(1))
       else:
           num_atoms = 0
    return num_atoms

def extract_first_frame_xyz(positions_filename, natoms, output_filename):

    command = f"head -n $(expr {natoms} + 2) {positions_filename} > {output_filename}"
    subprocess.run(command, shell=True)

def Energy_reader_cp2k_xyz(f, data_dir, no_skip=0):
    filename = os.path.join(data_dir, f)
    grep_command = f"grep -Po '(?<=E = ).*' {filename}"
    output = subprocess.check_output(grep_command, shell=True, text=True)
    matches = output.splitlines()
    if no_skip == 0:
        ener = [float(en) for en in matches]
    else:
        ener = [float(en) for en in matches[::no_skip]]
    return ener


def combine_trajectory(coordinates_file, forces_file, output_file, cell, interval = 1, mask_labels = False, dim = 0):

    print("enter combine_trajectory")
    coordinates = mda.coordinates.XYZ.XYZReader(coordinates_file)
    topology_coordinates = mda.topology.XYZParser.XYZParser(coordinates_file)
    forces = mda.coordinates.XYZ.XYZReader(forces_file)
    topology_forces = mda.topology.XYZParser.XYZParser(forces_file)    

    cell_box = mda.lib.mdamath.triclinic_box(cell[0,:], cell[1,:], cell[2, :])

    print("after mda topology and coordinates")

    coordinates_universe = mda.Universe(coordinates_file, topology_format = "XYZ", dt = 1.0)
    forces_universe = mda.Universe(forces_file, topology_format = "XYZ", dt = 1.0)

    print("after mda universe")

    coordinates_universe.dimensions = cell_box
    forces_universe.dimensions = cell_box
    natoms = read_n_atoms_xyz(coordinates_file)
    extract_first_frame_xyz(coordinates_file, natoms, "temp_pos.xyz")

    coordinates_ase = read("temp_pos.xyz", format='xyz', index=0, parallel = False)  # Read first frame
    forces_ase = read("temp_pos.xyz", format='xyz', index=0, parallel = False)  # Read first frame
    os.remove("temp_pos.xyz")

    print("after ase reading first frame")

    energies = Energy_reader_cp2k_xyz(coordinates_file, "./", no_skip = interval)

    print("Entering coordinates and forces loop")

    if os.path.isfile(output_file):
        os.remove(output_file)

    for i, (coords, force) in enumerate(itertools.zip_longest(coordinates_universe.trajectory[::interval], forces_universe.trajectory[::interval])):
        print(f"processing frame {i}")
        coordinates_ase.info['energy'] = energies[i]
        coordinates_ase.set_cell(cell)
        coordinates_ase.set_pbc([True, True, True])
        coordinates_ase.set_array('forces', force.positions)  # Add forces to the copied atoms
        coordinates_ase.set_positions(coords.positions)
        if mask_labels:
           coordinates_ase.set_tags(force.positions[:,dim] != 0.0)       
        coordinates_ase_sort = sort(coordinates_ase)
        write(output_file, coordinates_ase_sort, format='extxyz', append = True)

def MD_writer_xyz(
    positions, forces, cell_vec_abc, energies, data_dir, f, conv_frc=1.0, conv_ener=1.0):

    filename = os.path.join(data_dir, f)
    fo = open(filename, "w")

    for it, frame in enumerate(positions):
        natoms = len(frame)
        fo.write("{:5d}\n".format(natoms))
        fo.write(
            'Lattice="{:.5f} 0.0 0.0 0.0 {:.5f} 0.0 0.0 0.0 {:.5f}" \
    Properties="species:S:1:pos:R:3:forces:R:3" \
    energy={:.10f} pbc="T T T"\n'.format(
                cell_vec_abc[0],
                cell_vec_abc[1],
                cell_vec_abc[2],
                energies[it] * conv_ener,
            )
        )
        if it % 1000 == 0.0:
            print(it)

        sorted_frame = sort(frame)
        sorted_forces = sort(forces[it])

        fo.write(
            "".join(
                "{:8s} {:.8f} {:16.8f} {:16.8f}\
     {:16.8f} {:16.8f} {:16.8f}\n".format(
                    sorted_frame[iat].symbol,
                    sorted_frame[iat].position[0],
                    sorted_frame[iat].position[1],
                    sorted_frame[iat].position[2],
                    sorted_forces[iat].position[0] * conv_frc,
                    sorted_forces[iat].position[1] * conv_frc,
                    sorted_forces[iat].position[2] * conv_frc,
                )
                for iat in range(len(frame))
            )
        )

def set_tags_frc_constr(ase_traj, dim = 0):
    for i in range(len(ase_traj)):
        ase_traj[i].set_tags(ase_traj[i].get_forces()[:,dim] != 0.0)
    return ase_traj

def sort_xyz_file(input_file, output_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Extract the number of atoms and comment line
    num_atoms = int(lines[0].strip())
    comment_line = lines[1]
    
    # Extract lattice and properties from the comment line
    lattice_start_index = comment_line.find("Lattice=")
    lattice_end_index = comment_line.find(" Properties=")
    lattice_line = comment_line[lattice_start_index:lattice_end_index]
    properties_line = comment_line[lattice_end_index:]
    
    # Extract atomic symbols and positions
    symbols = []
    positions = []
    for line in lines[2:]:
        parts = line.split()
        symbols.append(parts[0])
        positions.append([float(p) for p in parts[1:4]])
    
    # Get atomic numbers for each symbol
    atomic_nums = [ase.data.atomic_numbers[sym] for sym in symbols]
    
    # Sort the symbols and positions by atomic number
    symbols_sorted = [sym for _, sym in sorted(zip(atomic_nums, symbols))]
    positions_sorted = [pos for _, pos in sorted(zip(atomic_nums, positions))]
    
    # Update the symbols and positions in the XYZ file lines
    lines[2:] = [f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}\n" for sym, pos in zip(symbols_sorted, positions_sorted)]
    
    # Write the sorted structure back to an XYZ file
    with open(output_file, 'w') as f:
        # Write the number of atoms
        f.write(f"{num_atoms}\n")
        
        # Write the comment line
        f.write(comment_line)
        
        # Write the sorted symbols and positions
        f.writelines(lines[2:])

def generate_allegro_input(*args, **kwargs):

    default_resultsdir = "resultsdir"
    default_system_name = "system"    
    default_cutoff = 5.0
    default_polynomial_cutoff_p = 48
    default_default_dtype = "float64"
    default_l_max = 2
    default_num_layers = 2
    default_num_features = 64    
    default_dataset_file_name = "dataset.extxyz"
    default_n_train = 1000
    default_n_val = 100
    default_max_epochs = 100
    default_mask_labels = False
    default_forces_loss = "MSELoss"
    default_batch_size = 1
    default_validation_loss_delta = 0.002
    default_hidden_layers_dim = [128, 256, 512, 1024]
    default_parity = "o3_full"

    cutoff = kwargs.get('cutoff', default_cutoff)
    polynomial_cutoff_p = kwargs.get('polynomial_cutoff_p', default_polynomial_cutoff_p)
    resultsdir = kwargs.get('resultsdir', default_resultsdir)
    system_name = kwargs.get('system_name', default_system_name)
    default_dtype = kwargs.get('default_dtype', default_default_dtype)    
    forces_loss = kwargs.get('forces_loss', default_forces_loss)
    
    l_max = kwargs.get('l_max', default_l_max)
    batch_size = kwargs.get('batch_size', default_batch_size)    
    num_layers = kwargs.get('num_layers', default_num_layers)  
    num_features = kwargs.get('num_features', default_num_features)
    dataset_file_name = kwargs.get('dataset_file_name', default_dataset_file_name)
    n_train = kwargs.get('n_train', default_n_train)
    n_val = kwargs.get('n_val', default_n_val)
    max_epochs = kwargs.get('max_epochs', default_max_epochs)
    parity = kwargs.get('parity', default_parity)
    
    validation_loss_delta = kwargs.get('validation_loss_delta', default_validation_loss_delta)

    chemical_symbols = kwargs.get('chemical_symbols', [])
    symbols = textwrap.indent('\n'.join(f"- {symbol}" for symbol in chemical_symbols), '  ')
    hidden_layers_dim = kwargs.get('hidden_layers_dim', default_hidden_layers_dim)
    hidden_layers_dim_str = '['+', '.join(f"{layer_size}" for layer_size in hidden_layers_dim)+']'

    mask_labels = kwargs.get('mask_labels', default_mask_labels)
    if mask_labels:
        mask_hack = ""
        ignore_nan_value = "True"
    else:
        mask_hack = "#"
        ignore_nan_value = "False"
    

    allegro_input = f"""
# general
root: {resultsdir}
run_name: {system_name}
seed: 42
dataset_seed: 42
append: true
# To use float64 with cp2k, need to implement it, for now use float32
default_dtype: {default_dtype}
model_dtype: {default_dtype}

# -- network --
model_builders:
 - allegro.model.Allegro
 # the typical model builders from `nequip` can still be used:
 - PerSpeciesRescale
 - ForceOutput
 - RescaleEnergyEtc

# cutoffs
r_max: {cutoff}
avg_num_neighbors: auto

# radial basis
# Try to use a small cutoff and a large polynomial cutoff p
# use p=48 in the Li3PO4 case in the paper since the cutoff is quite small
# see https://github.com/mir-group/allegro/discussions/20
BesselBasis_trainable: true
### Use normalize_basis: true, see https://github.com/mir-group/allegro/discussions/20
normalize_basis: true
PolynomialCutoff_p: {polynomial_cutoff_p}

# symmetry
l_max: {l_max}
# whether to include E(3)-symmetry / parity
# allowed: o3_full, o3_restricted, so3
parity: {parity}

# Allegro layers:
# number of tensor product layers, 1-3 usually best, more is more accurate but slower    
num_layers: {num_layers}
# number of features, more is more accurate but slower, 1, 4, 8, 16, 64, 128 are good options to try depending on data set
env_embed_multiplicity: {num_features}
embed_initial_edge: true

# hidden layer dimensions of the 2-body embedding MLP
two_body_latent_mlp_latent_dimensions: {hidden_layers_dim_str}
two_body_latent_mlp_nonlinearity: silu
two_body_latent_mlp_initialization: uniform

# hidden layer dimensions of the latent MLP
# these MLPs are cheap if you have have large l/env_embed_multiplicity, so a good place to put model capacity if you can afford it
# only if you are in the ultra-fast/scalable regime, make these smaller
latent_mlp_latent_dimensions: [{hidden_layers_dim[-1]}, {hidden_layers_dim[-1]}, {hidden_layers_dim[-1]}]
latent_mlp_nonlinearity: silu
latent_mlp_initialization: uniform
latent_resnet: true

env_embed_mlp_latent_dimensions: []
env_embed_mlp_nonlinearity: null
env_embed_mlp_initialization: uniform

# - end allegro layers -

# Final MLP to go from Allegro latent space to edge energies:
# hidden layer dimensions of the per-edge energy final MLP
edge_eng_mlp_latent_dimensions: [{hidden_layers_dim[0]}]
edge_eng_mlp_nonlinearity: null
edge_eng_mlp_initialization: uniform

#include_keys:
#  - user_label
key_mapping:
  user_label: label0

# -- data --
dataset: ase
dataset_file_name: {dataset_file_name}

ase_args:
  format: extxyz

# A mapping of chemical species to type indexes is necessary if the dataset is provided with atomic numbers instead of type indexes.
chemical_symbols:
{symbols}

# ! This important line adds a "pre-transform" to the dataset that processes the `AtomicData` *after* it is loaded and processed
#   from the original data file, but before it is cached to disk for later training use.  This function can be anything, but here
#   we use `nequip.data.transforms.MaskByAtomTag` to make ground truth force labels on atoms with a certain tag NaN (i.e. masked)
#   We can mask multiple tags if we like, or mask other per-atom fields.  In this case, we mask out tag 0, which is subsurface
#   atoms in the toy data from `generate_slab.py`.
{mask_hack}dataset_include_keys: [\"tags\"]
{mask_hack}dataset_pre_transform: !!python/object:nequip.data.transforms.MaskByAtomTag {{'tag_values_to_mask': [0], 'fields_to_mask': ['forces']}}

# logging
wandb: false
#wandb_project: allegro-water-tutorial
verbose: info
log_batch_freq: 10

# training
n_train: {n_train}
n_val: {n_val}
batch_size: {batch_size}
validation_batch_size: {batch_size}

max_epochs: {max_epochs}
learning_rate: 0.005
train_val_split: random
shuffle: true
metrics_key: validation_loss

# use an exponential moving average of the weights
use_ema: true
ema_decay: 0.99
ema_use_num_updates: true

# loss function
loss_coeffs:
  forces:
    - 1
    - {forces_loss}
{mask_hack}    - {{\"ignore_nan\": {ignore_nan_value}}}
  total_energy:
    - 1.
    - PerAtomMSELoss

# optimizer
optimizer_name: Adam # default optimizer is Adam
optimizer_amsgrad: false
optimizer_betas: !!python/tuple
  - 0.9
  - 0.999
optimizer_eps: 1.0e-08
optimizer_weight_decay: 0

metrics_components:
  - - forces                               # key
    - mae                                  # "rmse" or "mae"
{mask_hack}    - ignore_nan: {ignore_nan_value}
  - - forces
    - rmse
{mask_hack}    - ignore_nan: {ignore_nan_value}    
  - - total_energy
    - mae
  - - total_energy
    - mae
    - PerAtom: True                        # if true, energy is normalized by the number of atoms

# lr scheduler, drop lr if no improvement for 50 epochs
lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 25
lr_scheduler_factor: 0.5

early_stopping_lower_bounds:
  LR: 1.0e-5

early_stopping_patiences:
  validation_loss: 50

early_stopping_delta: # If delta is defined, a decrease smaller than delta will not be considered as a decrease
  validation_loss: {validation_loss_delta}

early_stopping_cumulative_delta: false # If True, the minimum value recorded will not be updated when the decrease is smaller than delta

early_stopping_lower_bounds: # stop early if a metric value is lower than the bound
  LR: 1.0e-5
"""
    return allegro_input


def generate_nequip_input(*args, **kwargs):

    default_resultsdir = "results"
    default_system_name = "system"
    default_cutoff = 5.0
    default_default_dtype = "float64"
    default_l_max = 2
    default_polynomial_cutoff_p = 48    
    default_num_layers = 4
    default_num_features = 32
    default_dataset_file_name = "dataset.extxyz"
    default_n_train = 1000
    default_n_val = 100
    default_max_epochs = 100
    default_mask_labels = False
    default_forces_loss = "MSELoss"    
    default_batch_size = 1
    default_validation_loss_delta = 0.002
    default_parity = "o3_full"

    cutoff = kwargs.get('cutoff', default_cutoff)
    polynomial_cutoff_p = kwargs.get('polynomial_cutoff_p', default_polynomial_cutoff_p)
    resultsdir = kwargs.get('traindir', default_resultsdir)
    system_name = kwargs.get('system_name', default_system_name)
    forces_loss = kwargs.get('forces_loss', default_forces_loss)
    batch_size = kwargs.get('batch_size', default_batch_size)

    parity = kwargs.get('parity', default_parity)
    if parity == "o3_full" or "o3_restricted":
        parity = "true"
    elif parity == "so3":
       parity = "false"
    
    default_dtype = kwargs.get('default_dtype', default_default_dtype)
    l_max = kwargs.get('l_max', default_l_max)
    num_layers = kwargs.get('num_layers', default_num_layers)
    num_features = kwargs.get('num_features', default_num_features)
    dataset_file_name = kwargs.get('dataset_file_name', default_dataset_file_name)
    n_train = kwargs.get('n_train', default_n_train)
    n_val = kwargs.get('n_val', default_n_val)
    max_epochs = kwargs.get('max_epochs', default_max_epochs)
    validation_loss_delta = kwargs.get('validation_loss_delta', default_validation_loss_delta)

    chemical_symbols = kwargs.get('chemical_symbols', [])
    symbols = textwrap.indent('\n'.join(f"- {symbol}" for symbol in chemical_symbols), '  ')

    mask_labels = kwargs.get('mask_labels', default_mask_labels)
    if mask_labels:
        mask_hack = ""
        ignore_nan_value = "True"
    else:
        mask_hack = "#"
        ignore_nan_value = "False"

    nequip_input = f"""
# IMPORTANT: READ THIS

# This is a full yaml file with all nequip options.
# It is primarily intented to serve as documentation/reference for all options
# For a simpler yaml file containing all necessary features to get you started, we strongly recommend to start with configs/example.yaml

# Two folders will be used during the training: 'root'/process and 'root'/'run_name'
# run_name contains logfiles and saved models
# process contains processed data sets
# if 'root'/'run_name' exists, 'root'/'run_name'_'year'-'month'-'day'-'hour'-'min'-'s' will be used instead.
root: {resultsdir}
run_name: {system_name}
seed: 42 # model seed
dataset_seed: 42 # data set seed
append: true # set true if a restarted run should append to the previous log file
default_dtype: {default_dtype} # type of float to use, e.g. float32 and float64
model_dtype: {default_dtype}
allow_tf32: false # whether to use TensorFloat32 if it is available
# device:  cuda                                                                   # which device to use. Default: automatically detected cuda or "cpu"

# network
r_max: {cutoff} # cutoff radius in length units, here Angstrom, this is an important hyperparamter to scan
num_layers: {num_layers} # number of interaction blocks, we find 3-5 to work best

l_max: {l_max} # the maximum irrep order (rotation order) for the network's features, l=1 is a good default, l=2 is more accurate but slower
parity: true # whether to include features with odd mirror parityy; often turning parity off gives equally good results but faster networks, so do consider this
num_features: {num_features} # the multiplicity of the features, 32 is a good default for accurate network, if you want to be more accurate, go larger, if you want to be faster, go lower

# alternatively, the irreps of the features in various parts of the network can be specified directly:
# the following options use e3nn irreps notation
# either these four options, or the above three options, should be provided--- they cannot be mixed.
# chemical_embedding_irreps_out: 32x0e                                              # irreps for the chemical embedding of species
# feature_irreps_hidden: 32x0o + 32x0e + 32x1o + 32x1e                              # irreps used for hidden features, here we go up to lmax=1, with even and odd parities; for more accurate but slower networks, use l=2 or higher, smaller number of features is faster
# irreps_edge_sh: 0e + 1o                                                           # irreps of the spherical harmonics used for edges. If a single integer, indicates the full SH up to L_max=that_integer
# conv_to_output_hidden_irreps_out: 16x0e                                           # irreps used in hidden layer of output block

nonlinearity_type: gate # may be 'gate' or 'norm', 'gate' is recommended
resnet:
  false # set true to make interaction block a resnet-style update
  # the resnet update will only be applied when the input and output irreps of the layer are the same

# scalar nonlinearities to use — available options are silu, ssp (shifted softplus), tanh, and abs.
# Different nonlinearities are specified for e (even) and o (odd) parity;
# note that only tanh and abs are correct for o (odd parity).
# silu typically works best for even
nonlinearity_scalars:
  e: silu
  o: tanh

nonlinearity_gates:
  e: silu
  o: tanh

# radial network basis
num_basis: 8 # number of basis functions used in the radial basis, 8 usually works best
BesselBasis_trainable: true # set true to train the bessel weights
PolynomialCutoff_p: {polynomial_cutoff_p} # p-exponent used in polynomial cutoff function, smaller p corresponds to stronger decay with distance

# radial network
invariant_layers: 2 # number of radial layers, usually 1-3 works best, smaller is faster
invariant_neurons: 64 # number of hidden neurons in radial function, smaller is faster
avg_num_neighbors: auto # number of neighbors to divide by, null => no normalization, auto computes it based on dataset
use_sc: true # use self-connection or not, usually gives big improvement

# to specify different parameters for each convolutional layer, try examples below
# layer1_use_sc: true                                                       
# priority for different definitions:
#   invariant_neurons < InteractionBlock_invariant_neurons < layer_i_invariant_neurons

# data set
# there are two options to specify a dataset, npz or ase
# npz works with npz files, ase can ready any format that ase.io.read can read
# in most cases working with the ase option and an extxyz file is by far the simplest way to do it and we strongly recommend using this
# simply provide a single extxyz file that contains the structures together with energies and forces (generated with ase.io.write(atoms, format='extxyz', append=True))

#include_keys:
#  - user_label
key_mapping:
  user_label: label0

# A list of chemical species found in the data. The NequIP atom types will be named after the chemical symbols and ordered by atomic number in ascending order.
# (In this case, NequIP's internal atom type 0 will be named H and type 1 will be named C.)
# Atoms in the input will be assigned NequIP atom types according to their atomic numbers.
chemical_symbols:
{symbols}

# As an alternative option to npz, you can also pass data ase ASE Atoms-objects
# This can often be easier to work with, simply make sure the ASE Atoms object
# has a calculator for which atoms.get_potential_energy() and atoms.get_forces() are defined
dataset: ase
dataset_file_name: {dataset_file_name}

ase_args: # any arguments needed by ase.io.read
  format: extxyz

# ! This important line adds a "pre-transform" to the dataset that processes the `AtomicData` *after* it is loaded and processed
#   from the original data file, but before it is cached to disk for later training use.  This function can be anything, but here
#   we use `nequip.data.transforms.MaskByAtomTag` to make ground truth force labels on atoms with a certain tag NaN (i.e. masked)
#   We can mask multiple tags if we like, or mask other per-atom fields.  In this case, we mask out tag 0, which is subsurface
#   atoms in the toy data from `generate_slab.py`.
{mask_hack}dataset_include_keys: [\"tags\"]
{mask_hack}dataset_pre_transform: !!python/object:nequip.data.transforms.MaskByAtomTag {{'tag_values_to_mask': [0], 'fields_to_mask': ['forces']}}

# If you want to use a different dataset for validation, you can specify
# the same types of options using a `validation_` prefix:
# validation_dataset: ase
# validation_dataset_file_name: xxx.xyz                                            # need to be a format accepted by ase.io.read

# logging
#wandb: true # we recommend using wandb for logging
#wandb_project: water-example # project name used in wandb
#wandb_watch: false

# see https://docs.wandb.ai/ref/python/watch
# wandb_watch_kwargs:
#   log: all
#   log_freq: 1
#   log_graph: true

verbose: info # the same as python logging, e.g. warning, info, debug, error. case insensitive
log_batch_freq: 10 # batch frequency, how often to print training errors withinin the same epoch
log_epoch_freq: 1 # epoch frequency, how often to print
save_checkpoint_freq: -1 # frequency to save the intermediate checkpoint. no saving of intermediate checkpoints when the value is not positive.
save_ema_checkpoint_freq: -1 # frequency to save the intermediate ema checkpoint. no saving of intermediate checkpoints when the value is not positive.

# training
n_train: {n_train} # number of training data
n_val: {n_val} # number of validation data
learning_rate: 0.005 # learning rate, we found values between 0.01 and 0.005 to work best - this is often one of the most important hyperparameters to tune
batch_size: {batch_size} # batch size, we found it important to keep this small for most applications including forces (1-5); for energy-only training, higher batch sizes work better
validation_batch_size: {batch_size} # batch size for evaluating the model during validation. This does not affect the training results, but using the highest value possible (<=n_val) without running out of memory will speed up your training.
max_epochs: {max_epochs} # stop training after _ number of epochs, we set a very large number here, it won't take this long in practice and we will use early stopping instead
train_val_split: random # can be random or sequential. if sequential, first n_train elements are training, next n_val are val, else random, usually random is the right choice
shuffle: true # If true, the data loader will shuffle the data, usually a good idea
metrics_key: validation_loss # metrics used for scheduling and saving best model. Options: `set`_`quantity`, set can be either "train" or "validation, "quantity" can be loss or anything that appears in the validation batch step header, such as f_mae, f_rmse, e_mae, e_rmse
use_ema: true # if true, use exponential moving average on weights for val/test, usually helps a lot with training, in particular for energy errors
ema_decay: 0.99 # ema weight, typically set to 0.99 or 0.999
ema_use_num_updates: true # whether to use number of updates when computing averages
report_init_validation: true # if True, report the validation error for just initialized model

# early stopping based on metrics values.
# LR, wall and any keys printed in the log file can be used.
# The key can start with Training or validation. If not defined, the validation value will be used.
early_stopping_patiences: # stop early if a metric value stopped decreasing for n epochs
  validation_loss: 50

early_stopping_delta: # If delta is defined, a decrease smaller than delta will not be considered as a decrease
  validation_loss: {validation_loss_delta}

early_stopping_cumulative_delta: false # If True, the minimum value recorded will not be updated when the decrease is smaller than delta

early_stopping_lower_bounds: # stop early if a metric value is lower than the bound
  LR: 1.0e-5

early_stopping_upper_bounds: # stop early if a metric value is higher than the bound
  cumulative_wall: 1.0e+100

# loss function
loss_coeffs: # different weights to use in a weighted loss functions
  forces:
    - 1 # if using PerAtomMSELoss, a default weight of 1:1 on each should work well
    - {forces_loss}
{mask_hack}    - {{\"ignore_nan\": {ignore_nan_value}}}
  total_energy:
    - 1
    - PerAtomMSELoss

# # default loss function is MSELoss, the name has to be exactly the same as those in torch.nn.
# the only supprted targets are forces and total_energy

# here are some example of more ways to declare different types of loss functions, depending on your application:
# loss_coeffs:
#   total_energy: MSELoss
#
# loss_coeffs:
#   total_energy:
#   - 3.0
#   - MSELoss
#
# loss_coeffs:
#   total_energy:
#   - 1.0
#   - PerAtomMSELoss
#
# loss_coeffs:
#   forces:
#   - 1.0
#   - PerSpeciesL1Loss
#
# loss_coeffs: total_energy
#
# loss_coeffs:
#   total_energy:
#   - 3.0
#   - L1Loss
#   forces: 1.0

# output metrics
metrics_components:
  - - forces # key
    - mae # "rmse" or "mae"
{mask_hack}    - ignore_nan: {ignore_nan_value}
  - - forces
    - rmse
{mask_hack}    - ignore_nan: {ignore_nan_value}
##  - - forces
##    - mae
##{mask_hack}    - ignore_nan: {ignore_nan_value}
##    - PerSpecies: True # if true, per species contribution is counted separately
##      report_per_component: False # if true, statistics on each component (i.e. fx, fy, fz) will be counted separately
##  - - forces
##    - rmse
##{mask_hack}    - ignore_nan: {ignore_nan_value}
##    - PerSpecies: True
##      report_per_component: False
  - - total_energy
    - mae
  - - total_energy
    - mae
    - PerAtom: True # if true, energy is normalized by the number of atoms

# optimizer, may be any optimizer defined in torch.optim
# the name `optimizer_name`is case sensitive
# IMPORTANT: for NequIP (not for Allegro), we find that in most cases AMSGrad strongly improves
# out-of-distribution generalization over Adam. We highly recommed trying both AMSGrad (by setting
# optimizer_amsgrad: true) and Adam (by setting optimizer_amsgrad: false)
optimizer_name: Adam # default optimizer is Adam
optimizer_amsgrad: true
optimizer_betas: !!python/tuple
  - 0.9
  - 0.999
optimizer_eps: 1.0e-08
optimizer_weight_decay: 0

# gradient clipping using torch.nn.utils.clip_grad_norm_
# see https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html#torch.nn.utils.clip_grad_norm_
# setting to inf or null disables it
max_gradient_norm: null

# lr scheduler, currently only supports the two options listed below, if you need more please file an issue
# first: on-plateau, reduce lr by factory of lr_scheduler_factor if metrics_key hasn't improved for lr_scheduler_patience epoch
lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 100
lr_scheduler_factor: 0.5

# second, cosine annealing with warm restart
# lr_scheduler_name: CosineAnnealingWarmRestarts
# lr_scheduler_T_0: 10000
# lr_scheduler_T_mult: 2
# lr_scheduler_eta_min: 0
# lr_scheduler_last_epoch: -1

# we provide a series of options to shift and scale the data
# these are for advanced use and usually the defaults work very well
# the default is to scale the energies and forces by scaling them by the force standard deviation and to shift the energy by its mean
# in certain cases, it can be useful to have a trainable shift/scale and to also have species-dependent shifts/scales for each atom

per_species_rescale_scales_trainable: false
# whether the scales are trainable. Defaults to False. Optional
per_species_rescale_shifts_trainable: false
# whether the shifts are trainable. Defaults to False. Optional
per_species_rescale_shifts: dataset_per_atom_total_energy_mean
# initial atomic energy shift for each species. default to the mean of per atom energy. Optional
# the value can be a constant float value, an array for each species, or a string
# string option include:
# *  "dataset_per_atom_total_energy_mean", which computes the per atom average
# *  "dataset_per_species_total_energy_mean", which automatically compute the per atom energy mean using a GP model
per_species_rescale_scales: dataset_forces_rms
# initial atomic energy scale for each species. Optional.
# the value can be a constant float value, an array for each species, or a string
# string option include:
# *  "dataset_per_atom_total_energy_std", which computes the per atom energy std
# *  "dataset_per_species_total_energy_std", which uses the GP model uncertainty
# *  "dataset_per_species_forces_rms", which compute the force rms for each species
# If not provided, defaults to dataset_per_species_force_rms or dataset_per_atom_total_energy_std, depending on whether forces are being trained.
# per_species_rescale_kwargs:
#   total_energy:
#     alpha: 0.1
#     max_iteration: 20
#     stride: 100
# keywords for GP decomposition of per specie energy. Optional. Defaults to 0.1
# per_species_rescale_arguments_in_dataset_units: True
# if explicit numbers are given for the shifts/scales, this parameter must specify whether the given numbers are unitless shifts/scales or are in the units of the dataset. If ``True``, any global rescalings will correctly be applied to the per-species values.

# global energy shift and scale
# When "dataset_total_energy_mean", the mean energy of the dataset. When None, disables the global shift. When a number, used directly.
# Warning: if this value is not None, the model is no longer size extensive
global_rescale_shift: null

# global energy scale. When "dataset_force_rms", the RMS of force components in the dataset. When "dataset_total_energy_std", the stdev of energies in the dataset. When null, disables the global scale. When a number, used directly.
# If not provided, defaults to either dataset_force_rms or dataset_total_energy_std, depending on whether forces are being trained.
global_rescale_scale: dataset_forces_rms

# whether the shift of the final global energy rescaling should be trainable
global_rescale_shift_trainable: false

# whether the scale of the final global energy rescaling should be trainable
global_rescale_scale_trainable: false
# # full block needed for per specie rescale
# global_rescale_shift: null
# global_rescale_shift_trainable: false
# global_rescale_scale: dataset_forces_rms
# global_rescale_scale_trainable: false
# per_species_rescale_trainable: true
# per_species_rescale_shifts: dataset_per_atom_total_energy_mean
# per_species_rescale_scales: dataset_per_atom_total_energy_std

# # full block needed for global rescale
# global_rescale_shift: dataset_total_energy_mean
# global_rescale_shift_trainable: false
# global_rescale_scale: dataset_forces_rms
# global_rescale_scale_trainable: false
# per_species_rescale_trainable: false
# per_species_rescale_shifts: null
# per_species_rescale_scales: null

# Options for e3nn's set_optimization_defaults. A dict:
# e3nn_optimization_defaults:
#   explicit_backward: True
"""
    return nequip_input

def generate_cp2k_input_md(*args, **kwargs):

    default_system_name = "system"
    default_model_name = "model.pth"
    default_method_name = "NEQUIP"
    default_coord_file_name = "coords.xyz"
    default_n_steps = 1000
    default_temperature = 300
    default_unit_coords = "angstrom"
    default_unit_energy = "Hartree"
    default_unit_forces = "Hartree*Bohr^-1"
    default_restart_cp2k = False

    system_name = kwargs.get('system_name', default_system_name)
    model_name = kwargs.get('model_name', default_model_name)
    method_name = kwargs.get('method_name', default_method_name)
    coord_file_name = kwargs.get('coord_file_name', default_coord_file_name)
    n_steps = kwargs.get('n_steps', default_n_steps)
    temperature = kwargs.get('temperature', default_temperature)    

    chemical_symbols = kwargs.get('chemical_symbols', [])
    symbols = ' '.join(f"{symbol}" for symbol in chemical_symbols)

    cell_vals_a = kwargs.get('cell_a', [])
    cell_a = ' '.join(f"{cell_value}" for cell_value in cell_vals_a)    
    cell_vals_b = kwargs.get('cell_b', [])
    cell_b = ' '.join(f"{cell_value}" for cell_value in cell_vals_b)
    cell_vals_c = kwargs.get('cell_c', [])
    cell_c = ' '.join(f"{cell_value}" for cell_value in cell_vals_c)

    restart_cp2k = kwargs.get('restart_cp2k', default_restart_cp2k)
    restart_cp2k = int(restart_cp2k)

    unit_coords = kwargs.get('unit_coords', default_unit_coords)
    unit_forces = kwargs.get('unit_forces', default_unit_forces)
    unit_energy = kwargs.get('unit_energy', default_unit_energy)
    
    cp2k_input_md = f"""
@SET RESTART  {restart_cp2k}   
@SET RESTART {system_name}
&GLOBAL
  PROJECT {system_name}
  RUN_TYPE MD
&END GLOBAL
&FORCE_EVAL
  METHOD FIST
  &MM
    &FORCEFIELD
     &NONBONDED
     &{method_name}
        ATOMS {symbols}
        PARM_FILE_NAME {model_name}
        UNIT_COORDS {unit_coords}
        UNIT_ENERGY {unit_energy}
        UNIT_FORCES {unit_forces}
     &END {method_name}
    &END NONBONDED
    &END FORCEFIELD
    &POISSON
      &EWALD
        EWALD_TYPE none
      &END EWALD
    &END POISSON
  &END MM
  &SUBSYS
    &CELL
       A {cell_a}
       B {cell_b}
       C {cell_c}
#      MULTIPLE_UNIT_CELL 2 2 2
    &END CELL
    &TOPOLOGY
#     MULTIPLE_UNIT_CELL 2 2 2
      COORD_FILE_NAME {coord_file_name}
      COORD_FILE_FORMAT XYZ
    &END TOPOLOGY
  &END SUBSYS
&END FORCE_EVAL
&MOTION
#  &CONSTRAINT
#    &FIXED_ATOMS
#      LIST 1..240
#    &END
#  &END
  &MD
    ENSEMBLE NVT
    STEPS {n_steps}
    TIMESTEP 0.5
    TEMPERATURE {temperature}
    &THERMOSTAT
   &NOSE
      LENGTH 4
      TIMECON 100.
      MTS 2
    &END NOSE
    &END    
  &END MD

    &PRINT
      &FORCES
       UNIT eV*angstrom^-1
        &EACH
          MD 1
        &END
      &END

      &TRAJECTORY
        &EACH
          MD 1
        &END
      &END

      &RESTART
        &EACH
          MD 100
        &END
      &END
      &RESTART_HISTORY
        &EACH
          MD 1000
        &END
      &END
    &END

&END MOTION

@if ${{RESTART}} == 1
&EXT_RESTART
  RESTART_FILE_NAME ${{RESFILE}}-1.restart
   RESTART_DEFAULT T
   RESTART_COUNTERS T
   RESTART_POS T
   RESTART_VEL T
   RESTART_CELL T
   RESTART_THERMOSTAT T
&END EXT_RESTART
@endif

"""
    return cp2k_input_md

