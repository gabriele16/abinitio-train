
import warnings
import os
import subprocess
import matplotlib.pyplot as plt
import argparse
from argparse import RawTextHelpFormatter

import torch

print("torch version ", torch. __version__)

import numpy as np
import site
site.main()
import os
os.environ["WANDB_ANONYMOUS"] = "must"
import numpy as np

from ase import Atoms 
from ase.io import read, write
from ase.io.trajectory import Trajectory
import nequip
print(nequip.__version__)

from nequip.utils import Config
import MDAnalysis as mda
from MDAnalysis.analysis import rdf
import matplotlib.pyplot as plt


description = """pipeline to train NequIP potential and run MD with lammps
"""

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
                        help="run md with lammps",
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
arg_parser.add_argument('--data_dir',
                        default = "./",
                        help="directory where training set is")
arg_parser.add_argument('--forces',
                        required = True,
                        help="compulsory, 'GRAS-frc-1.xyz' as default")
arg_parser.add_argument('--positions',
                        required = True,
                        default = "GRAS-pos-1.xyz",
                        help="compulsory, use 'GRAS-pos-1.xyz' as default")
arg_parser.add_argument('--data_set',
                        required = True,
                        default = "water_graphene_dataset",
                        help="compulsory, use 'water_graphene_dataset' as default")
arg_parser.add_argument('--model_system',
                        required = True,
                        default = "water_graphene",
                        help="compulsory, use 'water_graphene_dataset' as default")


args = arg_parser.parse_args()

#data_dir = '/home/ubuntu/projects/data/aimd_water_graphene/aimd_water_graphene/small_0/'
data_dir = args.data_dir
data_pos = args.positions
data_frc = args.forces
data_set = args.data_set
model_system = args.model_system

np.random.seed(0)
torch.manual_seed(0)

def MD_reader_xyz(f, data_dir, no_skip = 0):
  filename = os.path.join(data_dir, f)
  fo = open(filename, 'r')
  natoms_str = fo.read().rsplit(' i = ')[0]
  natoms = int(natoms_str.split('\n')[0])
  fo.close()  
  fo = open(filename, 'r')
  samples = fo.read().split(natoms_str)[1:]
  steps = []
  xyz = []
  temperatures = []
  energies = []
  for sample in samples[::no_skip]:
     entries = sample.split('\n')[:-1]
     energies.append(float(entries[0].split("=")[-1]))
     temp = np.array([list(map(float, lv.split()[1:])) for lv in entries[1:]])
     xyz.append(temp[:,:])
  return natoms_str, np.array(xyz), np.array(energies)

def MD_writer_xyz(positions,forces,cell_vec_abc,energies,
                  data_dir,f,  conv_frc = 1.0 , conv_ener = 1.0 ):

  filename = os.path.join(data_dir, f)
  fo = open(filename, 'w')

  for it, frame in enumerate(positions):
    natoms = len(frame)
    fo.write("{:5d}\n".format(natoms))
    fo.write('Lattice="{:.5f} 0.0 0.0 0.0 {:.5f} 0.0 0.0 0.0 {:.5f}" \
    Properties="species:S:1:pos:R:3:forces:R:3" \
    energy={:.10f} pbc="T T T"\n'.format(cell_vec_abc[0],cell_vec_abc[1],cell_vec_abc[2],energies[it]*conv_ener)    
    )
    if it%1000 == 0.0:
      print(it)
    
    sorted_frame = frame[frame.numbers.argsort()]
    sorted_forces = forces[it][forces[it].numbers.argsort()]

    fo.write("".join("{:8s} {:.8f} {:16.8f} {:16.8f}\
     {:16.8f} {:16.8f} {:16.8f}\n".format(sorted_frame[iat].symbol,
                                          sorted_frame[iat].position[0],
                                          sorted_frame[iat].position[1],
                                          sorted_frame[iat].position[2],
                                          sorted_forces[iat].position[0]*conv_frc,
                                          sorted_forces[iat].position[1]*conv_frc,
                                          sorted_forces[iat].position[2]*conv_frc)
                                          for iat in range(len(frame))))

def read_cell(f,data_dir):
    filename = os.path.join(data_dir,f)
    fo = open(filename,'r')
    cell_list_abc = fo.read().split('\n')[:-1]
    cell_vec_abc = np.array([list(map(float, lv.split())) for lv in cell_list_abc]).squeeze()
    return(cell_vec_abc)

cell_vec_abc = read_cell('celldata.dat',data_dir )

#if (args.data_load or vars(args)['data_load'] != 'False' or vars(args)['data_load'][0] != 'F' or vars(args)['data_load'] == 'True' or vars(args)['data_load'][0] == 'T'):
if (args.data_load):
    wat_traj = read(data_dir + data_pos, index='::100')
    wat_frc = read(data_dir + data_frc, index='::100')
    natoms, positions, energies = MD_reader_xyz(data_pos, data_dir , no_skip=1)
    MD_writer_xyz(wat_traj, wat_frc, cell_vec_abc, energies, data_dir , f'{data_set}.extxyz',conv_frc = 1.0, conv_ener = 27.211399)

config = Config.from_file(f'{model_system}.yaml')

#if (args.train or vars(args)['train'] != 'False' or vars(args)['train'][0] != 'F' or vars(args)['train'] == 'True' or vars(args)['train'][0] == 'T'):
if args.train:
    print("##################")
    print("Train model")
    subprocess.call("rm -rf results", shell=True)
    subprocess.call(f"nequip-train {model_system}.yaml", shell=True)
    print("##################")
    print("Training complete")

#else:
#    subprocess.call(f"mkdir results", shell=True)
#    subprocess.call(f"mkdir results/{model_system}", shell=True)
#    subprocess.call(f"mkdir results/{model_system}/run-{model_system}", shell=True)
#    subprocess.call(f"cp {model_system}.pth results/{model_system}/run-{model_system}/best_model.pth", shell=True)
#    subprocess.call(f"cp {model_system}.yaml results/{model_system}/run-{model_system}/config.yaml", shell=True)

print("##################")
print("Deploy model:")
print(f"nequip-deploy build --train-dir results/{model_system}/run-{model_system} {model_system}.pth")
subprocess.call(f"nequip-deploy build --train-dir results/{model_system}/run-{model_system} {model_system}.pth", shell=True)
print("Model deployed")
print("##################")

if args.train:
    print("##################")
    print("Evaluate model")
    subprocess.call(f"nequip-evaluate --train-dir results/{model_system}/run-{model_system} --batch-size 50", shell=True)
    print("##################")

lammps_input_md = f"""
units           metal
boundary        p p p
atom_style      atomic
thermo 1
newton off
read_data structure.data
replicate       1 1 1

neighbor        1.0 bin
neigh_modify    every 10 delay 0 check no

pair_style	nequip
#pair_coeff	* * ../{model_system}.pth H C O
pair_coeff     * * ../{model_system}.pth H O
mass            1 1.00794
mass            2 12.011
#mass            3 15.9994

velocity        all create 300.0 23456789
timestep        0.0005
fix             1 all nvt temp 300.0 300.0 $(100.0*dt)

#print log every X steps
thermo          1
thermo_style    custom step pe ke etotal temp press vol cpu 

#print trajectory in xyz every X time units
dump              1 all xyz 1 {model_system}.xyz 
# dump_modify       1 element H O

# dump            2 all custom 1 dump_frc.lamppstrj id type element fx fy fz
#dump_modify     2 element O H
# dump            3 all custom 1 dump.lammpstrj id type element x y z
# dump_modify     3 element O H

run             100
"""

subprocess.call("mkdir lammps_run",shell=True)

with open(f"lammps_run/{model_system}.in", "w") as f:
    f.write(lammps_input_md)

#subprocess.call(f"cp {model_system}.pth lammps_run/.",shell=True)

print(f"{data_dir}{data_set}.extxyz")
data_pos_frc_trj = read(f"{data_dir}{data_set}.extxyz")
write("lammps_run/structure.data", data_pos_frc_trj,format='lammps-data',specorder=["H","O"])

#subprocess.call(f"cat lammps_run/{model_system}.in",shell=True)
#subprocess.call(f"cat lammps_run/structure.data",shell=True)

if (args.run_md): # or vars(args)['run_md'] != 'False' or vars(args)['run_md'][0] != 'F' or vars(args)['run_md'] == 'True' or vars(args)['run_md'][0] == 'T'):
    print("##################")
    print("Run MD")
    subprocess.call(f"cd lammps_run/ && /data/gtocci/scratch/lammps/build/lmp -in {model_system}.in",shell=True)
    wat_traj = read(f"lammps_run/{model_system}.xyz",index='::10')
    print("MD completed")
    print("##################")

if (args.analysis): # or vars(args)['analysis'] != 'False' or vars(args)['analysis'][0] != 'F' or vars(args)['analysis'] == 'True' or vars(args)['analysis'][0] == 'T'):
    wat_traj = read(f"lammps_run/{model_system}.xyz",index='::10')

    for i in range(len(wat_traj)):
        wat_traj[i].cell = cell_vec_abc
        wat_traj[i].pbc = np.array([True,True,True])

    reader = mda.coordinates.XYZ.XYZReader(f"lammps_run/{model_system}.xyz")
    topology = mda.topology.XYZParser.XYZParser(f"lammps_run/{model_system}.xyz")

    u = mda.Universe(f"lammps_run/{model_system}.xyz")
    u.dimensions = [cell_vec_abc[0], cell_vec_abc[1],cell_vec_abc[2], 90., 90., 90. ]

    O_at = u.select_atoms('name O')
    H_at = u.select_atoms('name H')
    
    Ordf = rdf.InterRDF(O_at, O_at,
                        nbins=75,  # default
                        range=(0.00001, 4.9),  # distance in angstroms
                       )
    Ordf.run()
    
    OHrdf = rdf.InterRDF(O_at, H_at,
                        nbins=75,  # default
                        range=(0.00001, 4.9),  # distance in angstroms
                       )
    OHrdf.run()
    
    HHrdf = rdf.InterRDF(H_at, H_at,
                        nbins=75,  # default
                        range=(0.00001, 4.9),  # distance in angstroms
                       )
    HHrdf.run()
    
    plt.plot(Ordf.bins, Ordf.rdf)
    plt.xlabel('Radius (angstrom)')
    plt.ylabel('Radial distribution')
    
    plt.plot(OHrdf.bins, OHrdf.rdf)
    plt.xlabel('Radius (angstrom)')
    plt.ylabel('Radial distribution')
    
    plt.plot(HHrdf.bins, HHrdf.rdf)
    plt.xlabel('Radius (angstrom)')
    plt.ylabel('Radial distribution')
