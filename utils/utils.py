import os
import re
from ase.build import sort
import numpy as np
import site
from ase.io import read, write


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

def read_cell(f, data_dir):
    filename = os.path.join(data_dir, f)
    fo = open(filename, "r")
    cell_list_abc = fo.read().split("\n")[:-1]
    cell_vec_abc = np.array(
        [list(map(float, lv.split())) for lv in cell_list_abc]
    ).squeeze()
    return cell_vec_abc


def Energy_reader_xyz(f, data_dir, no_skip = 0):
    filename = os.path.join(data_dir, f)
    with open(filename, "r") as fo:
        file_contents = fo.read()

    pattern = r"E = ([^\n]*)"
    matches = re.findall(pattern, file_contents)

    if no_skip == 0:

      ener = [float(en) for en in matches]
    else:
      
      ener = [float(en) for en in matches[::no_skip]]

    return ener

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
        ase_traj[i].set_tags(ase_traj[i].get_forces()[:,dim] == 0.0)
    return ase_traj
