from utils import *

cell_vec_abc = read_cell("celldata.dat", "./gra_bi")

data_dir = "gra_bi_film/"

wat_traj = read(data_dir + "aimd_hetero-pos-1.xyz", index="::200")
wat_frc = read(data_dir + "aimd_hetero-frc-1.xyz", index="::200")

natoms, positions, energies = MD_reader_xyz(
    "aimd_hetero-pos-1.xyz", data_dir, no_skip=1)

MD_writer_xyz(
    wat_traj,
    wat_frc,
    cell_vec_abc,
    energies,
    data_dir,
    "wat_pos_frc.extxyz",
    conv_frc=1.0,
    conv_ener=27.211399)

last_conf = sort(wat_traj[-1])

write('gra_bi_film/last_conf.xyz', last_conf)
