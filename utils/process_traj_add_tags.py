from utils import *

wat_traj = read("../datasets/wat_gra_bil_film/wat_pos_frc.extxyz", index = ":")

trajtag = set_tags_frc_constr(wat_traj)

write("../datasets/wat_gra_bil_film/trajtag_pos_frc.extxyz", trajtag[:])
