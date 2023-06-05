#!/bin/bash -e

abinitio-train-workflow  --run_md --method nequip --no-train --no-deploy --no-data_load  --no-analysis --cp2k_file_name trajtag_pos_frc.extxyz   --n_steps 100 --cp2k_exe /home/ubuntu/software/cp2k/exe/local_cuda
