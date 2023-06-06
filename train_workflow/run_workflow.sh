#!/bin/bash -e

abinitio-train-workflow  --train --method nequip --no-data_load --deploy --run_md --mask_labels --no-analysis --data_dir /mnt/abinitio-train/datasets/wat_bil_gra_aimd --dataset trajtag_pos_frc.extxyz  --system_name water_graphene_bi --n_train 400 --n_val 50 --max_epochs 1 --default_dtype float64 --cutoff 5.0 --num_layers 2 --cp2k_coord_file_name trajtag_pos_frc.extxyz   --n_steps 100000 --cp2k_exe /home/ubuntu/software/cp2k/exe/local_cuda/cp2k.ssmp --unit_coords angstrom --unit_energy eV --unit_forces eV*angstrom^-1
