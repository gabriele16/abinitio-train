#!/bin/bash -e

abinitio-train-workflow  --deploy --no-train --no-data_load --no-run_md --no-analysis --results-dir --system_name water_graphene_bi


#--data_dir ../datasets/wat_bil_gra_aimd --dataset trajtag_pos_frc.extxyz  --system_name water_graphene_bi --n_train 400 --n_val 50 --max_epochs 100 --default_dtype float64 --cutoff 5.0 --num_layers 2
