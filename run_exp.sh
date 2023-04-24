#!/bin/sh

export CUDA_VISIBLE_DEVICES=1

#        python3 implicit_psfs/main_fit_psf.py --system=$1 --batch_size=1024 --lr=1e-4 --epochs=250 --exp=4 --losses = "MSE" --losses_weights=1 --results_folder=$4 --exp_folder="System_$1_Np_$2" --Nz=1 --Nw=25 --Nx=1 --Ny=1 --Nu=128 --Nv=128 --training_points 1 $2 1 1 --win_reg 1 3 1 1 3 3 --hidden_layers=6 --hidden_features=400 --model=$3 --omega=0.5 --scale=1 --std_gaussian 1.0 1.0 0.01 0.01 1.0 1.0  --regularization_weights=1e-6 --start_z=-0.5 --end_z=0.5

if [ $1 = "cover" ]; then

    python "main cover.py" --cover=$2
else
    python "main snr.py" --snr=$2
fi





