#!/bin/bash
#SBATCH --account=def-sutton
#SBATCH --ntasks-per-node=1         # CPU cores/threads
#SBATCH --cpus-per-task=40
#SBATCH --ntasks=1
#SBATCH --mem=0             # memory per node
#SBATCH --time=00-05:00            # time (DD-HH:MM)
#SBATCH --nodes=1

export OMP_NUM_THREADS=1
parallel -j 39 ./MNIST --config experiment_configs/experiment_mnist_baselines.json --run ::: {0..149}
