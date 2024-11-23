#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sigurdurdet@hi.is # for example uname@hi.is
#SBATCH --partition=gpu-1xA100  # request node from a specific partition
#SBATCH --nodes=1                 # number of nodes
#SBATCH --ntasks-per-node=48      # 48 cores per node (96 in total)
#SBATCH --mem=188G      # MB RAM per cpu core
#SBATCH --hint=nomultithread      # Suppress multithread
#SBATCH --output=slurm_job_output.log   
#SBATCH --error=slurm_job_errors.log   # Logs if job crashes

source ~/.bashrc

ml unuse /hpcapps/libsci-gcc/modules/all
ml use /hpcapps/libsci-tools/modules/all

ml load Anaconda3

conda activate tf

python3 train.py -p ./experiments/logreg -d ./data/Users_isQ_words.csv
