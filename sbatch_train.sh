#!/bin/bash
#SBATCH --job-name=enclap-marginloss-b128-noacce
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=5000
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tien.luong@monash.edu
#SBATCH --output=logs/%x.out
#SBATCH --error=logs/%x.err
#SBATCH --partition=A100

conda init bash
source ~/.bashrc
conda activate enclap

# accelerate config
# CFG_PATH="cfg/audiocaps/large.yaml"
# accelerate launch --multi_gpu --main_process_port=1200 train.py $CFG_PATH
python train_fn_cl.py cfg/audiocaps/base.yaml