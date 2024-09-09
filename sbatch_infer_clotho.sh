#!/bin/bash
#SBATCH --job-name=enclap-top-p0.7
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5000
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tien.luong@monash.edu
#SBATCH --output=logs/abalation_study/cl_clotho/%x-%j.out
#SBATCH --error=logs/abalation_study/cl_clotho/%x-%j.err
#SBATCH --partition=defq

conda init bash
source ~/.bashrc
conda activate enclap

# echo ${1}/epoch_${2}
python evaluate.py --ckpt ${1}/epoch_${2} --clap_ckpt 630k-audioset-fusion-best.pt --test_csv csv/clotho/test.csv --audio_path Clotho/