#!/bin/bash
#SBATCH --job-name=data-processing
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=5000
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tien.luong@monash.edu
#SBATCH --output=logs/infer_ot_baseline/%x-%j.out
#SBATCH --error=logs/infer_ot_baseline/%x-%j.err
#SBATCH --partition=defq

conda init bash
source ~/.bashrc
conda activate enclap

# echo ${1}/epoch_${2}
python data/infer_encodec.py --data_path Clotho/train --save_path data/clotho/encodec/train

python data/infer_clap.py --data_path Clotho/test --save_path data/clotho/clap/test --clap_ckpt 630k-audioset-fusion-best.pt
python data/infer_clap.py --data_path Clotho/val --save_path data/clotho/clap/valid --clap_ckpt 630k-audioset-fusion-best.pt
python data/infer_clap.py --data_path Clotho/train --save_path data/clotho/clap/train --clap_ckpt 630k-audioset-fusion-best.pt

