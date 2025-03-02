#!/bin/bash 
# 
#SBATCH --job-name=crica-eval
#SBATCH --output=slurm/slurm-%j.out
# 
#SBATCH --partition=all
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00 
#
#SBATCH --mail-user=iorais@scu.edu
#SBATCH --mail-type=ALL

DS_PATH=datasets_vg/datasets
EVAL=msls
TEST=msls

PRETRAIN_PATH=zoo/pretrain/CricaVPR.pth

module load Anaconda3 

source ~/.bashrc
conda activate cricavpr

cd /WAVE/projects/CSEN-342-Wi25/henchmen/CricaVPR

# test command
python3 eval.py --eval_datasets_folder=$DS_PATH \
                 --eval_dataset_name=$TEST \
                 --resume=$PRETRAIN_PATH

#python3 eval.py --eval_datasets_folder=datasets_vg/datasets
#                --eval_dataset_name=msls
#                --resume=logs/default/2025-02-07_18-52-38/best_model.pth
#                --pca_dim=4096
#                --pca_dataset_folder=msls/images/train

#python3 eval.py --eval_datasets_folder=datasets_vg/datasets --eval_dataset_name=msls --resume=logs/default/2025-02-07_18-52-38/best_model.pth --pca_dim=4096 --pca_dataset_folder=msls/images/train
#python3 eval.py --eval_datasets_folder=datasets_vg/datasets --eval_dataset_name=msls --resume=zoo/pretrain/CricaVPR.pth --pca_dim=4096 --pca_dataset_folder=msls/images/train
