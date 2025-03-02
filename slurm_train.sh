#!/bin/bash 
# 
#SBATCH --job-name=crica-train
#SBATCH --output=slurm/slurm-%j.out
# 
#SBATCH --partition=all
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00 
#
#SBATCH --mail-user=iorais@scu.edu
#SBATCH --mail-type=ALL

DS_PATH=datasets_vg/datasets
EVAL=msls
TEST=msls

BACKBONE_PATH=zoo/backbone/dinov2_vitb14_pretrain.pth
PRETRAIN_PATH=zoo/pretrain/CricaVPR.pth

module load Anaconda3 

source ~/.bashrc
conda activate cricavpr

cd /WAVE/projects/CSEN-342-Wi25/henchmen/CricaVPR

# train command
python3 train.py    --eval_datasets_folder=$DS_PATH \
                    --eval_dataset_name=$EVAL \
                    --foundation_model_path=$BACKBONE_PATH \
                    --epochs_num=1 \
                    # --train_batch_size=8 \
                    # --resume