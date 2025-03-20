#!/bin/bash 
# 
#SBATCH --job-name=crica-train-emitch
#SBATCH --output=slurm/slurm-%j.out
# 
#SBATCH --partition=all
#SBATCH --nodes=1 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=4 
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00 
#
#SBATCH --mail-user=emitchell@scu.edu
#SBATCH --mail-type=ALL

DS_PATH=/WAVE/projects/CSEN-342-Wi25/henchmen/CricaVPR/datasets_vg/datasets
EVAL=/WAVE/projects/CSEN-342-Wi25/henchmen/CricaVPR/datasets_vg/datasets/msls
TEST=/WAVE/projects/CSEN-342-Wi25/henchmen/CricaVPR/datasets_vg/datasets/msls

BACKBONE_PATH=/WAVE/projects/CSEN-342-Wi25/henchmen/CricaVPR/zoo/backbone/dinov2_vitb14_pretrain.pth
PRETRAIN_PATH=/WAVE/projects/CSEN-342-Wi25/henchmen/CricaVPR/zoo/pretrain/CricaVPR.pth

module load Anaconda3 

source ~/.bashrc

#conda create -n newcrica1 -y python=3.9

cd /WAVE/projects/CSEN-342-Wi25/henchmen/emitchell/CricaVPR

#conda remove -n newcrica -y python=3.9

conda create -n newcrica6 -y python=3.9
#conda create -n cricavpr -y

conda activate newcrica6

conda install -c pytorch -c nvidia -y faiss-gpu=1.7.2 cudatoolkit=11.4 cuda-nvcc=11.4

pip install faiss-gpu-cu12

pip3 install timm

pip3 install pytorch-metric-learning

#conda install -c pytorch -c nvidia -y faiss-gpu=1.7.2 cudatoolkit=11.4 cuda-nvcc=11.4

#conda install -c pytorch -c nvidia -y faiss-gpu cudatoolkit=12.8 cuda-nvcc=12.8

#conda install -c pytorch -c nvidia -y faiss-gpu=1.7.2 cudatoolkit cuda-nvcc

#conda remove faiss

#conda install faiss-gpu

##

# train command
python3 train.py    --eval_datasets_folder=$DS_PATH \
                    --eval_dataset_name=$EVAL \
                    --foundation_model_path=$BACKBONE_PATH \
                    --epochs_num=10 \
                    --train_batch_size=8 \
                    # --resume

#python3 train.py --eval_datasets_folder="/WAVE/projects/CSEN-342-Wi25/henchmen/CricaVPR/datasets_vg/datasets" --eval_dataset_name="/WAVE/projects/CSEN-342-Wi25/henchmen/CricaVPR/datasets_vg/datasets/msls" --foundation_model_path="/WAVE/projects/CSEN-342-Wi25/henchmen/CricaVPR/zoo/backbone/dinov2_vitb14_pretrain.pth" --epochs_num=10 --train_batch_size=8