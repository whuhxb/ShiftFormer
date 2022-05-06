#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p ai-jumpstart
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512Gb
#SBATCH --time=1-23:59:00
#SBATCH --output=%j_fcvt_v5_32_TTFF_W_11_11.log

source activate timm
cd /scratch/ma.xu1/ShiftFormer
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./distributed_train.sh 8 /dev/shm/imagenet/ --model fcvt_v5_32_TTFF_W_11_11 -b 256 --lr 1e-3 --drop-path 0.1 --apex-amp --model-ema
