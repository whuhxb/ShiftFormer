#!/bin/bash
#SBATCH --mail-type=END
#SBATCH --mail-user=ma.xu1@northeastern.edu
#SBATCH -N 1
#SBATCH -p ai-jumpstart
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512Gb
#SBATCH --time=1-23:59:00
#SBATCH --output=%j_fcvt_b12.log

source activate timm
cd /scratch/ma.xu1/ShiftFormer/detection
./dist_train.sh configs/mask_rcnn_fcvt_b12_fpn_1x_coco.py 8
