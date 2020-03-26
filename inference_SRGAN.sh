#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --output_dir ./trails/results/ \
    --summary_dir ./trails/result/log/ \
    --mode inference \
    --is_training False \
    --task unet \
    --input_dir_LR /home/share/ziyumeng/Yucheng/Data/LR_lipid_test \
    --perceptual_mode MSE \
    --pre_trained_model True \
    --checkpoint ./trails/SRGAN/model-40000 \
