#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result/ \
    --summary_dir ./result/log/ \
    --mode test \
    --is_training False \
    --task unet \
    --batch_size 16 \
    --input_dir_LR /home/share/ziyumeng/Yucheng/Data/LR_lipid_test \
    --input_dir_HR /home/share/ziyumeng/Yucheng/Data/LR_lipid_test \
    --perceptual_mode MSE \
    --pre_trained_model True \
    --checkpoint ./trails/SRGAN/model-40000 \

