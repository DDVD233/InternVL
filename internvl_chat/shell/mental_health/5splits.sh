#!/bin/bash

for i in {1..5}; do
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PYTHONPATH:./ LAUNCHER=pytorch python internvl/train/internvl_chat_audio_finetune.py \
    --model_name_or_path "/home/dvd/data/outputs/all_public_backbones_8B" \
    --conv_style "internlm2-chat" \
    --output_dir "/home/dvd/data/outputs/phq9_8B_split$i" \
    --meta_path "shell/data/behavioral_phq_split$i.json" \
    --overwrite_output_dir True \
    --force_image_size 448 \
    --max_dynamic_patch 9 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.1 \
    --pad2square False \
    --freeze_llm True \
    --freeze_mlp False \
    --freeze_backbone True \
    --vision_select_layer -1 \
    --use_data_resampling False \
    --dataloader_num_workers 1 \
    --bf16 True \
    --num_train_epochs 12 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 140 \
    --save_total_limit 3 \
    --learning_rate 4e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_seq_length 8192 \
    --do_train True \
    --grad_checkpoint True \
    --group_by_length True \
    --dynamic_image_size True \
    --use_thumbnail True \
    --ps_version 'v2' \
    --report_to wandb \
    --use_class_weights True \
    --unfreeze_vit_layers 2
done