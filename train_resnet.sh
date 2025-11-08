#!/bin/bash

# check the time and insert to exp name
date=$(date +%Y%m%d_%H%M%S)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=enp4s0
export GLOO_SOCKET_IFNAME=enp4s0
export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun \
    --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    --master_addr=192.168.50.88 --master_port=12347 \
    train_pilot.py \
    --ep=100 --opt=adamw --cum=3 \
    --sche=lin0 \
    --fp16=2 \
    --ada=0.9_0.97 \
    --tini=-1 \
    --tclip=5 \
    --flash=0 \
    --alng=5e-06 \
    --saln=1 \
    --cos=1 \
    --cdec=True \
    --enable_checkpointing=full-block \
    --local_out_path=/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/output_a6000 \
    --task_type=t2i \
    --bed=/mnt/syndata/VAR_ckpt/checkpoints/pilot \
    --data_path=/mnt/syndata/toy_data/ \
    --video_data_path= \
    --exp_name=pilot_debug_$date \
    --project_name=InfinityPilot \
    --tblr=1e-4 \
    --pn=1M \
    --model=2bc8 \
    --lbs=4 \
    --workers=1 \
    --short_cap_prob=0.0 \
    --online_t5=1 \
    --use_streaming_dataset=1 \
    --iterable_data_buffersize=500 \
    --Ct5=2048 \
    --t5_path=weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
    --rush_resume=/mnt/syndata/VAR_ckpt/infinity_2b_reg.pth \
    --vae_type=32 \
    --vae_ckpt=weights/infinity_vae_d32_reg.pth \
    --wp0=0.1 \
    --wp=0.0 \
    --wpe=0.05 \
    --dynamic_resolution_across_gpus=1 \
    --enable_dynamic_length_prompt=0 \
    --reweight_loss_by_scale=1 \
    --add_lvl_embeding_only_first_block=1 \
    --rope2d_each_sa_layer=1 \
    --rope2d_normalized_by_hw=2 \
    --use_fsdp_model_ema=0 \
    --always_training_scales=7 \
    --use_bit_label=1 \
    --zero=3 \
    --save_car_epoch_freq=5 \
    --log_freq=50 \
    --checkpoint_type=torch \
    --prefetch_factor=2 \
    --noise_apply_strength=0.3 \
    --noise_apply_layers=13 \
    --apply_spatial_patchify=0 \
    --use_flex_attn=False \
    --pad=128 \
    --save_car_separately=True \
    --control_depth=4 \
    --special_car_init=merge \
    # --sync_tensorboard=True,
    # --control_resume_path=None \
    # --debug=False, \
