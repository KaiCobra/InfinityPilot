#!/bin/bash
date=$(date +%Y%m%d_%H%M%S)

export OMP_NUM_THREADS=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO

# 介面名稱請用 eth0（TWCC 容器常見），不要 enp4s0
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

# 單機多卡必備
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500

remote_local_sel=2

if [ $remote_local_sel -eq 1 ]; then
  export DATA_PATH=/work/u7677435/Upload/car_dataset
  export VIDEO_DATA_PATH=
  export BED=/work/u7677435/Upload/VAR_ckpt/checkpoints/pilot
  export LOCAL_OUT_PATH=/work/u7677435/Upload/output_v100_ds
  export T5_PATH=/work/u7677435/Upload/weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001
  export VAE_CKPT=/work/u7677435/Upload/weights/infinity_vae_d32_reg.pth
  export RUSH_RESUME=/work/u7677435/Upload/weights/mm_2b.pth

else
  export DATA_PATH=/mnt/syndata/car_dataset
  export VIDEO_DATA_PATH=
  export BED=/mnt/syndata/VAR_ckpt/checkpoints/pilot
  export LOCAL_OUT_PATH=/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/output_v100_ds
  export T5_PATH=weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001
  export VAE_CKPT=weights/infinity_vae_d32_reg.pth
  export RUSH_RESUME=weights/mm_2b.pth
fi

missing_paths=()
[ -d "$T5_PATH" ] || missing_paths+=("T5_PATH=$T5_PATH")
[ -f "$VAE_CKPT" ] || missing_paths+=("VAE_CKPT=$VAE_CKPT")
[ -f "$RUSH_RESUME" ] || missing_paths+=("RUSH_RESUME=$RUSH_RESUME")
[ -d "$DATA_PATH" ] || missing_paths+=("DATA_PATH=$DATA_PATH")
[ -z "$VIDEO_DATA_PATH" ] || { [ -d "$VIDEO_DATA_PATH" ] || missing_paths+=("VIDEO_DATA_PATH=$VIDEO_DATA_PATH"); }

if [ ${#missing_paths[@]} -ne 0 ]; then
  echo "[ERROR] The following paths do not exist for remote_local_sel=$remote_local_sel:" >&2
  for p in "${missing_paths[@]}"; do
    echo "  - $p" >&2
  done
  echo "Please update train_pilot_deepspeed.sh with the correct local locations before launching DeepSpeed." >&2
  exit 1
fi

# 可依實際 GPU 數量調整 --num_gpus
deepspeed --num_gpus=1 \
  train_pilot_deepspeed.py \
  --deepspeed \
  --deepspeed_config configs/ds_zero2.json \
  --ep=100 --opt=adamw --cum=3 \
  --sche=lin0 \
  --fp16=1 \
  --ada=0.9_0.97 \
  --tclip=5 \
  --flash=0 \
  --alng=5e-06 \
  --saln=1 \
  --cos=1 \
  --cdec=True \
  --tini=-1 \
  --oeps=1e-6 \
  --norm_eps=1e-4 \
  --local_out_path=$LOCAL_OUT_PATH \
  --task_type=t2i \
  --bed=$BED \
  --data_path=$DATA_PATH \
  --video_data_path=$VIDEO_DATA_PATH \
  --exp_name=pilot_v100_ds_$date \
  --project_name=InfinityPilot \
  --tblr=5e-4 \
  --pn=1M \
  --model=2bc8 \
  --lbs=1 \
  --workers=1 \
  --short_cap_prob=0.0 \
  --online_t5=1 \
  --use_streaming_dataset=1 \
  --iterable_data_buffersize=500 \
  --Ct5=2048 \
  --t5_path=$T5_PATH \
  --rush_resume=$RUSH_RESUME \
  --vae_type=32 \
  --vae_ckpt=$VAE_CKPT \
  --wp0=0.01 \
  --wp=0.1 \
  --wpe=0.05 \
  --dynamic_resolution_across_gpus=1 \
  --enable_dynamic_length_prompt=0 \
  --reweight_loss_by_scale=1 \
  --add_lvl_embeding_only_first_block=1 \
  --rope2d_each_sa_layer=1 \
  --rope2d_normalized_by_hw=2 \
  --use_fsdp_model_ema=0 \
  --use_bit_label=1 \
  --save_car_epoch_freq=1 \
  --log_freq=10 \
  --checkpoint_type=torch \
  --prefetch_factor=2 \
  --noise_apply_strength=0.3 \
  --noise_apply_layers=13 \
  --apply_spatial_patchify=0 \
  --use_flex_attn=False \
  --pad=128 \
  --save_car_separately=True \
  --car_depth=8 \
  --special_car_init=merge \
  --disable_car_fusion=False \
  --disable_car_merge=False \
  --rms_norm=True \
  --enable_checkpointing=full-block \
  --initial_training_scales 7 \
  --enable_dynamic_scales=True \
  --dynamic_scale_target 13 \
  --dynamic_scale_patience_transition 10 \
  --dynamic_scale_patience_low=200 \
  --dynamic_scale_patience_high=2000 \
  --dynamic_scale_loss_window=150 \
  --dynamic_scale_loss_delta=5e-3
