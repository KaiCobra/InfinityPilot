conda activate infinity-clean;
torchrun \
    --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=12347 \
    train_pilot.py \
    --ep=10 --opt=adamw --cum=3 \
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
    --local_out_path=/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/VAR_ckpt/local_output/pilot \
    --task_type=t2i \
    --bed=/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/VAR_ckpt/checkpoints/pilot \
    # --data_path=data/infinity_toy_data/splits \
    --data_path=/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/SynData/real_data/combined_splits_by_ratio \
    --video_data_path= \
    --exp_name=pilot_debug \
    --project_name=SceneTxtVAR_Debug \
    --tblr=1e-4 \  
    # 降低學習率以避免 NaN
    --pn=0.06M \
    # --model=infinity_pilot_2b \
    --model=2bc8 \
    --lbs=8 \
    --workers=8 \
    --short_cap_prob=0.0 \
    --online_t5=1 \
    --use_streaming_dataset=1 \
    --iterable_data_buffersize=1000 \
    --Ct5=2048 \
    --t5_path=weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 \
    --rush_resume=/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/VAR_ckpt/infinity_2b_reg.pth \
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
    --zero=0 \
    # --save_model_iters_freq=1000 \
    --save_car_epoch_freq=1 \
    --log_freq=50 \
    --checkpoint_type=torch \
    --prefetch_factor=4 \
    --noise_apply_strength=0.3 \
    --noise_apply_layers=13 \
    --apply_spatial_patchify=0 \
    --use_flex_attn=False \
    --pad=128 \
    # --car_resume_path=/path/to/your/car_weights.pth \  # 可選：載入預訓練的 CAR 權重
    --save_car_separately=True