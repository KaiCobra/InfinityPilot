"""
InfinityPilot 訓練配置文件
"""

# 模型配置
MODEL_CONFIG = {
    'embed_dim': 1024,
    'depth': 16,
    'num_heads': 16,
    'mlp_ratio': 4.0,
    'drop_rate': 0.0,
    'drop_path_rate': 0.0,
    'norm_eps': 1e-6,
    'rms_norm': False,
    'shared_aln': False,
    'head_aln': True,
    'cond_drop_rate': 0.1,
    'cross_attn_layer_scale': -1.0,
    'tau': 1,
    'cos_attn': True,
    'swiglu': False,
    'customized_flash_attn': False,
    'fused_mlp': False,
    'fused_norm': False,
    'block_chunks': 1,
    'checkpointing': None,
    'use_flex_attn': False,
    'add_lvl_embeding_only_first_block': 1,
    'use_bit_label': 1,
    'rope2d_each_sa_layer': 0,
    'rope2d_normalized_by_hw': 0,
    'video_frames': 1,
    'always_training_scales': 20,
    'apply_spatial_patchify': 0,
}

# 訓練配置
TRAINING_CONFIG = {
    'batch_size': 4,
    'learning_rate': 1e-4,
    'epochs': 100,
    'warmup_steps': 1000,
    'grad_accum_steps': 1,
    'label_smooth': 0.1,
    'max_steps': 100000,
    'use_amp': True,
    'fp16': True,
    'log_every_n_steps': 50,
    'val_check_interval': 1.0,
}

# 數據配置
DATA_CONFIG = {
    'text_channels': 4096,
    'text_maxlen': 77,
    't5_model_type': 'flan-t5-xl',
    't5_extra_len': 0,
}

# 路徑配置
PATH_CONFIG = {
    'checkpoint_dir': './checkpoints/infinity_pilot',
    'tensorboard_log_dir': './tb_logs/infinity_pilot',
    'infinity_ckpt': None,  # 預訓練 Infinity 模型路徑
    'vae_ckpt': None,       # VAE 模型路徑
    'train_data_path': None,
    'val_data_path': None,
}

# 硬體配置
HARDWARE_CONFIG = {
    'gpus': 1,
    'seed': 42,
}

# 控制配置
CONTROL_CONFIG = {
    'use_control': True,
    'control_scale': 1.0,
}

# 合併所有配置
CONFIG = {
    **MODEL_CONFIG,
    **TRAINING_CONFIG,
    **DATA_CONFIG,
    **PATH_CONFIG,
    **HARDWARE_CONFIG,
    **CONTROL_CONFIG,
}

# 凍結參數配置
FREEZE_CONFIG = {
    # 需要凍結的 Infinity 模塊
    'freeze_modules': [
        'pos_start', 'lvl_embed', 'norm0_ve', 'word_embed', 'shared_ada_lin',
        'text_norm', 'text_proj_for_sos', 'text_proj_for_ca', 'cfg_uncond',
        'blocks', 'unregistered_blocks', 'block_chunks', 'head_nm', 'head'
    ],
    
    # 需要訓練的 CAR 模塊
    'trainable_modules': [
        'car_control_convs', 'car_var_conv', 'car_blocks', 
        'car_skip_norm', 'car_skip_linear'
    ]
}

def get_config():
    """獲取配置"""
    return CONFIG

def get_freeze_config():
    """獲取凍結配置"""
    return FREEZE_CONFIG

if __name__ == '__main__':
    # 打印配置信息
    print("InfinityPilot Training Configuration:")
    print("=" * 50)
    
    for category, config in [
        ("Model", MODEL_CONFIG),
        ("Training", TRAINING_CONFIG),
        ("Data", DATA_CONFIG),
        ("Paths", PATH_CONFIG),
        ("Hardware", HARDWARE_CONFIG),
        ("Control", CONTROL_CONFIG),
    ]:
        print(f"\n{category} Configuration:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print(f"\nFreeze Configuration:")
    freeze_config = get_freeze_config()
    print(f"  Freeze modules: {len(freeze_config['freeze_modules'])}")
    print(f"  Trainable modules: {len(freeze_config['trainable_modules'])}")
