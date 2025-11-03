# train_pilot_ori.py
import gc
import json
import math
import os
import random
import sys
import time
import traceback
from collections import deque
from contextlib import nullcontext
from distutils.util import strtobool
from typing import List, Optional, Tuple
import os.path as osp
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from torch.nn import functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
import torch.distributed as tdist

import infinity.utils.dist as dist
from infinity.dataset.build_pilot import build_t2i_dataset
from infinity.utils.save_and_load import CKPTSaver, auto_resume
from infinity.utils import arg_util, misc
from infinity.utils.wandb_utils import initialize as init_wandb, log_metrics, finish
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from debug_utils.frozen_state import debug_parameter_freeze_status, debug_parameter_number
from infinity.utils.build_model_optimizer_pilot import build_model_optimizer_pilot

enable_timeline_sdk = False

# Mock timeline object to avoid undefined variable issues
class MockTimeline:
    def inc_step(self): pass
    def flush(self): pass

if not enable_timeline_sdk:
    ndtimeline = MockTimeline()

def save_checkpoint_pilot(saver, args, trainer, epoch, iteration, acc_str):
    """為 InfinityPilot 保存 CAR 權重 (T5風格)"""
    
    # 創建保存目錄
    save_dir = os.path.join(args.local_out_path, f'car_weights_ep{epoch:04d}_it{iteration:06d}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 獲取正確的模型對象
    if hasattr(trainer, 'gpt'):
        model = trainer.gpt  # FSDP wrapped model
    elif hasattr(trainer, 'gpt_wo_ddp'):
        model = trainer.gpt_wo_ddp  # unwrapped model
    else:
        raise AttributeError("Cannot find model in trainer")
    
    # 如果當前模型沒有 CAR 模塊，直接跳過保存
    base_model = model.module if hasattr(model, 'module') else model
    has_car = getattr(base_model, 'has_car_modules', None)
    if callable(has_car) and not has_car():
        print("[warn] CAR modules not initialized; skipping car checkpoint save.")
        return None

    # 分離 CAR 權重
    car_weights = {}
    infinity_weights = {}
    
    try:
        state_dict = model.state_dict()
    except AssertionError as exc:
        print(f"[warn] Skipping CAR checkpoint save because state_dict fetch failed: {exc}")
        return None

    for name, param in state_dict.items():
        # 檢查是否為 CAR 參數
        if any(car_prefix in name for car_prefix in ['car_', 'control_']):
            car_weights[name] = param.cpu()
        else:
            infinity_weights[name] = param.cpu()
    
    # 只保存 CAR 權重
    torch.save(car_weights, os.path.join(save_dir, 'car_weights.pth'))
    
    # 創建符號鏈接到最新的權重（方便resume）
    latest_link = os.path.join(args.local_out_path, 'latest_car_weights')
    if os.path.islink(latest_link):
        os.unlink(latest_link)
    elif os.path.exists(latest_link):
        os.remove(latest_link)
    
    os.symlink(os.path.basename(save_dir), latest_link)
    
    print(f"Saved CAR weights to: {save_dir}")
    print(f"  CAR parameters: {len(car_weights)}")
    print(f"  Infinity parameters (not saved): {len(infinity_weights)}")
    return save_dir


def build_everything_from_args(args: arg_util.Args, saver):
    # Set default scale_schedule if not provided
    # if not hasattr(args, 'scale_schedule') or args.scale_schedule is None:
    #     args.scale_schedule = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # Default from infinity.py
    #     print(f"Setting default scale_schedule: {args.scale_schedule}")
    num_scales = args.always_training_scales
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
    # 使用 ratio=1.0 作為 placeholder
    placeholder_scales = dynamic_resolution_h_w[1.0][args.pn]['scales'][:num_scales]

    if getattr(args, 'task_type', 't2i') == 't2i':
        args.scale_schedule = [(1, h, w) for (_, h, w) in placeholder_scales]
    else:
        args.scale_schedule = placeholder_scales
    
    print(f"Setting placeholder scale_schedule (will be overridden during training): {args.scale_schedule}")

    # set seed
    args.set_initial_seed(benchmark=True)
    if args.seed is not None and not args.rand: # check the randomness
        misc.check_randomness(args)

    # build data
    iters_train, ld_train, ld_val = build_dataloaders(args)   
    train_h_div_w_list = list(ld_train.dataset.h_div_w_template2generator.keys())
    print(f"{train_h_div_w_list=}")
    args.train_h_div_w_list = train_h_div_w_list 

    # load VAE
    print(f'Load vae form {args.vae_ckpt}')
    if not os.path.exists(args.vae_ckpt):
        vae_ckpt = {}
    else:
        vae_ckpt = torch.load(args.vae_ckpt, map_location='cpu')

    # build models. Note that here gpt is the causal VAR transformer which performs next scale prediciton with text guidance
    text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim = build_model_optimizer_nonused(args, vae_ckpt)
    
    # IMPORTANT: import heavy package `InfinityPilotTrainer` after the Dataloader object creation/iteration to avoid OOM
    from trainer_pilot import InfinityPilotTrainer
    # build trainer
    trainer = InfinityPilotTrainer(
        is_visualizer=dist.is_visualizer(), device=args.device, raw_scale_schedule=args.scale_schedule,
        vae_local=vae_local, gpt_wo_ddp=gpt_wo_ddp, gpt=gpt_ddp, ema_ratio=args.tema, max_it=iters_train * args.ep,
        gpt_opt=gpt_optim, label_smooth=args.ls, z_loss_ratio=args.lz, eq_loss=args.eq, xen=args.xen,
        dbg_unused=args.dbg, zero=args.zero, vae_type=args.vae_type,
        reweight_loss_by_scale=args.reweight_loss_by_scale, gpt_wo_ddp_ema=gpt_wo_ddp_ema, 
        gpt_ema=gpt_ddp_ema, use_fsdp_model_ema=args.use_fsdp_model_ema, other_args=args,
    )
    trainer.register_text_encoder(text_tokenizer, text_encoder)
    
    # auto resume from broken experiment
    auto_resume_info, start_ep, start_it, acc_str, eval_milestone, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    print(f'global bs={args.glb_batch_size}, local bs={args.batch_size}')
    print(f'initial args:\n{str(args)}')
    args.dump_log()

    if start_ep == args.ep:
        args.dump_log()
        print(f'[vgpt] AR finished ({acc_str}), skipping ...\n\n')
        return None
    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True) # don't load vae again
    if auto_resume_info is not None and len(auto_resume_info):
        # check is there a single CAR weights
        car_ckpt_path = getattr(args, 'car_resume_path', None)
        trainer.load_state_dict(trainer_state, strict=False, skip_vae=True, car_ckpt_path=car_ckpt_path)
    start_it = start_it % iters_train
    print(f"{start_it=}, {iters_train=}")
    
    del vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim
    dist.barrier()
    return (
        text_tokenizer, text_encoder, trainer,
        start_ep, start_it, acc_str, eval_milestone, iters_train, ld_train, ld_val
    )


def build_model_optimizer_nonused(args, vae_ckpt):
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from infinity.models.infinity_pilot import InfinityPilot, MultipleLayers
    from infinity.models.init_param import init_weights
    from infinity.utils.amp_opt import AmpOptimizer
    from infinity.utils.lr_control import filter_params
    
    # disable builtin initialization for speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    
    # Build VAE only, skip the intermediate InfinityPilot creation
    if args.vae_type in [8,16,18,20,24,32,64,128]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type # 18
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae_local = vae_model(vae_ckpt, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                              encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(args.model_init_device)
        if args.fake_vae_input:
            vae_local.encoder = None
            vae_local.decoder = None
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"vae_type {args.vae_type} not supported")
    
    del vae_ckpt
    
    # Set scale_schedule from args if not provided
    # if not hasattr(args, 'scale_schedule') or args.scale_schedule is None:
    #     args.scale_schedule = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # Default from infinity.py
    #     print(f"Setting default scale_schedule: {args.scale_schedule}")
    
    # Load checkpoint for architecture detection if available - MUST BE BEFORE model dimension detection
    infinity_checkpoint = None
    if args.rush_resume and os.path.exists(args.rush_resume):
        print(f"Loading checkpoint for architecture detection: {args.rush_resume}")
        try:
            if os.path.isdir(args.rush_resume):
                infinity_base_path = os.path.join(args.rush_resume, 'infinity_base_weights.pth')
                if os.path.exists(infinity_base_path):
                    infinity_checkpoint = torch.load(infinity_base_path, map_location='cpu')
            else:
                cpu_d = torch.load(args.rush_resume, 'cpu', weights_only=False)
                if 'trainer' in cpu_d:
                    infinity_checkpoint = cpu_d['trainer']['gpt_fsdp']
                elif 'gpt_fsdp' in cpu_d:
                    infinity_checkpoint = cpu_d['gpt_fsdp']
                else:
                    infinity_checkpoint = cpu_d
        except Exception as e:
            print(f"Could not load checkpoint for architecture detection: {e}")
    
    # Get model dimensions - first try to detect from checkpoint, then from args.model
    embed_dim, depth, num_heads = None, None, None
    
    # Try to detect dimensions from checkpoint if available
    if infinity_checkpoint is not None:
        try:
            # Check key parameter shapes to determine model size
            if 'pos_start' in infinity_checkpoint:
                checkpoint_embed_dim = infinity_checkpoint['pos_start'].shape[-1]
                if checkpoint_embed_dim == 2048:
                    embed_dim, depth, num_heads = 2048, 32, 16  # 2B model
                    print("Detected 2B model from checkpoint (embed_dim=2048)")
                elif checkpoint_embed_dim == 4608:
                    embed_dim, depth, num_heads = 4608, 58, 36  # 20B model  
                    print("Detected 20B model from checkpoint (embed_dim=4608)")
                elif checkpoint_embed_dim == 1024:
                    # Custom smaller model
                    embed_dim, depth, num_heads = 1024, 16, 16
                    print("Detected custom 1B model from checkpoint (embed_dim=1024)")
                else:
                    print(f"Unknown embed_dim {checkpoint_embed_dim} from checkpoint")
        except Exception as e:
            print(f"Could not detect model size from checkpoint: {e}")
    
    # Fallback to args.model if checkpoint detection failed
    if embed_dim is None:
        model_str = getattr(args, 'gpt', args.model).replace('-', '_')
        if '2b' in model_str:
            embed_dim, depth, num_heads = 2048, 32, 16  # Corrected 2B dimensions
        elif '8b' in model_str:
            # No official 8B model, assume similar to other models
            embed_dim, depth, num_heads = 3072, 40, 24  # Interpolated dimensions
        elif '20b' in model_str:
            embed_dim, depth, num_heads = 4608, 58, 36  # Corrected 20B dimensions
        else:
            # Default values - use 2B as default since it's most common
            embed_dim, depth, num_heads = 2048, 32, 16
            print(f"Unknown model size {model_str}, using 2B default dimensions")
        print(f"Using model dimensions from args: embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}")
    
    # Get kwargs for InfinityPilot creation
    gpt_kw = dict(args.__dict__.items())
    gpt_kw.update(
        vae_local=vae_local,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,  # Standard value
    )
    # Get the kwargs for InfinityPilot creation
    gpt_kw = dict(
        # missing params
        # pretrained=False, global_pool='',
        vae_local=vae_local,
        text_channels=args.Ct5,
        # norm_eps=args.norm_eps,
        text_maxlen=args.tlen,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,              # Standard value 
        drop_path_rate=getattr(args, 'dp', 0.0) if hasattr(args, 'dp') and args.dp >= 0 else 0.0,
        norm_eps=getattr(args, 'norm_eps', 1e-6),
        cond_drop_rate=args.cfg, 
        rand_uncond=args.rand_uncond, 
        drop_rate=args.drop,
        raw_scale_schedule=args.scale_schedule,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        use_bit_label=args.use_bit_label, 
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn, 
        train_h_div_w_list=getattr(args, 'train_h_div_w_list', None), 
        video_frames=getattr(args, 'video_frames', 1),
        always_training_scales=args.always_training_scales,
        apply_spatial_patchify=args.apply_spatial_patchify,
        inference_mode=getattr(args, 'inference_mode', False),
        # Additional args-based parameters
        shared_aln=getattr(args, 'saln', False),
        head_aln=getattr(args, 'haln', True),
        cross_attn_layer_scale=getattr(args, 'ca_gamma', -1.0),
        nm0=getattr(args, 'nm0', False),
        tau=getattr(args, 'tau', 1),
        cos_attn=getattr(args, 'cos', True),
        swiglu=getattr(args, 'swi', False),
        head_depth=getattr(args, 'dec', 1),
        top_p=getattr(args, 'tp', 0.0),
        top_k=getattr(args, 'tk', 0.0),
        rms_norm=getattr(args, 'rms', False),
        customized_flash_attn=getattr(args, 'customized_flash_attn', False),
        fused_mlp=getattr(args, 'fused_mlp', False),
        fused_norm=getattr(args, 'fused_norm', False),
        checkpointing=getattr(args, 'enable_checkpointing', False),     # added
        pad_to_multiplier=getattr(args, 'pad_to_multiplier', 1),       # added
        use_flex_attn=args.use_flex_attn,                   # added 
        batch_size=getattr(args, 'batch_size', 4),          # added
        car_depth=getattr(args, 'car_depth', 16),          # added
        save_car_separately=getattr(args, 'save_car_separately', True),  # added
        car_condition_channels=getattr(args, 'car_condition_channels', 3),
        disable_car_fusion=getattr(args, 'disable_car_fusion', False),
        disable_car_merge=getattr(args, 'disable_car_merge', False),

    )
    
    # # Add optional parameters if they exist
    if getattr(args, 'num_block_chunks', 1) > 1:
        gpt_kw['block_chunks'] = args.num_block_chunks
    else:
        gpt_kw['block_chunks'] = 1
    
    
    # Create InfinityPilot with memory optimization
    # Step 1: Create model without loading weights first
    gpt_wo_ddp = InfinityPilot(
        infinity_base_model=None,  # Don't load weights yet
        init_car_modules=False,    # Don't init CAR yet
        freeze_infinity=True,     # Freeze infinity
        **gpt_kw
    )

    if getattr(args, 'disable_car_fusion', False):
        print("[debug] Disabling CAR fusion as requested.")
        gpt_wo_ddp.disable_car_fusion = True
    if getattr(args, 'disable_car_merge', False):
        print("[debug] Skipping CAR weight merge; using random initialization.")
        gpt_wo_ddp.disable_car_merge = True
    
    # Step 2: Load infinity weights if available (memory efficient)
    if infinity_checkpoint is not None:
        print("[Memory Optimization] Loading Infinity weights...")
        gpt_wo_ddp.load_infinity_weights(infinity_checkpoint)
        print("\nVerifying loaded pretrained weights...")
        for name, param in gpt_wo_ddp.named_parameters():
            if not param.requires_grad: # 只檢查凍結的 Infinity 權重
                if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                    print(f"!!!!!!!!!! FATAL: NaN/Inf found in loaded parameter: {name}")
                    raise RuntimeError("Pretrained weights are corrupted.")
        print("✓ Pretrained weights are clean.")
        # Clear checkpoint from memory immediately after loading
        del infinity_checkpoint
        torch.cuda.empty_cache()
    
    # Step 3: Initialize CAR modules and freeze infinity
    gpt_wo_ddp._init_car_modules()
    gpt_wo_ddp.freeze_infinity_parameters()
    # Debug console: {(name, p.grad.shape) for name, p in gpt_wo_ddp.named_parameters() if p.grad is not None}

    # verify freeze state
    # debug_parameter_freeze_status(gpt_wo_ddp)
    
    # Memory monitoring after model creation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after model creation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    if args.tini < 0:
        args.tini = math.sqrt(1 / gpt_wo_ddp.C / 3)
    
    gpt_kw['car_resume_path'] = getattr(args, 'car_resume_path', None)
    gpt_kw['special_car_init'] = getattr(args, 'special_car_init', None)
    if gpt_kw['car_resume_path'] is not None and gpt_kw['car_resume_path'] != '' and gpt_kw['car_resume_path'].lower() != 'none':
        print(f"Resuming CAR weights from: {gpt_kw['car_resume_path']}")
        car_ckpt = torch.load(gpt_kw['car_resume_path'], map_location='cpu')
        gpt_wo_ddp.load_car_weights(car_ckpt)
        del car_ckpt
        torch.cuda.empty_cache()
    else:
        # Use default CAR initialization (simple Xavier)
        print("Didn't load CAR weights; using default initialization.")
    
    # Update word embedding settings if needed
    if args.rwe:
        gpt_wo_ddp.word_embed.weight.requires_grad = False
        torch.nn.init.trunc_normal_(gpt_wo_ddp.word_embed.weight.data, std=1.5 * math.sqrt(1 / gpt_wo_ddp.C / 3))
        if hasattr(gpt_wo_ddp.word_embed, 'bias'):
            gpt_wo_ddp.word_embed.bias.requires_grad = False
            gpt_wo_ddp.word_embed.bias.data.zero_()
    
    # Only count CAR parameters for training (過濾掉凍結的參數)
    # 只為 requires_grad=True 的參數創建 ndim_dict
    ndim_dict = {name: para.ndim for name, para in gpt_wo_ddp.named_parameters() if para.requires_grad}
    
    print(f'[PT] GPT model = {gpt_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    count_trainable_p = lambda m: f'{sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6:.2f}'
    
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.quant', vae_local.quantize)
    )]))
    print(f'[PT][#para] GPT total={count_p(gpt_wo_ddp)}, trainable={count_trainable_p(gpt_wo_ddp)}')
    
    # 計算 Infinity 和 CAR 參數數量
    debug_parameter_number(gpt_wo_ddp)

    gpt_uncompiled = gpt_wo_ddp
    gpt_wo_ddp = args.compile_model(gpt_wo_ddp, args.tfast)

    # Move model to GPU before creating DDP/FSDP wrapper
    gpt_wo_ddp = gpt_wo_ddp.to(args.device)
    print(f"Model moved to device: {args.device}")

    # For InfinityPilot with frozen base model, EMA is usually not needed for CAR training
    gpt_wo_ddp_ema = None
    gpt_ddp_ema = None
    
    if args.zero:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import ModuleWrapPolicy
        from torch.distributed.device_mesh import init_device_mesh

        # use mix prec: https://github.com/pytorch/pytorch/issues/76607
        if gpt_wo_ddp.num_block_chunks == 1:  # no chunks
            auto_wrap_policy = ModuleWrapPolicy([type(gpt_wo_ddp.unregistered_blocks[0]), ])
        else:
            auto_wrap_policy = ModuleWrapPolicy([MultipleLayers, ])
        
        if args.enable_hybrid_shard:
            sharding_strategy = ShardingStrategy.HYBRID_SHARD if args.zero == 3 else ShardingStrategy._HYBRID_SHARD_ZERO2
            world_size = dist.get_world_size()
            assert world_size % args.inner_shard_degree == 0
            assert args.inner_shard_degree > 1 and args.inner_shard_degree < world_size
            device_mesh = init_device_mesh('cuda', (world_size // args.inner_shard_degree, args.inner_shard_degree))
        else:
            sharding_strategy = ShardingStrategy.FULL_SHARD if args.zero == 3 else ShardingStrategy.SHARD_GRAD_OP
            device_mesh = None
        print(f'{">" * 45 + " " * 5} FSDP INIT with {args.zero=} {sharding_strategy=} {auto_wrap_policy=} {" " * 5 + "<" * 45}', flush=True)
        
        gpt_ddp: FSDP = FSDP(
            gpt_wo_ddp, 
            device_id=dist.get_local_rank(),
            sharding_strategy=sharding_strategy, 
            mixed_precision=None,
            auto_wrap_policy=auto_wrap_policy, 
            use_orig_params=True, 
            sync_module_states=True, 
            limit_all_gathers=True,
            device_mesh=device_mesh,
        ).to(args.device)
        
        # For InfinityPilot CAR training, we typically don't need EMA since base model is frozen
        if args.use_fsdp_model_ema:
            print("[INFO] FSDP EMA disabled for InfinityPilot CAR training (base model is frozen)")
            # Optionally, you can enable EMA for CAR modules only if needed:
            # gpt_wo_ddp_ema = copy.deepcopy(gpt_wo_ddp)
            # gpt_ddp_ema: FSDP = FSDP(gpt_wo_ddp_ema, ...)
    else:
        ddp_class = DDP if dist.initialized() else misc.NullDDP
        # Enable find_unused_parameters for InfinityPilot with frozen base model
        find_unused = True  # Always enable for InfinityPilot CAR training
        gpt_ddp: DDP = ddp_class(gpt_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=find_unused, broadcast_buffers=False)
    torch.cuda.synchronize()
    
    # 重要：DDP 包装後重新確保凍結狀態
    print("[DEBUG] Re-verifying and enforcing freeze status after DDP wrapping:")
    base_model = gpt_ddp.module if hasattr(gpt_ddp, 'module') else gpt_ddp
    
    # 强制重新冻结 Infinity 参数
    frozen_count = 0
    for name, param in base_model.named_parameters():
        if not any(car_prefix in name for car_prefix in ['car_', 'control_']):
            if param.requires_grad:
                param.requires_grad = False
                frozen_count += 1
    
    if frozen_count > 0:
        print(f"  Re-frozen {frozen_count} Infinity parameters after DDP wrapping")
    
    # 验证最终状态
    debug_parameter_freeze_status(base_model)

    # =============== build optimizer ===============
    # 創建一個只包含可訓練參數的模型包裝器，以避免 filter_params 中的 names_no_grad 斷言錯誤
    class TrainableParametersWrapper:
        """只包含可訓練參數的模型包裝器"""
        def __init__(self, model, prefixes_to_ignore: Optional[List[str]] = ['car_', 'control_']):
            self.model = model
            self.prefixes_to_ignore = prefixes_to_ignore if prefixes_to_ignore is not None else []
            # self._trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
            self._trainable_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if any(pfx in  name for pfx in self.prefixes_to_ignore):
                    self._trainable_params.append((name, param))
            
        def named_parameters(self):
            return self._trainable_params
        
        def __getattr__(self, name):
            # 轉發其他屬性到原始模型
            return getattr(self.model, name)
    
    # 使用包裝器來只處理可訓練的參數
    trainable_model_wrapper = TrainableParametersWrapper(gpt_ddp if args.zero else gpt_wo_ddp,
                                                         prefixes_to_ignore=['car_', 'control_'])
    
    nowd_keys = set()
    _temp_ = args.nowd 
    if args.nowd >= 1:
        # 只包含實際存在於可訓練參數中的 no weight decay 參數
        nowd_keys |= {
            # CAR 相關的參數
            'car_blocks', 'car_control_proj', 'car_fusion_linears', 'car_fusion_gates',
             'car_skip_norm', 'car_skip_linear',
            # 一些通用的參數（如果它們在 CAR 模塊中）
            'gamma', 'beta', 'bias',
        }
    if args.nowd >= 2:
        nowd_keys |= {'class_emb', 'embedding'}
    
    # 使用包裝器來過濾參數，這樣 filter_params 只會看到可訓練的參數
    names, paras, para_groups = filter_params(trainable_model_wrapper, ndim_dict, nowd_keys=nowd_keys)
    del ndim_dict
    if '_' in args.ada:
        beta0, beta1 = map(float, args.ada.split('_'))
    else:
        beta0, beta1 = float(args.ada), -1
    
    opt_clz = {
        'sgd':   partial(torch.optim.SGD, momentum=beta0, nesterov=True),
        'adam':  partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
        'adamw': partial(torch.optim.AdamW, betas=(beta0, beta1), fused=args.afuse),
    }[args.opt]
    opt_kw = dict(lr=args.tlr, weight_decay=0)
    if args.oeps: opt_kw['eps'] = args.oeps
    print(f'[vgpt] optim={opt_clz}, opt_kw={opt_kw}\n')
    for group in para_groups:
        group['lr_sc'] = group.get('lr_sc', 1.0) * args.car_lr_scale

    gpt_optim = AmpOptimizer('gpt',
                              args.fp16,
                              opt_clz(params=para_groups, **opt_kw), 
                              gpt_ddp if args.zero else gpt_wo_ddp, 
                              args.r_accu, 
                              args.tclip, 
                              args.zero)
    del names, paras, para_groups
    
    if args.online_t5:
        print(f'Loading T5 from {args.t5_path}...')
        text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(args.t5_path, revision=None, legacy=True)
        text_tokenizer.model_max_length = args.tlen
        text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(args.t5_path, torch_dtype=torch.float16)
        text_encoder.to(args.device)
        text_encoder.eval()
        text_encoder.requires_grad_(False)
        [p.requires_grad_(False) for p in text_encoder.parameters()]
    else:
        text_tokenizer = text_encoder = None
    
    return text_tokenizer, text_encoder, vae_local, gpt_uncompiled, gpt_wo_ddp, gpt_ddp, gpt_wo_ddp_ema, gpt_ddp_ema, gpt_optim


def build_dataloaders(args):
    if args.task_type == 't2i':
        dataset_train = build_t2i_dataset(
            args,
            args.data_path,
            args.data_load_reso,
            max_caption_len=args.tlen,
            short_prob=args.short_cap_prob,
            load_vae_instead_of_image=False
        )
    else:
        raise NotImplementedError(f'args.task_type={args.task_type} not supported')
    type_train_set = type(dataset_train).__name__
    vbs = round(args.batch_size * 1.5)
    print(f"{args.batch_size=}, {vbs=}", flush=True)
    ld_val = math.ceil(50000 / vbs)
    ld_train = DataLoader(dataset=dataset_train, num_workers=args.workers, pin_memory=True, generator=args.get_different_generator_for_each_rank(), batch_size=None, prefetch_factor=args.prefetch_factor)
    iters_train = len(ld_train)
    print(f'len(dataloader): {len(ld_train)}, len(dataset): {len(dataset_train)}, total_samples: {dataset_train.total_samples()}')
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, iters_train={iters_train}, type(train_set)={type_train_set}')
    return iters_train, ld_train, ld_val


def main_train(args: arg_util.Args):
    saver = CKPTSaver(dist.is_master(), eval_milestone=None)
    ret = build_everything_from_args(args, saver)
    debug = args.debug_bsc
    
    if ret is None:
        return
    
    (
        text_tokenizer, text_encoder, trainer,
        start_ep, start_it, acc_str, eval_milestone,
        iters_train, ld_train, ld_val
    ) = ret
    gc.collect(), torch.cuda.empty_cache()
    
    # import heavy packages after Dataloader object creation
    from trainer_pilot import InfinityPilotTrainer
    ret: Tuple[
        misc.TensorboardLogger, T5TokenizerFast, T5EncoderModel, InfinityPilotTrainer,
        int, int, str, List[Tuple[float, float]], Optional[int], Optional[DataLoader], DataLoader,
    ]

    world_size = int(os.environ["WORLD_SIZE"])
    start_time, min_L_mean, min_L_tail, max_acc_mean, max_acc_tail = time.time(), 999., 999., -1., -1.
    last_val_loss_mean, best_val_loss_mean, last_val_acc_mean, best_val_acc_mean = 999., 999., 0., 0.
    last_val_loss_tail, best_val_loss_tail, last_val_acc_tail, best_val_acc_tail = 999., 999., 0., 0.
    seg5 = np.linspace(1, args.ep, 5+1, dtype=int).tolist()
    logging_params_milestone: List[int] = np.linspace(1, args.ep, 10+1, dtype=int).tolist()
    milestone_ep_feishu_log = set(seg5[:])
    vis_milestone_ep = set(seg5[:]) | set(x for x in (2, 4, 8, 16) if x <= args.ep)
    for x in [6, 12, 3, 24, 18, 48, 72, 96]:
        if len(vis_milestone_ep) < 10 and x <= args.ep:
            vis_milestone_ep.add(x)
    
    PARA_EMB, PARA_ALN, PARA_OT = 0, 0, 0
    for n, p in trainer.gpt_wo_ddp.named_parameters():
        if not p.requires_grad: continue
        if any(k in n for k in ('class_emb', 'pos_1LC', 'lvl_embed')):
            PARA_EMB += p.numel()
        elif any(k in n for k in ('ada_lin',)):
            PARA_ALN += p.numel()
        else:
            PARA_OT += p.numel()
    PARA_ALL = PARA_EMB + PARA_ALN + PARA_OT
    
    trainer.gpt_opt.log_param(ep=-1)
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    ep_lg = max(1, args.ep // 10) if args.ep <= 100 else max(1, args.ep // 20)
    
    # ============================================= epoch loop begins =============================================
    L_mean, L_tail = -1, -1
    epochs_loss_nan = 0
    # build wandb logger
    if dist.is_master():
        init_wandb(args, 
              exp_name=args.exp_name, 
              project_name=args.project_name,
              )
    
        # 記錄模型參數數量 - 移除 step=0，讓 wandb 自動處理
        log_metrics({
            'params/emb_params': PARA_EMB,
            'params/align_params': PARA_ALN,
            'params/other_params': PARA_OT,
            'params/total_params': PARA_ALL
        })
    for ep in range(start_ep, args.ep):
        if ep % ep_lg == 0 or ep == start_ep:
            print(f'[PT info]  from ep{start_ep} it{start_it}, acc_str: {acc_str}, diffs: {args.diffs},    =======>  bed: {args.bed}  <=======\n')
        
        # [save checkpoint]
        save_car_epoch_freq = getattr(args, 'save_car_epoch_freq', 1)  # 預設值為 1
        if dist.is_master() and ep != 0 and ep % save_car_epoch_freq == 0:
            with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=3, verbose=True):
            # 使用我們的 save_checkpoint_pilot 函數
                save_checkpoint_pilot(
                    saver=saver, 
                    args=args, 
                    trainer=trainer, 
                    epoch=ep, 
                    iteration=ep * iters_train,  # 或者使用當前的迭代數
                    acc_str=f'Lm:{min_L_mean:.3f}_Lt:{min_L_tail:.3f}_Am:{max_acc_mean:.2f}_At:{max_acc_tail:.2f}'
                )

        # set epoch for dataloader
        if args.use_streaming_dataset:
            ld_train.dataset.set_epoch(ep)

        # [train one epoch]
        stats, (sec, remain_time, finish_time) = train_one_ep(
            ep=ep,
            is_first_ep=ep == start_ep,
            start_it=start_it if ep == start_ep else 0,
            me=None,
            saver=saver,
            args=args,
            ld_or_itrt=iter(ld_train),
            iters_train=iters_train,
            text_tokenizer=text_tokenizer, text_encoder=text_encoder,
            trainer=trainer,
            logging_params_milestone=logging_params_milestone,
            enable_timeline_sdk=enable_timeline_sdk,
        )
        
        # [update the best loss or acc]
        L_mean, L_tail, acc_mean, acc_tail, grad_norm = stats['Lm'], stats['Lt'], stats['Accm'], stats['Acct'], stats['tnm']
        min_L_mean, max_acc_mean, max_acc_tail = min(min_L_mean, L_mean), max(max_acc_mean, acc_mean), max(max_acc_tail, acc_tail)
        if L_tail != -1:
            min_L_tail = min(min_L_tail, L_tail)
        
        # [check nan]
        epochs_loss_nan += int(not math.isfinite(L_mean))
        if (args.fp16 == 1 and epochs_loss_nan >= 2) or (args.fp16 != 1 and epochs_loss_nan >= 1):
            print(f'[rk{dist.get_rank():02d}] L_mean is {L_mean}, stopping training!', flush=True, force=True)
            sys.exit(666)
        
        # [logging]
        args.cur_phase = 'AR'
        args.cur_ep = f'{ep+1}/{args.ep}'
        args.remain_time, args.finish_time = remain_time, finish_time
        args.last_Lnll, args.last_Ld, args.acc_all, args.acc_real, args.acc_fake, args.last_wei_g = min_L_mean, min_L_tail, None, (None if max_acc_mean < 0 else max_acc_mean), (None if max_acc_tail < 0 else max_acc_tail), grad_norm
        if math.isfinite(args.last_wei_g) and args.last_wei_g > 4:
            args.grad_boom = 'boom'
        
        AR_ep_loss = {}
        is_val_and_also_saving = (ep + 1) % max(1, args.ep // 25) == 0
        if (ep + 1) < 10:
            law_stats = {
                'last_Lm': L_mean, 'best_Lm': min_L_mean, 'last_Am': acc_mean, 'best_Am': max_acc_mean,
                'last_Lt': L_tail, 'best_Lt': min_L_tail, 'last_At': acc_tail, 'best_At': max_acc_tail,
                'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
            }
        elif is_val_and_also_saving:
            if ld_val is None or isinstance(ld_val, int):    # args.nodata or args.nova
                last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = 0.666, 0.555, 5.55, 6.66, 50000, 0.001
            else:
                print("[DEBUG] Starting evaluation...")
                last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail, tot, cost = trainer.eval_ep(ep, args, ld_val)

            best_val_loss_mean, best_val_loss_tail = min(best_val_loss_mean, last_val_loss_mean), min(best_val_loss_tail, last_val_loss_tail)
            best_val_acc_mean, best_val_acc_tail = max(best_val_acc_mean, last_val_acc_mean), max(best_val_acc_tail, last_val_acc_tail)
            AR_ep_loss['vL_mean'], AR_ep_loss['vL_tail'], AR_ep_loss['vacc_mean'], AR_ep_loss['vacc_tail'] = last_val_loss_mean, last_val_loss_tail, last_val_acc_mean, last_val_acc_tail
            print(f'  [*] [ep{ep}]  VAL {tot}  |  Lm: {L_mean:.4f}, Lt: {L_tail:.4f}, Accm: {acc_mean:.2f}, Acct: {acc_tail:.2f}, cost: {cost:.2f}s')
            law_stats = {
                'last_Lm': last_val_loss_mean, 'best_Lm': best_val_loss_mean, 'last_Am': last_val_acc_mean, 'best_Am': best_val_acc_mean,
                'last_Lt': last_val_loss_tail, 'best_Lt': best_val_loss_tail, 'last_At': last_val_acc_tail, 'best_At': best_val_acc_tail,
                'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
            }
        else: law_stats = None
        if dist.is_master() and law_stats is not None:
            stat_file = os.path.join(args.bed, 'law.stat')
            if os.path.exists(stat_file):
                try:
                    with open(stat_file, 'r', encoding='utf-8') as law_fp:
                        raw = law_fp.read().strip()
                        tag_to_epv = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    print(f'[warning] law.stat is not valid JSON, reinitializing: {stat_file}')
                    tag_to_epv = {}
            else:
                tag_to_epv = {}
            if not tag_to_epv:
                tag_to_epv = {tag: {} for tag in law_stats.keys()}
            else:
                for tag in law_stats.keys():
                    tag_to_epv.setdefault(tag, {})
            for tag, v in law_stats.items():
                tag_to_epv[tag][ep + 1] = v
            with open(stat_file, 'w', encoding='utf-8') as law_fp: json.dump(tag_to_epv, law_fp, indent=2)
            
            # ============= LEGACY =============
            with open(os.path.join(args.bed, 'law'), 'w') as law_fp:
                json.dump({
                    'last_Lm': last_val_loss_mean, 'best_Lm': best_val_loss_mean, 'last_Am': last_val_acc_mean, 'best_Am': best_val_acc_mean,
                    'last_Lt': last_val_loss_tail, 'best_Lt': best_val_loss_tail, 'last_At': last_val_acc_tail, 'best_At': best_val_acc_tail,
                    'pe': PARA_EMB, 'paln': PARA_ALN, 'pot': PARA_OT, 'pall': PARA_ALL,
                }, law_fp, indent=2)
        print(f'  [*] [ep{ep}]  Lmean: {min_L_mean:.3f} ({L_mean:.3f}), Ltail {min_L_tail:.3f} ({L_tail:.3f}),  Acc m-t: {max_acc_mean:.2f} {max_acc_tail:.2f},  Remain: {remain_time},  Finish: {finish_time}', flush=True)
        AR_ep_loss['L_mean'], AR_ep_loss['L_tail'], AR_ep_loss['acc_mean'], AR_ep_loss['acc_tail'] = L_mean, L_tail, acc_mean, acc_tail        
        args.dump_log()
    # ============================================= epoch loop ends =============================================
    
    total_time = f'{(time.time() - start_time) / 60 / 60:.1f}h'
    print('\n\n')
    print(f'  [*] [PT finished]  Total Time: {total_time},   Lm: {min_L_mean:.3f} ({L_mean}),   Lt: {min_L_tail:.3f} ({L_tail})')
    print('\n\n')
    
    del stats, iters_train, ld_train
    time.sleep(3), gc.collect(), torch.cuda.empty_cache(), time.sleep(3)
    return


g_speed_ls = deque(maxlen=128)
def train_one_ep(
    ep: int, is_first_ep: bool, start_it: int, me: misc.MetricLogger,
    saver: CKPTSaver, args: arg_util.Args, ld_or_itrt, iters_train: int, 
    text_tokenizer: T5TokenizerFast, text_encoder: T5EncoderModel, trainer, logging_params_milestone, enable_timeline_sdk: bool,
):
    # IMPORTANT: import heavy packages after the Dataloader object creation/iteration to avoid OOM
    from trainer_pilot import InfinityPilotTrainer
    from infinity.utils.lr_control import lr_wd_annealing
    trainer: InfinityPilotTrainer
    
    step_cnt = 0
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=20, verbose=True) as telling_dont_kill:
        last_touch = time.time()
        g_it, max_it = ep * iters_train, args.ep * iters_train
        
        doing_profiling = args.prof and ep == 0 and (args.profall or dist.is_master())
        maybe_record_function = record_function if doing_profiling else nullcontext
        trainer.gpt_wo_ddp.maybe_record_function = maybe_record_function
        
        last_t_perf = time.time()
        speed_ls: deque = g_speed_ls
        FREQ = min(args.prof_freq, iters_train//2-1)
        NVIDIA_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8))
        ranges = set([2 ** i for i in range(20)])
        if ep <= 1: ranges |= {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40}
        PRINTABLE_IT_PLUS_1 = set(FREQ*i for i in ranges)

        me = misc.MetricLogger()
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['tlr']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['tnm']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.3f} ({global_avg:.3f})')) for x in ['Lm', 'Lt']]
        [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['Accm', 'Acct']]
        # ============================================= iteration loop begins =============================================
        for it, data in me.log_every(start_it, iters_train, ld_or_itrt, args.log_freq, args.log_every_iter, header):
            g_it = ep * iters_train + it

            # adding early stopping condition for debugging
            # if it > start_it + 10:
            #     print(f"[DEBUG] Early stopping condition met at iteration {it}, stopping training.")
            #     break

            # calling inc_step to sync the global_step
            if enable_timeline_sdk:
                ndtimeline.inc_step()

            if (it+1) % FREQ == 0:
                speed_ls.append((time.time() - last_t_perf) / FREQ)
                last_t_perf = time.time()

                if enable_timeline_sdk:
                    ndtimeline.flush()
            
            # if (g_it+1) % args.save_model_iters_freq == 0:
            #     with misc.Low_GPU_usage(files=[args.log_txt_path], sleep_secs=3, verbose=True):
            #         saver.sav(args=args, g_it=(g_it+1), next_ep=ep, next_it=it+1, trainer=trainer, acc_str=f'[todo]', eval_milestone=None, also_save_to=None, best_save_to=None)
            
            with maybe_record_function('before_train'):
                # [get data] - Handle pilot data format with condition
                condition_inputs = {}
                if len(data) == 4:
                    inp, condition_mask, condition_normal, captions = data
                elif len(data) == 3:
                    inp, condition_normal, captions = data
                    condition_mask = None
                else:
                    inp, captions = data
                    condition_mask = None
                    condition_normal = None
                tokens = text_tokenizer(text=captions, max_length=text_tokenizer.model_max_length, padding='max_length', truncation=True, return_tensors='pt')  # todo: put this into dataset
                input_ids = tokens.input_ids.cuda(non_blocking=True)
                mask = tokens.attention_mask.cuda(non_blocking=True)
                text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
                
                lens: List[int] = mask.sum(dim=-1).tolist()
                cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
                Ltext = max(lens)
                
                kv_compact = []
                for len_i, feat_i in zip(lens, text_features.unbind(0)):
                    kv_compact.append(feat_i[:len_i])
                kv_compact = torch.cat(kv_compact, dim=0)
                text_cond_tuple: Tuple[torch.FloatTensor, List[int], torch.LongTensor, int] = (kv_compact, lens, cu_seqlens_k, Ltext)
                inp = inp.to(args.device, non_blocking=True)
                if condition_normal is not None:
                    condition_normal = condition_normal.to(args.device, non_blocking=True)
                    condition_inputs['normal'] = condition_normal
            if condition_mask is not None:
                condition_mask = condition_mask.to(args.device, non_blocking=True)
                if args.car_mask_drop_prob > 0 and random.random() < args.car_mask_drop_prob:
                    condition_mask = None
                if condition_mask is not None:
                    condition_inputs['mask'] = condition_mask
                if not condition_inputs:
                    condition_inputs = None
                # if it > start_it + 10:
                #     telling_dont_kill.early_stop()
                
                # [logging]
                args.cur_it = f'{it+1}/{iters_train}'
                args.last_wei_g = me.meters['tnm'].median
                if dist.is_local_master() and (it >= start_it + 10) and (time.time() - last_touch > 90):
                    _, args.remain_time, args.finish_time = me.iter_time.time_preds(max_it - g_it + (args.ep - ep) * 15)      # +15: other cost
                    args.dump_log()
                    last_touch = time.time()

                # [schedule learning rate]
                wp_it = max(int(args.wp * iters_train), args.min_warmup_iters)
                min_tlr, max_tlr, min_twd, max_twd = lr_wd_annealing(args.sche, trainer.gpt_opt.optimizer, args.tlr, args.twd, args.twde, g_it, wp_it, max_it, wp0=args.wp0, wpe=args.wpe)
                
                # 檢查學習率是否異常
                if not math.isfinite(max_tlr) or max_tlr <= 0 or max_tlr > 1:
                    print(f"[ERROR] Abnormal learning rate: {max_tlr}")
                    raise RuntimeError(f"Abnormal learning rate: {max_tlr}")
                
                if max_tlr > 1e-2:  # 學習率過大警告
                    print(f"[WARNING] Large learning rate detected: {max_tlr}")

                # 只在記錄頻率時記錄到 wandb，避免步驟衝突
                if dist.is_master() and (it % args.log_freq == 0 or it == 0):
                    log_metrics({
                        'train/loss_mean': me.meters['Lm'].median,
                        'train/loss_tail': me.meters['Lt'].median,
                        'train/acc_mean': me.meters['Accm'].median,
                        'train/acc_tail': me.meters['Acct'].median,
                        'train/learning_rate': max_tlr,
                        'train/grad_norm': me.meters['tnm'].median,
                        'train/epoch': ep,
                        'train/iter': g_it,
                    }, step=g_it)
                
                # [get scheduled hyperparameters]
                progress = g_it / (max_it - 1)
                clip_decay_ratio = (0.3 ** (20 * progress) + 0.2) if args.cdec else 1
                
                stepping = (g_it + 1) % args.ac == 0
                step_cnt += int(stepping)
            
            with maybe_record_function('in_training'):
                grad_norm_t, scale_log2_t = trainer.train_step(
                    ep=ep, it=it, g_it=g_it, stepping=stepping, clip_decay_ratio=clip_decay_ratio,
                    metric_lg=me, 
                    logging_params=stepping and step_cnt == 1 and (ep < 4 or ep in logging_params_milestone), 
                    inp_B3HW=inp, 
                    condition_inputs=condition_inputs,
                    text_cond_tuple=text_cond_tuple,
                    args=args,
                )
            
            with maybe_record_function('after_train'):
                me.update(tlr=max_tlr)

    # ============================================= iteration loop ends =============================================
    
    me.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in me.meters.items()}, me.iter_time.time_preds(max_it - (g_it + 1) + (args.ep - ep) * 15)  # +15: other cost


wait1 = os.path.join(os.path.expanduser('~'), 'wait1')
def main():     # # 'pt_le_ft' in train_vae.py is the same as 'pt_le_ft' in train_gpt.py
    # Set memory optimization environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations for better memory management
    
    # Enable DDP debugging for unused parameters
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    
    # Clear any existing CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    if dist.is_local_master(): misc.os_system(f'touch {wait1}')
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    try:
        main_train(args)
    finally:
        if dist.is_master():
            finish()
    
    args.remain_time, args.finish_time = '-', time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() - 60))
    args.cur_phase = 'OK'
    print(f'final args:\n\n{str(args)}')
    args.dump_log()
    if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
        sys.stdout.close(), sys.stderr.close()
    if dist.is_local_master(): misc.os_system(f'rm -rf {wait1}')
    # if args.vis and dist.is_visualizer():
    #     misc.os_system(f'hdfs dfs -get {args.tb_log_dir_online}/* {args.tb_log_dir}/ >/dev/null 2>&1')  # 'cp -r {args.local_out_path}/* {args.bed}/' is done by lockable.py or launch.py
    dist.barrier()
    time.sleep(120)


if __name__ == '__main__':
    try:
        main()
    except Exception as _e:
        time.sleep(dist.get_rank() * 1 + random.random() * 0.5)
        try:
            # noinspection PyArgumentList
            print(f'[rk{dist.get_rank():2d}] {type(_e).__name__}', flush=True, force=True)
        except:
            try: print(f'[rk{dist.get_rank():2d}] {type(_e).__name__}', flush=True)
            except: pass
        if dist.is_master():
            print(f'[err]:\n{_e}')
            traceback.print_exc()
        raise _e
    finally:
        misc.os_system(f'rm -rf {wait1}')
        dist.finalize()
        if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
            sys.stdout.close(), sys.stderr.close()
