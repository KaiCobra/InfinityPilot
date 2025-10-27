# import everything needed for building model and optimizer
import os
import torch
import math
from functools import partial

import infinity.utils.dist as dist
from infinity.utils import arg_util, misc
from infinity.models.infinity_pilot import InfinityPilot
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from debug_utils.frozen_state import debug_parameter_freeze_status, debug_parameter_number


def build_model_optimizer(args, vae_ckpt):
    """[misc] can be deleted"""
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
    if not hasattr(args, 'scale_schedule') or args.scale_schedule is None:
        args.scale_schedule = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)  # Default from infinity.py
        print(f"Setting default scale_schedule: {args.scale_schedule}")
    
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
    
    # Create InfinityPilot with memory optimization
    # Step 1: Create model without loading weights first
    gpt_wo_ddp = InfinityPilot(
        infinity_base_model=None,  # Don't load weights yet
        init_car_modules=True,    # Don't init CAR yet
        freeze_infinity=True,     # Freeze infinity
        **gpt_kw
    )
    
    # Step 2: Load infinity weights if available (memory efficient)
    if infinity_checkpoint is not None:
        print("[Memory Optimization] Loading Infinity weights...")
        gpt_wo_ddp.load_infinity_weights(infinity_checkpoint)
        # Clear checkpoint from memory immediately after loading
        del infinity_checkpoint
        torch.cuda.empty_cache()
    
    # Step 3: Initialize CAR modules and freeze infinity
    gpt_wo_ddp._init_car_modules()
    gpt_wo_ddp.freeze_infinity_parameters()
    # Debug console: {(name, p.grad.shape) for name, p in gpt_wo_ddp.named_parameters() if p.grad is not None}

    # verify freeze state
    debug_parameter_freeze_status(gpt_wo_ddp)
    
    # Memory monitoring after model creation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory after model creation: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    if args.tini < 0:
        args.tini = math.sqrt(1 / gpt_wo_ddp.C / 3)
    
    # Only initialize CAR modules, Infinity part is already loaded
    car_params = gpt_wo_ddp.get_car_parameters()
    for param in car_params:
        if param.ndim >= 2:  
            # for matrices in Linear and Conv layers
            torch.nn.init.xavier_uniform_(param, gain=args.tini)
        else:  
            # for biases and LayerNorm/GroupNorm weights
            torch.nn.init.zeros_(param)
    
    print("InfinityPilot initialized with automatic checkpoint loading and CAR modules trainable")
    
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
        def __init__(self, model):
            self.model = model
            self._trainable_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
            
        def named_parameters(self):
            return self._trainable_params
        
        def __getattr__(self, name):
            # 轉發其他屬性到原始模型
            return getattr(self.model, name)
    
    # 使用包裝器來只處理可訓練的參數
    trainable_model_wrapper = TrainableParametersWrapper(gpt_ddp if args.zero else gpt_wo_ddp)
    
    nowd_keys = set()
    _temp_ = args.nowd 
    if args.nowd >= 1:
        # 只包含實際存在於可訓練參數中的 no weight decay 參數
        nowd_keys |= {
            # CAR 相關的參數
            'car_control_convs', 'car_var_conv', 'car_skip_norm', 'car_skip_linear',
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
    gpt_optim = AmpOptimizer('gpt', args.fp16, opt_clz(params=para_groups, **opt_kw), gpt_ddp if args.zero else gpt_wo_ddp, args.r_accu, args.tclip, args.zero)
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


def build_model_optimizer_pilot(args, vae_ckpt):
    """
        [TODO]
        [V] load pretrained weights for Infinity module.
        [V] Drop out infinity_wo_ddp_ema if exists.
        [V] Drop out DDP/FSDP wrapper if exists.
        [ ] and then freeze infinity params.
        [ ] create CAR modules and initialize them.
        [ ] create optimizer only for CAR modules.
        [ ] verify freeze state.
        [ ] verify optimizer params.
    """
    from infinity.models.infinity import Infinity, MultipleLayers
    from infinity.models.init_param import init_weights
    from infinity.utils.amp_opt import AmpOptimizer
    from infinity.utils.lr_control import filter_params
    from infinity.utils.load import build_vae_gpt, build_vae_gpt_pilot
    
    # disable builtin initialization for speed
    setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)
    setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)
    vae_local, infinity_wo_ddp, _ = build_vae_gpt_pilot(args, vae_ckpt, skip_gpt=False, device=args.model_init_device)
    del vae_ckpt
    if args.tini < 0:
        args.tini = math.sqrt(1 / infinity_wo_ddp.C / 3)
    init_weights(infinity_wo_ddp, other_std=args.tini)
    # infinity_wo_ddp.special_init(aln_init=args.aln, aln_gamma_init=args.alng, scale_head=args.hd0, scale_proj=args.diva)

    # init weights --> load weights
    if args.rush_resume is not None:
        print(f"{args.rush_resume=}")
        cpu_d = torch.load(args.rush_resume, 'cpu', weights_only=False)
        if 'trainer' in cpu_d: #false
            state_dict = cpu_d['trainer']['gpt_fsdp']
            # ema_state_dict = cpu_d['trainer'].get('gpt_ema_fsdp', state_dict)
        elif 'gpt_fsdp' in cpu_d:
            state_dict = cpu_d['gpt_fsdp']
            # ema_state_dict = cpu_d.get('gpt_ema_fsdp', state_dict)
        else:
            state_dict = cpu_d
            # ema_state_dict = state_dict
        def drop_unfit_weights(state_dict):
            if 'word_embed.weight' in state_dict and (state_dict['word_embed.weight'].shape[1] != infinity_wo_ddp.word_embed.in_features):
                del state_dict['word_embed.weight']
            if 'head.weight' in state_dict and (state_dict['head.weight'].shape[0] != infinity_wo_ddp.head.out_features):
                del state_dict['head.weight']
            if 'head.bias' in state_dict and (state_dict['head.bias'].shape[0] != infinity_wo_ddp.head.bias.shape[0]):
                del state_dict['head.bias']
            if state_dict['text_proj_for_sos.ca.mat_kv.weight'].shape != infinity_wo_ddp.text_proj_for_sos.ca.mat_kv.weight.shape:
                del state_dict['cfg_uncond']
                for key in list(state_dict.keys()):
                    if 'text' in key:
                        del state_dict[key]
            return state_dict
        
        infinity_wo_ddp.load_state_dict(drop_unfit_weights(state_dict), strict=False)
        # if args.use_fsdp_model_ema:
        #     infinity_wo_ddp_ema.load_state_dict(drop_unfit_weights(ema_state_dict), strict=False)
        # freeze all infinity params
        for n, p in infinity_wo_ddp.named_parameters():
            p.requires_grad = False
        ndim_dict = {name: para.ndim for name, para in infinity_wo_ddp.named_parameters() if para.requires_grad}

    else:
        raise NotImplementedError("Please provide weight for InfinityPilot to load pretrained weights. [--args.rush_resume]")
    
    print(f'[PT] GPT model = {infinity_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}'
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_local), ('VAE.quant', vae_local.quantize)
    )]))
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('GPT', infinity_wo_ddp),
    )]) + '\n\n')
    
    
    # torch.cuda.synchronize()
    
    # ============== build CAR module ===============
    #infinityPilot = InfinityPilot([params])



    # =============== build optimizer ===============
    nowd_keys = set()
    if args.nowd >= 1:
        nowd_keys |= {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
            'text_proj_for_sos.ca.mat_q',
        }
    if args.nowd >= 2:
        nowd_keys |= {'class_emb', 'embedding'}
    names, paras, para_groups = filter_params(infinity_wo_ddp, ndim_dict, nowd_keys=nowd_keys)
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
    gpt_optim = AmpOptimizer('gpt', args.fp16, opt_clz(params=para_groups, **opt_kw), infinity_wo_ddp, args.r_accu, args.tclip, args.zero)
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
    
    return text_tokenizer, text_encoder, vae_local, None, infinity_wo_ddp, None, None, None, gpt_optim