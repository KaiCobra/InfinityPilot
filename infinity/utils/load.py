#!/usr/bin/python3
import gc
import os
import os.path as osp
import random
import sys
from copy import deepcopy
from typing import Tuple, Union

import colorama
import torch
import yaml

import infinity.utils.dist as dist

from infinity.models import Infinity
from infinity.models.infinity_pilot import InfinityPilot
from infinity.models.ema import get_ema_model
from infinity.utils import arg_util, misc
from infinity.utils.misc import os_system


def build_vae_gpt(args: arg_util.Args, vae_st: dict, skip_gpt: bool, force_flash=False, device='cuda'):
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
        vae_local = vae_model(vae_st, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                              encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(args.device)
        if args.fake_vae_input:
            vae_local.encoder = None
            vae_local.decoder = None
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"vae_type {args.vae_type} not supported")
    if force_flash: args.flash = True
    gpt_kw = dict(
        pretrained=False, global_pool='',
        text_channels=args.Ct5, text_maxlen=args.tlen,
        norm_eps=args.norm_eps, rms_norm=args.rms,
        shared_aln=args.saln, head_aln=args.haln,
        cond_drop_rate=args.cfg, rand_uncond=args.rand_uncond, drop_rate=args.drop,
        cross_attn_layer_scale=args.ca_gamma, nm0=args.nm0, tau=args.tau, cos_attn=args.cos, swiglu=args.swi,
        raw_scale_schedule=args.scale_schedule,
        head_depth=args.dec,
        top_p=args.tp, top_k=args.tk,
        customized_flash_attn=args.flash, fused_mlp=args.fuse, fused_norm=args.fused_norm,
        checkpointing=args.enable_checkpointing,
        pad_to_multiplier=args.pad_to_multiplier,
        use_flex_attn=args.use_flex_attn,
        batch_size=args.batch_size,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        use_bit_label=args.use_bit_label,
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn,
        train_h_div_w_list=args.train_h_div_w_list,
        always_training_scales=args.always_training_scales,
        apply_spatial_patchify=args.apply_spatial_patchify,
    )
    if args.dp >= 0: gpt_kw['drop_path_rate'] = args.dp
    if args.hd > 0: gpt_kw['num_heads'] = args.hd
    
    print(f'[create gpt_wo_ddp] constructor kw={gpt_kw}\n')
    gpt_kw['vae_local'] = vae_local
    
    model_str = args.model.replace('vgpt', 'infinity')   # legacy
    print(f"{model_str=}")
    if model_str.rsplit('c', maxsplit=1)[-1].isdecimal():
        model_str, block_chunks = model_str.rsplit('c', maxsplit=1)
        block_chunks = int(block_chunks)
    else:
        block_chunks = 1
    gpt_kw['block_chunks'] = block_chunks
    
    from infinity.models import Infinity
    from timm.models import create_model
    gpt_wo_ddp: Infinity = create_model(model_str, **gpt_kw)
    if args.use_fsdp_model_ema:
        gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
    else:
        gpt_wo_ddp_ema = None
    gpt_wo_ddp = gpt_wo_ddp.to(device)

    assert all(not p.requires_grad for p in vae_local.parameters())
    assert all(p.requires_grad for n, p in gpt_wo_ddp.named_parameters())
    
    return vae_local, gpt_wo_ddp, gpt_wo_ddp_ema


def build_vae_gpt_pilot(args: arg_util.Args, vae_st: dict, skip_gpt: bool, force_flash=False, device='cuda'):
    """Build VAE and InfinityPilot model for pilot training"""
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
        vae_local = vae_model(vae_st, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                              encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(args.device)
        if args.fake_vae_input:
            vae_local.encoder = None
            vae_local.decoder = None
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"vae_type {args.vae_type} not supported")
    
    if force_flash: args.flash = True
    gpt_kw = dict(
        pretrained=False, global_pool='',
        text_channels=args.Ct5, text_maxlen=args.tlen,
        norm_eps=args.norm_eps, rms_norm=args.rms,
        shared_aln=args.saln, head_aln=args.haln,
        cond_drop_rate=args.cfg, rand_uncond=args.rand_uncond, drop_rate=args.drop,
        cross_attn_layer_scale=args.ca_gamma, nm0=args.nm0, tau=args.tau, cos_attn=args.cos, swiglu=args.swi,
        raw_scale_schedule=args.scale_schedule,
        head_depth=args.dec,
        top_p=args.tp, top_k=args.tk,
        customized_flash_attn=args.flash, fused_mlp=args.fuse, fused_norm=args.fused_norm,
        checkpointing=args.enable_checkpointing,
        pad_to_multiplier=args.pad_to_multiplier,
        use_flex_attn=args.use_flex_attn,
        batch_size=args.batch_size,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        use_bit_label=args.use_bit_label,
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn,
        train_h_div_w_list=args.train_h_div_w_list,
        always_training_scales=args.always_training_scales,
        apply_spatial_patchify=args.apply_spatial_patchify,
    )
    if skip_gpt:
        return vae_local, None, None
        
    # For pilot training, we need to use the correct model dimensions
    model_str = getattr(args, 'gpt', args.model).replace('-', '_')
    # Keep the original model specification for correct dimensions
    # Don't replace 2b/8b/20b as they specify the correct embed_dim
    
    if getattr(args, 'num_block_chunks', 1) > 1:
        block_chunks = args.num_block_chunks
    else:
        block_chunks = 1
    gpt_kw['block_chunks'] = block_chunks
    
    # Import Infinity instead of InfinityPilot for base model
    from infinity.models.infinity import Infinity
    from timm.models import create_model
    
    # Filter out timm-specific arguments and device argument that Infinity doesn't accept
    TIMM_KEYS = {'pretrained', 'global_pool', 'img_size', 'pretrained_cfg', 'pretrained_cfg_overlay', 'device'}
    filtered_gpt_kw = {k: v for k, v in gpt_kw.items() if k not in TIMM_KEYS}
    
    # Use base Infinity model (without CAR modules)
    model_str = getattr(args, 'gpt', args.model).replace('-', '_')
    
    # 預載入 checkpoint 來檢測架構
    infinity_checkpoint = None
    if hasattr(args, 'rush_resume') and args.rush_resume and os.path.exists(args.rush_resume):
        print(f"Pre-loading checkpoint to detect architecture: {args.rush_resume}")
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
    
    # 根據 checkpoint 調整架構參數
    if infinity_checkpoint:
        has_ada_gss = any('ada_gss' in k for k in infinity_checkpoint.keys())
        has_shared_ada_lin = any('shared_ada_lin' in k for k in infinity_checkpoint.keys())
        has_zero_bias = any('zero_k_bias' in k or 'zero_v_bias' in k for k in infinity_checkpoint.keys())
        
        print(f"Checkpoint architecture detection:")
        print(f"  - ada_gss: {has_ada_gss}")
        print(f"  - shared_ada_lin: {has_shared_ada_lin}")
        print(f"  - zero_bias: {has_zero_bias}")
        
        if has_ada_gss or has_shared_ada_lin:
            filtered_gpt_kw['shared_aln'] = True
            print("Setting shared_aln=True to match checkpoint")
        
    # Remove pilot-specific naming but keep dimension specifications
    filtered_gpt_kw['vae_local'] = vae_local
    
    # 直接使用 InfinityPilot，不通過 TIMM（避免註冊問題和雙重初始化）
    print("Creating InfinityPilot directly (bypassing TIMM registry)")
    gpt_wo_ddp: InfinityPilot = InfinityPilot(
        infinity_base_model=infinity_checkpoint,  # 傳入 checkpoint 用於架構匹配
        **filtered_gpt_kw
    )
    
    if args.use_fsdp_model_ema:
        gpt_wo_ddp_ema = get_ema_model(gpt_wo_ddp)
    else:
        gpt_wo_ddp_ema = None
    gpt_wo_ddp = gpt_wo_ddp.to(device)

    assert all(not p.requires_grad for p in vae_local.parameters())
    # assert all(p.requires_grad for n, p in gpt_wo_ddp.named_parameters())
    
    return vae_local, gpt_wo_ddp, gpt_wo_ddp_ema


if __name__ == '__main__':
    pass
