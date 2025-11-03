#!/usr/bin/env python3
"""
InfinityPilot 運行程序
基於 run_infinity.py 修改，專門用於運行 InfinityPilot 模型
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from typing import List
import math
import time
import hashlib
import yaml
import argparse
import shutil
import re

import cv2
import numpy as np
import torch
torch._dynamo.config.cache_size_limit=64
import pandas as pd
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast
from PIL import Image, ImageEnhance
import torch.nn.functional as F
from torch.cuda.amp import autocast

# 導入 InfinityPilot 相關模塊
from infinity.models.infinity_pilot import InfinityPilot
from infinity.models.basic import *
import PIL.Image as PImage
from torchvision.transforms.functional import to_tensor
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


def extract_key_val(text):
    """從文本中提取鍵值對"""
    pattern = r'<(.+?):(.+?)>'
    matches = re.findall(pattern, text)
    key_val = {}
    for match in matches:
        key_val[match[0]] = match[1].lstrip()
    return key_val


def encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt=False):
    """編碼文本提示"""
    if enable_positive_prompt:
        print(f'before positive_prompt aug: {prompt}')
        prompt = aug_with_positive_prompt(prompt)
        print(f'after positive_prompt aug: {prompt}')
    print(f'prompt={prompt}')
    captions = [prompt]
    tokens = text_tokenizer(text=captions, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
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
    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, Ltext)
    return text_cond_tuple


def aug_with_positive_prompt(prompt):
    """為提示添加正向增強"""
    for key in ['man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person', 'human', 'adult', 'teenager', 'employee', 
                'employer', 'worker', 'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather', 'son', 'daughter']:
        if key in prompt:
            prompt = prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
            break
    return prompt


def enhance_image(image):
    """增強圖像質量"""
    for t in range(1):
        contrast_image = image.copy()
        contrast_enhancer = ImageEnhance.Contrast(contrast_image)
        contrast_image = contrast_enhancer.enhance(1.05)  # 增強對比度
        color_image = contrast_image.copy()
        color_enhancer = ImageEnhance.Color(color_image)
        color_image = color_enhancer.enhance(1.05)  # 增強飽和度
    return color_image


def prepare_control_image(control_path, target_h, target_w):
    """準備控制圖像"""
    if control_path and osp.exists(control_path):
        control_img = Image.open(control_path).convert('RGB')
        control_tensor = transform(control_img, target_h, target_w)
        return control_tensor.unsqueeze(0)  # 添加 batch 維度
    return None


def gen_one_img_pilot(
    infinity_pilot, 
    vae, 
    text_tokenizer,
    text_encoder,
    prompt, 
    control_path=None,
    cfg_list=[],
    tau_list=[],
    negative_prompt='',
    scale_schedule=None,
    top_k=900,
    top_p=0.97,
    cfg_sc=3,
    cfg_exp_k=0.0,
    cfg_insertion_layer=-5,
    vae_type=0,
    gumbel=0,
    softmax_merge_topk=-1,
    gt_leak=-1,
    gt_ls_Bl=None,
    g_seed=None,
    sampling_per_bits=1,
    enable_positive_prompt=0,
):
    """使用 InfinityPilot 生成一張圖像"""
    sstt = time.time()
    if not isinstance(cfg_list, list):
        cfg_list = [cfg_list] * len(scale_schedule)
    if not isinstance(tau_list, list):
        tau_list = [tau_list] * len(scale_schedule)
    
    # 編碼文本提示
    text_cond_tuple = encode_prompt(text_tokenizer, text_encoder, prompt, enable_positive_prompt)
    
    # 處理負提示
    if negative_prompt:
        negative_label_B_or_BLT = encode_prompt(text_tokenizer, text_encoder, negative_prompt)
    else:
        negative_label_B_or_BLT = None
    
    # 準備控制圖像（如果提供）
    control_tokens = None
    if control_path:
        print(f"[warn] control_path provided ({control_path}) but automatic conversion to control tokens is not implemented in this script yet.")
        print("       Proceeding without control by default.")

    print(f'cfg: {cfg_list}, tau: {tau_list}')
    
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        stt = time.time()
        
        # 使用 InfinityPilot 的推理方法
        _, _, img_list = infinity_pilot.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple, 
            g_seed=g_seed,
            B=1, 
            negative_label_B_or_BLT=negative_label_B_or_BLT, 
            force_gt_Bhw=None,
            cfg_sc=cfg_sc, 
            cfg_list=cfg_list, 
            tau_list=tau_list, 
            top_k=top_k, 
            top_p=top_p,
            returns_vemb=1, 
            ratio_Bl1=None, 
            gumbel=gumbel, 
            norm_cfg=False,
            cfg_exp_k=cfg_exp_k, 
            cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type, 
            softmax_merge_topk=softmax_merge_topk,
            ret_img=True, 
            trunk_scale=1000,
            gt_leak=gt_leak, 
            gt_ls_Bl=gt_ls_Bl, 
            inference_mode=True,
            sampling_per_bits=sampling_per_bits,
            control_tokens=control_tokens,
        )
    
    print(f"cost: {time.time() - sstt}, infinity_pilot cost={time.time() - stt}")
    img = img_list[0]
    return img


def get_prompt_id(prompt):
    """獲取提示的 MD5 hash"""
    md5 = hashlib.md5()
    md5.update(prompt.encode('utf-8'))
    prompt_id = md5.hexdigest()
    return prompt_id


def save_slim_model(infinity_model_path, save_file=None, device='cpu', key='gpt_fsdp'):
    """保存精簡模型"""
    print('[Save slim model]')
    full_ckpt = torch.load(infinity_model_path, map_location=device, weights_only=False)
    infinity_slim = full_ckpt['trainer'][key]
    if not save_file:
        save_file = osp.splitext(infinity_model_path)[0] + '-slim.pth'
    print(f'Save to {save_file}')
    torch.save(infinity_slim, save_file)
    print('[Save slim model] done')
    return save_file


def load_tokenizer(t5_path=''):
    """加載文本編碼器"""
    print(f'[Loading tokenizer and text encoder]')
    text_tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    text_tokenizer.model_max_length = 512
    text_encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
    text_encoder.to('cuda')
    text_encoder.eval()
    text_encoder.requires_grad_(False)
    return text_tokenizer, text_encoder


def load_infinity_pilot(
    rope2d_each_sa_layer, 
    rope2d_normalized_by_hw, 
    use_scale_schedule_embedding, 
    pn, 
    use_bit_label, 
    add_lvl_embeding_only_first_block, 
    model_path='', 
    scale_schedule=None, 
    vae=None, 
    device='cuda', 
    model_kwargs=None,
    text_channels=2048,
    apply_spatial_patchify=0,
    use_flex_attn=False,
    bf16=False,
    checkpoint_type='torch',
):
    """加載 InfinityPilot 模型"""
    print(f'[Loading InfinityPilot]')
    text_maxlen = 512
    
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16, cache_enabled=True), torch.no_grad():
        infinity_pilot = InfinityPilot(
            vae_local=vae, 
            text_channels=text_channels, 
            text_maxlen=text_maxlen,
            shared_aln=True, 
            raw_scale_schedule=scale_schedule,
            checkpointing='full-block',
            customized_flash_attn=False,
            fused_norm=True,
            pad_to_multiplier=128,
            use_flex_attn=use_flex_attn,
            add_lvl_embeding_only_first_block=add_lvl_embeding_only_first_block,
            use_bit_label=use_bit_label,
            rope2d_each_sa_layer=rope2d_each_sa_layer,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            pn=pn,
            apply_spatial_patchify=apply_spatial_patchify,
            inference_mode=True,
            train_h_div_w_list=[1.0],
            **model_kwargs,
        ).to(device=device)
        
        print(f'[InfinityPilot with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_pilot.parameters())/1e9:.2f}B, bf16={bf16}')

        if bf16:
            for b_idx, block in enumerate(infinity_pilot.unregistered_blocks):
                block.bfloat16()

        infinity_pilot.eval()
        infinity_pilot.requires_grad_(False)
        infinity_pilot.cuda()
        torch.cuda.empty_cache()

        print(f'[Load InfinityPilot weights]')
        if checkpoint_type == 'torch':
            state_dict = torch.load(model_path, map_location=device, weights_only=False)
            if 'trainer' in state_dict:
                state_dict = state_dict['trainer']['gpt_fsdp']
            elif 'gpt_fsdp' in state_dict:
                state_dict = state_dict.get('gpt_ema_fsdp', state_dict['gpt_fsdp'])
            else:
                state_dict = state_dict
            print(infinity_pilot.load_state_dict(state_dict, strict=False))
        elif checkpoint_type == 'torch_shard':
            from transformers.modeling_utils import load_sharded_checkpoint
            load_sharded_checkpoint(infinity_pilot, model_path, strict=False)
        
        infinity_pilot.rng = torch.Generator(device=device)
        return infinity_pilot


def transform(pil_img, tgt_h, tgt_w):
    """圖像變換函數"""
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=PImage.LANCZOS)
    # crop the center out
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.add(im).add_(-1)


def load_visual_tokenizer(args):
    """加載視覺分詞器"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.vae_type in [14,16,18,20,24,32,64]:
        from infinity.models.bsq_vae.vae import vae_model
        schedule_mode = "dynamic"
        codebook_dim = args.vae_type
        codebook_size = 2**codebook_dim
        if args.apply_spatial_patchify:
            patch_size = 8
            encoder_ch_mult=[1, 2, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4]
        else:
            patch_size = 16
            encoder_ch_mult=[1, 2, 4, 4, 4]
            decoder_ch_mult=[1, 2, 4, 4, 4]
        vae = vae_model(args.vae_path, schedule_mode, codebook_dim, codebook_size, patch_size=patch_size, 
                        encoder_ch_mult=encoder_ch_mult, decoder_ch_mult=decoder_ch_mult, test_mode=True).to(device)
    else:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    return vae


def load_transformer(vae, args):
    """加載 InfinityPilot transformer"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model_path
    
    if args.checkpoint_type == 'torch': 
        if osp.exists(args.cache_dir):
            local_model_path = osp.join(args.cache_dir, 'tmp', model_path.replace('/', '_'))
        else:
            local_model_path = model_path
            
        if args.enable_model_cache:
            slim_model_path = model_path.replace('ar-', 'slim-')
            local_slim_model_path = local_model_path.replace('ar-', 'slim-')
            os.makedirs(osp.dirname(local_slim_model_path), exist_ok=True)
            print(f'model_path: {model_path}, slim_model_path: {slim_model_path}')
            print(f'local_model_path: {local_model_path}, local_slim_model_path: {local_slim_model_path}')
            if not osp.exists(local_slim_model_path):
                if osp.exists(slim_model_path):
                    print(f'copy {slim_model_path} to {local_slim_model_path}')
                    shutil.copyfile(slim_model_path, local_slim_model_path)
                else:
                    if not osp.exists(local_model_path):
                        print(f'copy {model_path} to {local_model_path}')
                        shutil.copyfile(model_path, local_model_path)
                    save_slim_model(local_model_path, save_file=local_slim_model_path, device=device)
                    print(f'copy {local_slim_model_path} to {slim_model_path}')
                    if not osp.exists(slim_model_path):
                        shutil.copyfile(local_slim_model_path, slim_model_path)
                        os.remove(local_model_path)
                        os.remove(model_path)
            slim_model_path = local_slim_model_path
        else:
            slim_model_path = model_path
        print(f'load checkpoint from {slim_model_path}')
    elif args.checkpoint_type == 'torch_shard':
        slim_model_path = model_path

    # 模型配置
    if args.model_type == 'infinity_pilot_2b':
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_pilot_8b':
        kwargs_model = dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    elif args.model_type == 'infinity_pilot_layer12':
        kwargs_model = dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_pilot_layer16':
        kwargs_model = dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_pilot_layer24':
        kwargs_model = dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_pilot_layer32':
        kwargs_model = dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_pilot_layer40':
        kwargs_model = dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    elif args.model_type == 'infinity_pilot_layer48':
        kwargs_model = dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4)
    else:
        # 默認配置
        kwargs_model = dict(depth=32, embed_dim=2048, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    
    infinity_pilot = load_infinity_pilot(
        rope2d_each_sa_layer=args.rope2d_each_sa_layer, 
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        use_scale_schedule_embedding=args.use_scale_schedule_embedding,
        pn=args.pn,
        use_bit_label=args.use_bit_label, 
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block, 
        model_path=slim_model_path, 
        scale_schedule=None, 
        vae=vae, 
        device=device, 
        model_kwargs=kwargs_model,
        text_channels=args.text_channels,
        apply_spatial_patchify=args.apply_spatial_patchify,
        use_flex_attn=args.use_flex_attn,
        bf16=args.bf16,
        checkpoint_type=args.checkpoint_type,
    )
    return infinity_pilot


def add_common_arguments(parser):
    """添加通用參數"""
    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cfg_insertion_layer', type=int, default=0)
    parser.add_argument('--vae_type', type=int, default=1)
    parser.add_argument('--vae_path', type=str, default='')
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=0, choices=[0,1])
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0,1])
    parser.add_argument('--model_type', type=str, default='infinity_pilot_2b')
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0,1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0,1,2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0,1])
    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1,2,4,8,16])
    parser.add_argument('--text_encoder_ckpt', type=str, default='')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0,1])
    parser.add_argument('--h_div_w_template', type=float, default=1.000)
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0,1])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0,1])
    parser.add_argument('--cache_dir', type=str, default='/dev/shm')
    parser.add_argument('--enable_model_cache', type=int, default=0, choices=[0,1])
    parser.add_argument('--checkpoint_type', type=str, default='torch')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0,1])
    # InfinityPilot 特有參數
    parser.add_argument('--control_image', type=str, default='', help='Path to control image')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run InfinityPilot for text-to-image generation with optional control')
    add_common_arguments(parser)
    parser.add_argument('--prompt', type=str, default='a dog')
    parser.add_argument('--save_file', type=str, default='./output_infinity_pilot.jpg')
    args = parser.parse_args()

    # 解析 cfg
    args.cfg = list(map(float, args.cfg.split(',')))
    if len(args.cfg) == 1:
        args.cfg = args.cfg[0]
    
    print("=== InfinityPilot Image Generation ===")
    print(f"Prompt: {args.prompt}")
    if args.control_image:
        print(f"Control Image: {args.control_image}")
    print(f"Model: {args.model_type}")
    print(f"Output: {args.save_file}")
    print()
    
    # 加載文本編碼器
    print("Loading text encoder...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    
    # 加載 VAE
    print("Loading visual tokenizer...")
    vae = load_visual_tokenizer(args)
    
    # 加載 InfinityPilot
    print("Loading InfinityPilot...")
    infinity_pilot = load_transformer(vae, args)
    
    # 設置尺度排程
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    print(f"Scale schedule: {scale_schedule}")

    # 生成圖像
    print("Generating image...")
    with autocast(dtype=torch.bfloat16):
        with torch.no_grad():
            generated_image = gen_one_img_pilot(
                infinity_pilot,
                vae,
                text_tokenizer,
                text_encoder,
                args.prompt,
                control_path=args.control_image if args.control_image else None,
                g_seed=args.seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=args.cfg,
                tau_list=args.tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
            )
    
    # 保存圖像
    os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
    cv2.imwrite(args.save_file, generated_image.cpu().numpy())
    print(f'✓ Image saved to: {osp.abspath(args.save_file)}')
    print("=== Generation Complete ===")
