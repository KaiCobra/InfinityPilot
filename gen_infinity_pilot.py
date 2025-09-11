#!/usr/bin/env python3
"""
基於 gen.py 的 InfinityPilot 運行腳本
使用 gen.py 的結構來運行 run_infinity_pilot.py
"""

import random
import torch
import cv2
import numpy as np
import os
import os.path as osp
import argparse
from pathlib import Path

from torchview import draw_graph
import torch.nn as nn

# 設置 CUDA 設備
torch.cuda.set_device(0)

# 導入我們的 run_infinity_pilot 模組
import sys
sys.path.append('/home/avlab/SceneTxtVAR')
from tools.run_infinity_pilot import *
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

def setup_args():
    """設置參數，類似 gen.py 的配置"""
    
    # 模型路徑配置
    model_path = './weights/mm_2b.pth'  # InfinityPilot 模型路徑
    vae_path = './weights/infinity_vae_d32_reg.pth'
    text_encoder_ckpt = "./weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001"
    
    args = argparse.Namespace(
        # 基本模型配置
        pn='1M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_pilot_2b',  # 使用 InfinityPilot 模型類型
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        
        # 進階配置
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch',
        enable_model_cache=True,
        seed=0,
        bf16=1,
        save_file='output_infinity_pilot.jpg',
        
        # InfinityPilot 特有配置
        control_image='',  # 控制圖像路徑
        enable_positive_prompt=0,
    )
    
    return args

def load_models(args):
    """加載所有模型"""
    print("=== Loading Models ===")
    
    # 加載文本編碼器
    print("Loading text encoder...")
    text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
    
    # 加載 VAE
    print("Loading visual tokenizer...")
    vae = load_visual_tokenizer(args)
    
    # 加載 InfinityPilot
    print("Loading InfinityPilot...")
    infinity_pilot = load_transformer(vae, args)
    
    print("✓ All models loaded successfully!")
    return text_tokenizer, text_encoder, vae, infinity_pilot

def generate_with_prompts(text_tokenizer, text_encoder, vae, infinity_pilot, args):
    """使用多個提示詞生成圖像"""
    
    # 定義多個測試提示詞
    prompts = [
        """
        System prompt: Render the text "LAWSON" in the following image:
        A photograph of a Japanese convenience store, "LAWSON", set against the iconic Mount Fuji in the early morning. The store occupies the lower third of the frame, with its signature blue and white signboard featuring bold, capitalized lettering "LAWSON". In the background, the majestic Mount Fuji, covered in snow, dominates the upper portion of the frame, bathed in soft morning sunlight that casts a pinkish hue across its rugged slopes. The sky transitions from a gentle blue to a warm golden glow, enhancing the tranquil and serene atmosphere.
        """,
        
        """
        Render the text "Target" in the following image:
        This image shows the exterior of a large retail store on a sunny day. The building has a modern, clean architectural design with neutral tones of white and gray. A prominent feature is the bright blue rectangular sign centered above the entrance, which displays the name "Target" in bold, white letters. Next to the text is a simple, iconic yellow spark-like symbol that accompanies the branding.
        """,
        
        """
        System prompt: Render the text "ANNO DOMINI","SANER" in the following image:
        The image is taken from a slightly elevated angle, providing a clear view of the theater marquee. The marquee displays the text "ANNO DOMINI" in large, bold letters at the top, followed by "SANER" in smaller letters below. The background consists of a gray brick wall with some greenery peeking through at the top. The foreground features several hanging street lamps, adding to the urban setting.
        """,
        
        """
        A beautiful landscape with mountains and lake, peaceful scenery at sunset with warm golden light reflecting on the water surface.
        """
    ]
    
    # 生成參數
    cfg = 2.8
    tau = 0.53
    h_div_w = 1.0  # 長寬比
    
    # 計算尺度排程
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    
    print(f"Scale schedule: {scale_schedule}")
    print(f"Generation parameters: cfg={cfg}, tau={tau}")
    print()
    
    # 為每個提示詞生成圖像
    for i, prompt in enumerate(prompts):
        print(f"=== Generating Image {i+1}/{len(prompts)} ===")
        print(f"Prompt preview: {prompt[:100]}...")
        
        # 設置隨機種子
        seed = random.randint(0, 10000)
        print(f"Using seed: {seed}")
        
        # 設置輸出文件名
        output_file = f"output_infinity_pilot_{i+1}.jpg"
        
        # 生成圖像
        try:
            generated_image = gen_one_img_pilot(
                infinity_pilot,
                vae,
                text_tokenizer,
                text_encoder,
                prompt,
                control_path=args.control_image if args.control_image else None,
                g_seed=seed,
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=cfg,
                tau_list=tau,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[args.cfg_insertion_layer],
                vae_type=args.vae_type,
                sampling_per_bits=args.sampling_per_bits,
                enable_positive_prompt=args.enable_positive_prompt,
            )
            
            # 保存圖像
            os.makedirs(osp.dirname(osp.abspath(output_file)), exist_ok=True)
            cv2.imwrite(output_file, generated_image.cpu().numpy())
            print(f'✓ Image saved to: {osp.abspath(output_file)}')
            
        except Exception as e:
            print(f'✗ Error generating image {i+1}: {str(e)}')
            continue
        
        print()

def generate_single_image(text_tokenizer, text_encoder, vae, infinity_pilot, args, prompt, output_file=None):
    """生成單一圖像"""
    
    if output_file is None:
        output_file = args.save_file
    
    # 生成參數
    cfg = 2.8
    tau = 0.53
    h_div_w = 1.0
    
    # 計算尺度排程
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    
    # 設置隨機種子
    seed = random.randint(0, 10000) if args.seed == 0 else args.seed
    
    print(f"=== Generating Single Image ===")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Seed: {seed}")
    print(f"Output: {output_file}")
    
    # 生成圖像
    generated_image = gen_one_img_pilot(
        infinity_pilot,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        control_path=args.control_image if args.control_image else None,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=args.enable_positive_prompt,
    )
    
    # 保存圖像
    os.makedirs(osp.dirname(osp.abspath(output_file)), exist_ok=True)
    cv2.imwrite(output_file, generated_image.cpu().numpy())
    print(f'✓ Image saved to: {osp.abspath(output_file)}')
    
    return generated_image

def setup_mask_processing(infinity_pilot, vae, scale_schedule, args):
    """設置遮罩處理（類似 gen.py 中的功能）"""
    try:
        from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
        
        if not hasattr(args, 'debug_bsc'):
            args.debug_bsc = False
            
        bitwise_self_correction = BitwiseSelfCorrection(vae, args)
        infinity_pilot.setup_mask_processor(vae, scale_schedule, bitwise_self_correction)
        
        # 設置遮罩（如果有提供遮罩路徑）
        mask_path = '/home/avlab/SceneTxtVAR/control/lawson.jpg'
        if osp.exists(mask_path):
            infinity_pilot.set_mask(
                mask_path=mask_path,
                scale_idx=[0,1,2,3,4,5,6,7,8,9,10,11],
                method='weighted',
                alpha=0.3,
                strength=[0.35, 0.94, 0.97, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
            print("✓ Mask processing setup complete")
        else:
            print("! Mask file not found, skipping mask processing")
            
    except ImportError:
        print("! BitwiseSelfCorrection not available, skipping mask processing")
    except Exception as e:
        print(f"! Error setting up mask processing: {str(e)}")

def main():
    """主函數"""
    print("=== InfinityPilot Generation Script (based on gen.py) ===")
    print()
    
    # 設置參數
    args = setup_args()
    
    # 加載模型
    text_tokenizer, text_encoder, vae, infinity_pilot = load_models(args)
    
    # 計算尺度排程
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - args.h_div_w_template))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    # 單一圖像生成
    prompt = "A beautiful landscape with mountains and lake at sunset"

    control_image = '/home/avlab/SceneTxtVAR/text/02_06.jpg'

    if control_image and osp.exists(control_image):
        args.control_image = control_image
        
        # 為每個 scale 準備控制張量
        control_tensors = []
        control_img = Image.open(control_image).convert('RGB')
        for pt, ph, pw in scale_schedule:
            target_h, target_w = ph * 16, pw * 16
            control_resized = control_img.resize((target_w, target_h))
            control_tensor = transform(control_resized, target_h, target_w)
            control_tensors.append(control_tensor.unsqueeze(0).cuda())

    if control_image:
        args.control_image = control_image

    generate_single_image(text_tokenizer, text_encoder, vae, infinity_pilot, args, prompt)

if __name__ == "__main__":
    main()
