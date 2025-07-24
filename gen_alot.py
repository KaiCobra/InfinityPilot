import random
import re
import os
import os.path as osp
import json
import torch
torch.cuda.set_device(0)
import cv2
import numpy as np
from tools.run_infinity import *

def extract_between_quotes(text: str) -> list[str]:
    """
    Extract all unique substrings enclosed in double quotes from the given text.

    Args:
        text: A string potentially containing substrings within \"...\".

    Returns:
        A list of unique substrings found between the escaped double quotes, in order.
    """
    # Pattern: \"   => literal escaped double-quote
    # (.*?) => non-greedy capture of any characters
    # \"     => literal escaped double-quote
    pattern = r'\"(.*?)\"'
    matches = re.findall(pattern, text)

    # Remove duplicates while preserving order
    unique_matches: list[str] = []
    seen: set[str] = set()
    for match in matches:
        if match not in seen:
            unique_matches.append(match)
            seen.add(match)

    return unique_matches

model_path='./weights/mm_2b.pth'
vae_path='./weights/infinity_vae_d32_reg.pth'
text_encoder_ckpt = "./weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001"

args=argparse.Namespace(
    pn='1M',
    model_path=model_path,
    cfg_insertion_layer=0,
    vae_type=32,
    vae_path=vae_path,
    add_lvl_embeding_only_first_block=1,
    use_bit_label=1,
    model_type='infinity_2b',
    rope2d_each_sa_layer=1,
    rope2d_normalized_by_hw=2,
    use_scale_schedule_embedding=0,
    sampling_per_bits=1,
    text_encoder_ckpt=text_encoder_ckpt,
    
    text_channels=2048,
    apply_spatial_patchify=0,
    h_div_w_template=1.000,
    use_flex_attn=0,
    cache_dir='/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/VAR_ckpt/local_output/toy',
    checkpoint_type='torch',
    enable_model_cache=True, #modify
    seed=0,
    bf16=1,
    save_file='tmp.jpg',
    noise_apply_layers=1,
    noise_apply_requant=1,
    noise_apply_strength=0.01,
)

# load text encoder
text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
# load vae
vae = load_visual_tokenizer(args)
# load infinity
infinity = load_transformer(vae, args)

data_root = "/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/SynData/real_data/SVT"
data_img_pth = osp.join(data_root, "images")

for img in sorted(os.listdir(data_img_pth)):
    img_pth = osp.join(data_img_pth, img)
    metadata_pth = img_pth.replace('images', 'metadatas')
    metadata_pth = metadata_pth.replace('.jpg', '.json')
    with open(metadata_pth, 'r') as f:
        metadata = json.load(f)
    content = metadata['long_caption']
    short_content = metadata['text']
    content = content + short_content
    matches = extract_between_quotes(content)
    print(img)
    # print(matches)

    prompt = "Render the text \"" + "\" and \"".join(matches) + "\" in the following image: " + metadata['text']
    print(prompt)
    prompt = metadata['long_caption']
    # continue

    cfg = 3
    tau = 0.5 # my setting
    h_div_w = 1/1 # aspect ratio, height:width
    seed = random.randint(0, 10000)
    enable_positive_prompt=0

    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    # ----------------------------------------------------------------------------------
    # modified
    from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
    if not hasattr(args, 'debug_bsc'):
        args.debug_bsc = False
    bitwise_self_correction = BitwiseSelfCorrection(vae, args)
    infinity.setup_mask_processor(vae, scale_schedule, bitwise_self_correction)
    mask_path = img_pth
    # -----------------------------------------------------
    mask_path = mask_path.replace('images', 'normal_vis') #for control
    mask_path = mask_path.replace('.jpg', '.png')
    # -----------------------------------------------------
    infinity.set_mask(
        mask_path=mask_path,
        scale_idx=[0,1,2,3,4,5,6,7,8,9,10,11],
        method='weighted',
        alpha=0.3
    )
    # ----------------------------------------------------------------------------------
    generated_image = gen_one_img(
        infinity,
        vae,
        text_tokenizer,
        text_encoder,
        prompt,
        g_seed=seed,
        gt_leak=0,
        gt_ls_Bl=None,
        cfg_list=cfg,
        tau_list=tau,
        scale_schedule=scale_schedule,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=enable_positive_prompt,
    )
    args.save_file = './generated_control5/' + img
    os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
    cv2.imwrite(args.save_file, generated_image.cpu().numpy())
    print(f'Save to {osp.abspath(args.save_file)}')
