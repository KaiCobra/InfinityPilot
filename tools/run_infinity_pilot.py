#!/usr/bin/env python3
"""
Utility script for running InfinityPilot inference.
The script mirrors the updated runtime used by InfinityPilot training, and
supports loading base Infinity weights together with optional CAR modules
for controlled generation.
"""
import os, sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os.path as osp
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from transformers import AutoTokenizer, T5EncoderModel, T5TokenizerFast

from infinity.models.infinity_pilot import InfinityPilot
from infinity.models.infinity import MultipleLayers
from infinity.models.basic import *  # noqa: F401,F403 required for model construction
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w


def _as_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _parse_list_arg(value: str, cast_fn=float) -> List[float]:
    items = [cast_fn(v) for v in value.split(',') if v.strip()]
    return items if items else [cast_fn(0)]


def _ensure_list(values, length: int) -> List[float]:
    if not isinstance(values, list):
        values = [values]
    if not values:
        values = [1.0]
    if len(values) < length:
        values = values + [values[-1]] * (length - len(values))
    return values


def _requant_all_scales(
    bsc: BitwiseSelfCorrection,
    vae_scale_schedule: List[Tuple[int, int, int]],
    raw_features: torch.Tensor,
) -> List[torch.Tensor]:
    with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
        if raw_features.dim() == 4:
            codes_out = raw_features.unsqueeze(2)
        else:
            codes_out = raw_features

        final_size = vae_scale_schedule[-1]
        if codes_out.shape[-3:] != final_size:
            codes_out = F.interpolate(
                codes_out,
                size=final_size,
                mode=bsc.vae.quantizer.z_interplote_up,
            ).contiguous()

        cum_var_input = 0
        per_scale_tokens: List[torch.Tensor] = []

        for si, (pt, ph, pw) in enumerate(vae_scale_schedule):
            residual = codes_out - cum_var_input
            if si != len(vae_scale_schedule) - 1:
                residual = F.interpolate(
                    residual,
                    size=vae_scale_schedule[si],
                    mode=bsc.vae.quantizer.z_interplote_down,
                ).contiguous()

            quantized, _, _, _ = bsc.vae.quantizer.lfq(residual)

            tokens = quantized.squeeze(2)
            if bsc.apply_spatial_patchify:
                tokens = torch.nn.functional.pixel_unshuffle(tokens, 2)
            tokens = tokens.flatten(2).transpose(1, 2).contiguous()
            per_scale_tokens.append(tokens)

            cum_var_input = cum_var_input + F.interpolate(
                quantized,
                size=vae_scale_schedule[-1],
                mode=bsc.vae.quantizer.z_interplote_up,
            ).contiguous()

    return per_scale_tokens


def transform(pil_img: Image.Image, tgt_h: int, tgt_w: int) -> torch.Tensor:
    width, height = pil_img.size
    if width / height <= tgt_w / tgt_h:
        resized_width = tgt_w
        resized_height = int(tgt_w / (width / height))
    else:
        resized_height = tgt_h
        resized_width = int((width / height) * tgt_h)
    pil_img = pil_img.resize((resized_width, resized_height), resample=Image.LANCZOS)
    arr = np.array(pil_img)
    crop_y = (arr.shape[0] - tgt_h) // 2
    crop_x = (arr.shape[1] - tgt_w) // 2
    im = to_tensor(arr[crop_y: crop_y + tgt_h, crop_x: crop_x + tgt_w])
    return im.mul_(2).add_(-1)


def aug_with_positive_prompt(prompt: str) -> str:
    people_tokens = {
        'man', 'woman', 'men', 'women', 'boy', 'girl', 'child', 'person',
        'human', 'adult', 'teenager', 'employee', 'employer', 'worker',
        'mother', 'father', 'sister', 'brother', 'grandmother', 'grandfather',
        'son', 'daughter'
    }
    if any(token in prompt for token in people_tokens):
        return prompt + '. very smooth faces, good looking faces, face to the camera, perfect facial features'
    return prompt


def encode_prompt(
    text_tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    prompt: str,
    device: torch.device,
    enable_positive_prompt: bool = False,
) -> Tuple[torch.Tensor, List[int], torch.Tensor, int]:
    if enable_positive_prompt:
        prompt = aug_with_positive_prompt(prompt)
    tokens = text_tokenizer(
        text=[prompt], max_length=512, padding='max_length', truncation=True, return_tensors='pt'
    )
    input_ids = tokens.input_ids.to(device, non_blocking=True)
    mask = tokens.attention_mask.to(device, non_blocking=True)
    with torch.no_grad():
        text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    lens: List[int] = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    Ltext = max(lens)
    kv_compact = torch.cat([feat_i[:len_i] for len_i, feat_i in zip(lens, text_features.unbind(0))], dim=0)
    return kv_compact, lens, cu_seqlens_k, Ltext


def load_tokenizer(t5_path: str, device: torch.device) -> Tuple[T5TokenizerFast, T5EncoderModel]:
    tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(t5_path, revision=None, legacy=True)
    tokenizer.model_max_length = 512
    encoder: T5EncoderModel = T5EncoderModel.from_pretrained(t5_path, torch_dtype=torch.float16)
    encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)
    return tokenizer, encoder


def load_visual_tokenizer(args: argparse.Namespace, device: torch.device):
    if args.vae_type not in [14, 16, 18, 20, 24, 32, 64]:
        raise ValueError(f'vae_type={args.vae_type} not supported')
    from infinity.models.bsq_vae.vae import vae_model

    schedule_mode = "dynamic"
    codebook_dim = args.vae_type
    codebook_size = 2 ** codebook_dim
    if args.apply_spatial_patchify:
        patch_size = 8
        encoder_ch_mult = [1, 2, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4]
    else:
        patch_size = 16
        encoder_ch_mult = [1, 2, 4, 4, 4]
        decoder_ch_mult = [1, 2, 4, 4, 4]

    vae = vae_model(
        args.vae_path,
        schedule_mode,
        codebook_dim,
        codebook_size,
        patch_size=patch_size,
        encoder_ch_mult=encoder_ch_mult,
        decoder_ch_mult=decoder_ch_mult,
        test_mode=True,
    ).to(device)
    vae.eval()
    return vae


def _resolve_state_dict(blob: dict) -> dict:
    candidates = []
    if isinstance(blob, dict):
        trainer = blob.get('trainer') if isinstance(blob.get('trainer'), dict) else None
        if trainer:
            candidates.extend([
                trainer.get('gpt_fsdp'),
                trainer.get('gpt_ema_fsdp'),
                trainer.get('gpt'),
                trainer.get('model'),
            ])
        candidates.extend([
            blob.get('gpt_ema_fsdp'),
            blob.get('gpt_fsdp'),
            blob.get('state_dict'),
            blob.get('model'),
        ])
    for candidate in candidates:
        if isinstance(candidate, dict):
            return candidate
    return blob


def load_checkpoint(path: str) -> dict:
    print(f'[load] reading checkpoint: {path}')
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    state = _resolve_state_dict(ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f'Unsupported checkpoint format in {path}')
    return state


_MODEL_PRESETS: Dict[str, Dict[str, int]] = {
    'infinity_2b': dict(depth=32, embed_dim=2048, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8),
    'infinity_8b': dict(depth=40, embed_dim=3584, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=8),
    'infinity_layer12': dict(depth=12, embed_dim=768, num_heads=8, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
    'infinity_layer16': dict(depth=16, embed_dim=1152, num_heads=12, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
    'infinity_layer24': dict(depth=24, embed_dim=1536, num_heads=16, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
    'infinity_layer32': dict(depth=32, embed_dim=2080, num_heads=20, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
    'infinity_layer40': dict(depth=40, embed_dim=2688, num_heads=24, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
    'infinity_layer48': dict(depth=48, embed_dim=3360, num_heads=28, drop_path_rate=0.1, mlp_ratio=4, block_chunks=4),
}


def _model_kwargs_for_type(model_type: str) -> Dict[str, int]:
    normalized = model_type.lower()
    normalized = normalized.replace('infinity_pilot', 'infinity')
    if normalized not in _MODEL_PRESETS:
        raise ValueError(f'Unknown model_type: {model_type}')
    return _MODEL_PRESETS[normalized]

@torch.no_grad()
def build_infinity_pilot(
    args: argparse.Namespace,
    vae,
    device: torch.device,
) -> InfinityPilot:
    model_kwargs = _model_kwargs_for_type(args.model_type)
    configured_chunks = int(args.block_chunks) if int(args.block_chunks) > 0 else model_kwargs.get('block_chunks', 1)
    model_kwargs['block_chunks'] = configured_chunks

    base_state = None
    if args.checkpoint_type == 'torch' and args.model_path:
        base_state = load_checkpoint(args.model_path)

    pilot_kwargs = dict(
        vae_local=vae,
        text_channels=args.text_channels,
        text_maxlen=512,
        raw_scale_schedule=None,
        checkpointing='full-block',
        customized_flash_attn=False,
        fused_norm=True,
        pad_to_multiplier=128,
        use_flex_attn=bool(args.use_flex_attn),
        add_lvl_embeding_only_first_block=int(args.add_lvl_embeding_only_first_block),
        use_bit_label=int(args.use_bit_label),
        rope2d_each_sa_layer=int(args.rope2d_each_sa_layer),
        rope2d_normalized_by_hw=int(args.rope2d_normalized_by_hw),
        pn=args.pn,
        apply_spatial_patchify=int(args.apply_spatial_patchify),
        inference_mode=True,
        train_h_div_w_list=[1.0],
        disable_car_fusion=bool(args.disable_car_fusion),
        disable_car_merge=bool(args.disable_car_merge),
        init_car_modules=False,
        freeze_infinity=bool(args.freeze_infinity),
        car_depth=int(args.car_depth),
        **model_kwargs,
    )
    if args.shared_aln in (0, 1):
        pilot_kwargs['shared_aln'] = bool(args.shared_aln)
    if int(args.car_depth) > 0:
        pilot_kwargs['car_depth'] = int(args.car_depth)

    pilot = InfinityPilot(infinity_base_model=base_state, **pilot_kwargs).to(device=device)
    if not hasattr(pilot, 'block_chunks'):
        pilot.block_chunks = torch.nn.ModuleList([
            MultipleLayers(pilot.unregistered_blocks, len(pilot.unregistered_blocks), 0)
        ])
        pilot.num_block_chunks = 1
    base_state = None

    pilot.eval()
    pilot.requires_grad_(False)
    pilot.rng = torch.Generator(device=device)

    if args.checkpoint_type == 'torch_shard':
        from transformers.modeling_utils import load_sharded_checkpoint
        load_sharded_checkpoint(pilot, args.model_path, strict=False)
        return pilot
    else:
        if bool(args.init_car_modules) or args.car_path or args.pilot_checkpoint:
            pilot.init_car_modules_if_needed()
            pilot.to(device)

    if args.pilot_checkpoint:
        full_state = load_checkpoint(args.pilot_checkpoint)
        missing, unexpected = pilot.load_state_dict(full_state, strict=bool(args.strict_load))
        print(f'[pilot] load_state_dict missing={len(missing)} unexpected={len(unexpected)}')
    elif args.car_path:
        car_state = load_checkpoint(args.car_path)
        pilot.load_car_weights(car_state, strict=bool(args.strict_load))
    else:
        # If user did not request explicit CAR init, match training default when base weights available.
        pilot.init_car_modules_if_needed()
        pilot.to(device)

    return pilot


def _control_paths_from_args(args: argparse.Namespace) -> Dict[str, str]:
    control_paths = {}
    if args.control_image:
        control_paths['control'] = args.control_image
    if args.control_normal:
        control_paths['normal'] = args.control_normal
    if args.control_mask:
        control_paths['mask'] = args.control_mask
    return {k: v for k, v in control_paths.items() if v}


def _build_control_tokens(
    control_paths: Dict[str, str],
    vae,
    scale_schedule: List[Tuple[int, int, int]],
    device: torch.device,
    apply_spatial_patchify: bool,
) -> Optional[List[Optional[torch.Tensor]]]:
    if not control_paths:
        return None

    vae_scale_schedule = [
        (pt, 2 * ph, 2 * pw) if apply_spatial_patchify else (pt, ph, pw)
        for pt, ph, pw in scale_schedule
    ]
    target_h = vae_scale_schedule[-1][1]
    target_w = vae_scale_schedule[-1][2]
    args_ns = SimpleNamespace(
        noise_apply_layers=0,
        noise_apply_requant=1,
        noise_apply_strength=0.0,
        apply_spatial_patchify=apply_spatial_patchify,
        debug_bsc=0,
    )
    bsc = BitwiseSelfCorrection(vae, args_ns)

    tokens_by_type: Dict[str, List[torch.Tensor]] = {}
    for name, path in control_paths.items():
        if not osp.exists(path):
            print(f'[warn] control image missing: {path}')
            continue
        pil_img = Image.open(path).convert('RGB')
        tensor = transform(pil_img, target_h, target_w).unsqueeze(0).to(device)
        with torch.no_grad():
            raw_features, _, _ = vae.encode_for_raw_features(tensor, scale_schedule=vae_scale_schedule)
            per_scale = _requant_all_scales(bsc, vae_scale_schedule, raw_features)

        tokens_by_type[name] = [tok.to(device) for tok in per_scale]
        print(f'[control] loaded {name} tokens across {len(per_scale)} scales')

    if not tokens_by_type:
        return None

    combined: List[Optional[torch.Tensor]] = []
    num_scales = len(scale_schedule)
    for si in range(num_scales):
        tokens_at_scale = [tokens[si] for tokens in tokens_by_type.values() if si < len(tokens)]
        if not tokens_at_scale:
            combined.append(None)
            continue
        if len(tokens_at_scale) == 1:
            combined.append(tokens_at_scale[0])
        else:
            stacked = torch.stack(tokens_at_scale, dim=0)
            combined.append(stacked.mean(dim=0))
    return combined


def gen_one_img_pilot(
    infinity_pilot: InfinityPilot,
    vae,
    text_tokenizer: T5TokenizerFast,
    text_encoder: T5EncoderModel,
    prompt: str,
    *,
    scale_schedule: List[Tuple[int, int, int]],
    cfg_list,
    tau_list,
    device: torch.device,
    negative_prompt: str = '',
    g_seed: Optional[int] = None,
    cfg_sc: float = 3.0,
    cfg_exp_k: float = 0.0,
    cfg_insertion_layer: Optional[List[int]] = None,
    top_k: int = 900,
    top_p: float = 0.97,
    vae_type: int = 0,
    sampling_per_bits: int = 1,
    enable_positive_prompt: bool = False,
    softmax_merge_topk: int = -1,
    gumbel: int = 0,
    control_tokens: Optional[List[Optional[torch.Tensor]]] = None,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if cfg_insertion_layer is None:
        cfg_insertion_layer = [-5]

    text_cond_tuple = encode_prompt(
        text_tokenizer,
        text_encoder,
        prompt,
        device=device,
        enable_positive_prompt=enable_positive_prompt,
    )

    if negative_prompt:
        negative_label = encode_prompt(text_tokenizer, text_encoder, negative_prompt, device=device)
    else:
        negative_label = None

    cfg_schedule = _ensure_list(cfg_list, len(scale_schedule))
    tau_schedule = _ensure_list(tau_list, len(scale_schedule))

    with torch.amp.autocast('cuda', enabled=True, dtype=amp_dtype):
        _, _, img_list = infinity_pilot.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            negative_label_B_or_BLT=negative_label,
            g_seed=g_seed,
            B=1,
            cfg_sc=cfg_sc,
            cfg_list=cfg_schedule,
            tau_list=tau_schedule,
            top_k=top_k,
            top_p=top_p,
            cfg_exp_k=cfg_exp_k,
            cfg_insertion_layer=cfg_insertion_layer,
            vae_type=vae_type,
            sampling_per_bits=sampling_per_bits,
            softmax_merge_topk=softmax_merge_topk,
            gumbel=gumbel,
            gt_leak=-1,
            gt_ls_Bl=None,
            inference_mode=True,
            ret_img=True,
            returns_vemb=1,
            control_tokens=control_tokens,
        )
    return img_list[0]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    prompts=[
             "A high-resolution photograph of a modern bathroom interior, featuring a sleek towel warmer mounted on the wall beside a white door. The room is well-lit, showcasing beige tiled flooring and light-colored walls that create a clean and inviting atmosphere.",
             "A close-up view of the towel warmer, highlighting its elegant design and the soft, warm lighting that enhances the bathroom's ambiance.",
             "A beautifully designed bathroom with a contemporary aesthetic, highlighting a stylish towel warmer next to a white door.",
             "The image is taken from a slightly elevated angle, showcasing a bathroom interior with a vertical camera orientation. The prominent features include a towel warmer on the wall to the left and a white door in the center. The floor is adorned with beige tiles, complementing the light-colored walls. The background displays a clean, minimalistic design, emphasizing functionality with essential fixtures such as a mirrored cabinet above the bathtub.",
             ]

    parser.add_argument('--prompt', type=str, default='18-year-old porcelain-skinned Asian beauty, full-body **completely nude** on Tokyo rooftop at golden hour. **Extreme hourglass**: **razor-thin waist**, **toned flat abs**, erupting into **colossal, ultra-firm bubble butt** and **perky HH-cup natural breasts** that defy gravity. Arched back, mega-butt thrust toward camera, sultry over-shoulder smirk. Photorealistic, cinematic lighting, 8K.')
    parser.add_argument('--negative_prompt', type=str, default='')
    parser.add_argument('--save_file', type=str, default='./output_pilot/output_infinity_pilot_wo_condition_2.jpg')

    parser.add_argument('--cfg', type=str, default='3')
    parser.add_argument('--tau', type=str, default='1')
    parser.add_argument('--cfg_sc', type=float, default=3.0)
    parser.add_argument('--cfg_exp_k', type=float, default=0.0)
    parser.add_argument('--cfg_insertion_layer', type=int, default=-5)
    parser.add_argument('--top_k', type=int, default=900)
    parser.add_argument('--top_p', type=float, default=0.97)

    parser.add_argument('--pn', type=str, required=True, choices=['0.06M', '0.25M', '1M'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--pilot_checkpoint', type=str, default='')
    parser.add_argument('--car_path', type=str, default='/media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/output_a6000_bsqvae_parallel_toy/car_weights_ep0007_it008000/car_weights.pth')
    parser.add_argument('--car_depth', type=int, default=8)
    parser.add_argument('--block_chunks', type=int, default=1)
    parser.add_argument('--checkpoint_type', type=str, default='torch', choices=['torch', 'torch_shard'])

    parser.add_argument('--init_car_modules', type=int, default=1, choices=[0, 1])
    parser.add_argument('--freeze_infinity', type=int, default=1, choices=[0, 1])
    parser.add_argument('--disable_car_fusion', type=int, default=0, choices=[0, 1])
    parser.add_argument('--disable_car_merge', type=int, default=0, choices=[0, 1])
    parser.add_argument('--shared_aln', type=int, default=-1, choices=[-1, 0, 1])
    parser.add_argument('--strict_load', type=int, default=0, choices=[0, 1])

    parser.add_argument('--text_encoder_ckpt', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='infinity_2b')
    parser.add_argument('--text_channels', type=int, default=2048)
    parser.add_argument('--use_bit_label', type=int, default=1, choices=[0, 1])
    parser.add_argument('--add_lvl_embeding_only_first_block', type=int, default=1, choices=[0, 1])
    parser.add_argument('--rope2d_each_sa_layer', type=int, default=1, choices=[0, 1])
    parser.add_argument('--rope2d_normalized_by_hw', type=int, default=2, choices=[0, 1, 2])
    parser.add_argument('--use_scale_schedule_embedding', type=int, default=0, choices=[0, 1])
    parser.add_argument('--use_flex_attn', type=int, default=0, choices=[0, 1])

    parser.add_argument('--vae_type', type=int, default=32)
    parser.add_argument('--vae_path', type=str, required=True)
    parser.add_argument('--apply_spatial_patchify', type=int, default=0, choices=[0, 1])

    parser.add_argument('--sampling_per_bits', type=int, default=1, choices=[1, 2, 4, 8, 16])
    parser.add_argument('--enable_positive_prompt', type=int, default=0, choices=[0, 1])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bf16', type=int, default=1, choices=[0, 1])

    parser.add_argument('--h_div_w_template', type=float, default=2)
    parser.add_argument('--always_training_scales', type=int, default=13)
    parser.add_argument('--control_image', type=str, default='')
    parser.add_argument('--control_normal', type=str, default='')
    parser.add_argument('--control_mask', type=str, default='')
    # parser.add_argument('--control_image', type=str, default='/mnt/syndata/Syn3DTxt_scene_2words_toydataset/images_sample.png')
    # parser.add_argument('--control_normal', type=str, default='/mnt/syndata/Syn3DTxt_scene_2words_toydataset/normals_sample.png')
    # parser.add_argument('--control_mask', type=str, default='/mnt/syndata/Syn3DTxt_scene_2words_toydataset/normal_mask_sample.png')

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Run InfinityPilot inference with optional CAR control')
    add_arguments(parser)
    args = parser.parse_args()

    device = _as_device()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg_values = _parse_list_arg(args.cfg)
    tau_values = _parse_list_arg(args.tau)

    tokenizer, text_encoder = load_tokenizer(args.text_encoder_ckpt, device)
    vae = load_visual_tokenizer(args, device)
    infinity_pilot = build_infinity_pilot(args, vae, device)

    control_paths = _control_paths_from_args(args)
    scale_schedule = dynamic_resolution_h_w[args.h_div_w_template][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]
    control_tokens = _build_control_tokens(
        control_paths,
        vae,
        scale_schedule,
        device,
        bool(args.apply_spatial_patchify),
    )

    start = time.time()
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    image = gen_one_img_pilot(
        infinity_pilot,
        vae,
        tokenizer,
        text_encoder,
        args.prompt,
        scale_schedule=scale_schedule,
        cfg_list=cfg_values,
        tau_list=tau_values,
        device=device,
        negative_prompt=args.negative_prompt,
        g_seed=args.seed,
        cfg_sc=args.cfg_sc,
        cfg_exp_k=args.cfg_exp_k,
        cfg_insertion_layer=[args.cfg_insertion_layer],
        top_k=args.top_k,
        top_p=args.top_p,
        vae_type=args.vae_type,
        sampling_per_bits=args.sampling_per_bits,
        enable_positive_prompt=bool(args.enable_positive_prompt),
        control_tokens=control_tokens,
        amp_dtype=amp_dtype,
    )
    elapsed = time.time() - start

    os.makedirs(osp.dirname(osp.abspath(args.save_file)), exist_ok=True)
    cv2.imwrite(args.save_file, image.cpu().numpy())
    print(f'[done] saved image to {osp.abspath(args.save_file)}')
    print(f'[timing] total inference time: {elapsed:.2f}s')


if __name__ == '__main__':
    main()

"""
python tools/run_infinity_pilot.py --pn 1M --model_path /home/avlab/SceneTxtVAR/weights/mm_2b.pth --text_encoder_ckpt /home/avlab/SceneTxtVAR/weights/models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001 --vae_path /home/avlab/SceneTxtVAR/weights/infinity_vae_d32_reg.pth --prompt "a city street at dusk" --save_file ./outputs_pilot/sample.jpg --car_path /media/avlab/f09873b9-7c6a-4146-acdb-7db847b573201/output_a6000_bsqvae_parallel_toy/car_weights_ep0007_it008000/car_weights.pth
"""
