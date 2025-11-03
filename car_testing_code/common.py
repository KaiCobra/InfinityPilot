"""
Shared utilities for CAR / Infinity analysis scripts.

These helpers mirror the `train_pilot.sh` configuration:
  - Infinity 2B base model (`--model=2bc8`)
  - Shared-aln enabled, rope2d on every SA layer, pad=128
  - Bitwise VAE (bsq) with 32-bit code dimension
  - CAR depth = 4 with randomly initialised weights

The functions below build the full preprocessing pipeline used during training:
  1. Load pretrained Infinity weights and VAE.
  2. Encode RGB images into BSQ latent tokens with BitwiseSelfCorrection.
  3. Prepare conditioning embeddings (text KV, SOS tokens, attention masks).
  4. Reproduce the CAR ↔ VAR cooperation to log tensor statistics.
"""

from __future__ import annotations

import math
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_pythonpath() -> None:
    """
    Ensure the repository packages are importable and provide a safe flash-attn stub.
    """
    _ensure_flash_attn_stub()

    root = _project_root()
    for candidate in (root, root / "CAR", root / "infinity"):
        if str(candidate) not in sys.path:
            sys.path.append(str(candidate))


def _ensure_flash_attn_stub() -> None:
    """
    Provide CPU fallbacks for flash-attn so Infinity imports succeed even if the
    compiled extension is unavailable.
    """
    if "flash_attn" in sys.modules:
        return
    try:
        import flash_attn  # type: ignore  # noqa: F401
        return
    except ImportError:
        pass

    module = types.ModuleType("flash_attn")

    def _slow_flash_attn(q, k, v, dropout_p=0.0, softmax_scale=None, **kwargs):
        if softmax_scale is None or softmax_scale == 0:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)

    def _slow_flash_attn_varlen(q, kv, cu_seqlens_q, cu_seqlens_k,
                                max_seqlen_q, max_seqlen_k,
                                dropout_p=0.0, softmax_scale=None, **kwargs):
        if softmax_scale is None or softmax_scale == 0:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        outputs = []
        batch = cu_seqlens_q.shape[0] - 1
        for b in range(batch):
            qs = slice(cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item())
            ks = slice(cu_seqlens_k[b].item(), cu_seqlens_k[b + 1].item())
            q_slice = q[qs]            # (Lq, H, d)
            kv_slice = kv[ks]          # (Lk, 2, H, d)
            k_slice, v_slice = kv_slice[:, 0], kv_slice[:, 1]

            q_heads = q_slice.transpose(0, 1)  # (H, Lq, d)
            k_heads = k_slice.transpose(0, 1)  # (H, Lk, d)
            v_heads = v_slice.transpose(0, 1)  # (H, Lk, d)

            scores = torch.matmul(q_heads, k_heads.transpose(1, 2)) * softmax_scale
            weights = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(weights, v_heads)  # (H, Lq, d)
            outputs.append(ctx.transpose(0, 1))   # (Lq, H, d)
        return torch.cat(outputs, dim=0)

    module.flash_attn_func = _slow_flash_attn  # type: ignore[attr-defined]
    module.flash_attn_varlen_kvpacked_func = _slow_flash_attn_varlen  # type: ignore[attr-defined]
    sys.modules["flash_attn"] = module


# ---------------------------------------------------------------------------
# Basic tensor summaries
# ---------------------------------------------------------------------------

def summarize_tensor(t: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        flat = t.detach().float().cpu()
        return {
            "mean": float(flat.mean().item()),
            "std": float(flat.std().item()),
            "min": float(flat.min().item()),
            "max": float(flat.max().item()),
        }


def format_stats(name: str, tensor: torch.Tensor) -> str:
    stats = summarize_tensor(tensor)
    return (
        f"{name:<40} "
        f"mean={stats['mean']:+.6f}  "
        f"std={stats['std']:+.6f}  "
        f"min={stats['min']:+.6f}  "
        f"max={stats['max']:+.6f}"
    )


# ---------------------------------------------------------------------------
# Configuration mirroring train_pilot.sh
# ---------------------------------------------------------------------------

DEFAULT_VAE_CKPT = _project_root() / "weights" / "infinity_vae_d32_reg.pth"
DEFAULT_INFINITY_CKPT = Path("/mnt/syndata/VAR_ckpt/infinity_2b_reg.pth")
RAW_SCALE_SCHEDULE = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


class _AnalysisArgs:
    pass


def create_default_args():
    args = _AnalysisArgs()
    args.pn = '1M'
    args.scale_schedule = RAW_SCALE_SCHEDULE
    args.vae_ckpt = str(DEFAULT_VAE_CKPT)
    args.vae_type = 32
    args.Ct5 = 2048
    args.tlen = 512
    args.saln = True
    args.cos = True
    args.use_bit_label = 1
    args.add_lvl_embeding_only_first_block = 1
    args.rope2d_each_sa_layer = 1
    args.rope2d_normalized_by_hw = 2
    args.always_training_scales = 7
    args.use_flex_attn = False
    args.pad_to_multiplier = 128
    args.batch_size = 4
    args.enable_checkpointing = 'full-block'
    args.noise_apply_strength = 0.3
    args.noise_apply_layers = 13
    args.noise_apply_requant = 1
    args.apply_spatial_patchify = 0
    args.car_depth = 4
    args.car_condition_channels = 3
    args.cfg = 0.0
    args.rand_uncond = False
    args.drop = 0.0
    args.dp = 0.0
    args.norm_eps = 1e-6
    args.train_h_div_w_list = [1.0]
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.rush_resume = str(DEFAULT_INFINITY_CKPT)
    args.debug_bsc = 0
    return args


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_infinity_state_dict(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict):
        if 'trainer' in ckpt and isinstance(ckpt['trainer'], dict) and 'gpt_fsdp' in ckpt['trainer']:
            return ckpt['trainer']['gpt_fsdp']
        if 'gpt_fsdp' in ckpt:
            return ckpt['gpt_fsdp']
    return ckpt


def _detect_model_dims(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    embed_dim = state_dict['pos_start'].shape[-1]
    if embed_dim == 2048:
        return 2048, 32, 16  # 2B
    if embed_dim == 4608:
        return 4608, 58, 36  # 20B
    if embed_dim == 1024:
        return 1024, 16, 16  # 1B-style custom
    raise ValueError(f"Unsupported embed_dim {embed_dim} from checkpoint.")


def _load_vae(args, device: torch.device):
    configure_pythonpath()
    from infinity.models.bsq_vae.vae import vae_model  # type: ignore

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
        args.vae_ckpt,
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


def build_infinity_pilot(args, device: torch.device):
    configure_pythonpath()
    from infinity.models.infinity_pilot import InfinityPilot  # type: ignore
    from infinity.models.bitwise_self_correction import BitwiseSelfCorrection  # type: ignore

    vae = _load_vae(args, device)
    infinity_state = _load_infinity_state_dict(args.rush_resume)
    embed_dim, depth, num_heads = _detect_model_dims(infinity_state)

    model_kwargs = dict(
        vae_local=vae,
        text_channels=args.Ct5,
        text_maxlen=args.tlen,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        drop_rate=args.drop,
        drop_path_rate=args.dp,
        norm_eps=args.norm_eps,
        cond_drop_rate=args.cfg,
        rand_uncond=args.rand_uncond,
        raw_scale_schedule=args.scale_schedule,
        add_lvl_embeding_only_first_block=args.add_lvl_embeding_only_first_block,
        use_bit_label=args.use_bit_label,
        rope2d_each_sa_layer=args.rope2d_each_sa_layer,
        rope2d_normalized_by_hw=args.rope2d_normalized_by_hw,
        pn=args.pn,
        train_h_div_w_list=args.train_h_div_w_list,
        video_frames=1,
        always_training_scales=args.always_training_scales,
        apply_spatial_patchify=args.apply_spatial_patchify,
        shared_aln=bool(args.saln),
        head_aln=True,
        cross_attn_layer_scale=getattr(args, 'ca_gamma', -1.0),
        nm0=getattr(args, 'nm0', False),
        tau=getattr(args, 'tau', 1.0),
        cos_attn=bool(args.cos),
        swiglu=getattr(args, 'swi', False),
        head_depth=getattr(args, 'dec', 1),
        top_p=getattr(args, 'tp', 0.0),
        top_k=getattr(args, 'tk', 0.0),
        rms_norm=getattr(args, 'rms', False),
        customized_flash_attn=getattr(args, 'flash', False),
        fused_mlp=getattr(args, 'fuse', False),
        fused_norm=getattr(args, 'fused_norm', False),
        checkpointing=args.enable_checkpointing,
        pad_to_multiplier=args.pad_to_multiplier,
        use_flex_attn=args.use_flex_attn,
        batch_size=args.batch_size,
        car_depth=args.car_depth,
        car_condition_channels=args.car_condition_channels,
        block_chunks=getattr(args, 'num_block_chunks', 1),
    )

    model = InfinityPilot(
        infinity_base_model=infinity_state,
        init_car_modules=True,
        freeze_infinity=True,
        **model_kwargs,
    ).to(device)
    model.cond_drop_rate = 0.0  # deterministic for analysis
    model.eval()

    bsc = BitwiseSelfCorrection(vae, args)
    return model, vae, bsc


# ---------------------------------------------------------------------------
# VAE → token pipeline
# ---------------------------------------------------------------------------

def get_scale_schedule(args, ratio: float = 1.0) -> List[Tuple[int, int, int]]:
    from infinity.utils.dynamic_resolution import dynamic_resolution_h_w  # type: ignore

    schedule = dynamic_resolution_h_w[ratio][args.pn]['scales']
    normalized = [(1, h, w) for (_, h, w) in schedule]  # T dimension collapses to 1 for images
    return normalized[:args.always_training_scales]


@torch.no_grad()
def bsq_encode_tokens(
    vae,
    images: torch.Tensor,
    schedule: Sequence[Tuple[int, int, int]],
    args,
    device: torch.device,
):
    """
    Convert images into per-scale BSQ latent tokens following BitwiseSelfCorrection.
    Returns:
        tokens_concat: B x (sum_{i>0} L_i) x d_vae
        input_tokens_per_scale: list length len(schedule), tokens fed to Infinity at each scale
        quantized_tokens_per_scale: list of quantized residual tokens (useful for stage-0 control)
    """
    raw_features, _, _ = vae.encode_for_raw_features(images, scale_schedule=schedule)
    if raw_features.dim() == 4:
        codes_out = raw_features.unsqueeze(2)
    else:
        codes_out = raw_features

    B = codes_out.shape[0]
    total_scales = len(schedule)
    cum_var_input = torch.zeros_like(codes_out)
    input_tokens: List[Optional[torch.Tensor]] = [None] * total_scales
    quant_tokens: List[Optional[torch.Tensor]] = [None] * total_scales
    tokens_list: List[torch.Tensor] = []

    noise_schedule = torch.linspace(0.99, 0.1, total_scales, device=device)

    for si, target in enumerate(schedule):
        residual = codes_out - cum_var_input
        if si != total_scales - 1:
            residual = F.interpolate(residual, size=target, mode=vae.quantizer.z_interplote_down).contiguous()

        quantized, _, bit_indices, _ = vae.quantizer.lfq(residual)

        if si < args.noise_apply_layers and args.noise_apply_strength > 0:
            strength = noise_schedule[si] * args.noise_apply_strength
            if strength > 0:
                mask = torch.rand_like(bit_indices, dtype=torch.float32, device=device) < strength
                flipped = bit_indices.clone()
                flipped[mask] = 1 - flipped[mask]
                if args.noise_apply_requant:
                    quantized = vae.quantizer.lfq.indices_to_codes(flipped, label_type='bit_label')

        quant_spatial = quantized.squeeze(2)
        quant_tokens_si = quant_spatial.view(B, quant_spatial.shape[1], -1).transpose(1, 2)
        quant_tokens[si] = quant_tokens_si

        cum_var_input = cum_var_input + F.interpolate(
            quantized, size=schedule[-1], mode=vae.quantizer.z_interplote_up
        ).contiguous()

        if si < total_scales - 1:
            next_input = F.interpolate(
                cum_var_input, size=schedule[si + 1], mode=vae.quantizer.z_interplote_up
            ).contiguous()
            if args.apply_spatial_patchify:
                next_input = torch.nn.functional.pixel_unshuffle(next_input.squeeze(2), 2).unsqueeze(2)
            next_spatial = next_input.squeeze(2)
            next_tokens = next_spatial.view(B, next_spatial.shape[1], -1).transpose(1, 2)
            input_tokens[si + 1] = next_tokens
            tokens_list.append(next_tokens)

    tokens_concat = torch.cat(tokens_list, dim=1) if tokens_list else images.new_zeros(B, 0, vae.codebook_dim)
    return tokens_concat, input_tokens, quant_tokens


# ---------------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------------

def build_text_condition(
    batch_size: int,
    channel_dim: int,
    max_len: int,
    device: torch.device,
    seed: int = 0,
) -> Tuple[torch.Tensor, List[int], torch.Tensor, int]:
    torch.manual_seed(seed)
    lengths = [int(torch.randint(low=max_len // 4, high=max_len // 2, size=(1,)).item()) for _ in range(batch_size)]
    total = sum(lengths)
    kv_compact = torch.randn(total, channel_dim, device=device) / math.sqrt(channel_dim)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    offset = 0
    for i, length in enumerate(lengths, start=1):
        offset += length
        cu_seqlens[i] = offset
    max_seqlen = max(lengths)
    return kv_compact, lengths, cu_seqlens, max_seqlen


def prepare_conditioning(
    model,
    kv_tuple,
    x_tokens_wo_prefix: torch.Tensor,
    scale_schedule: Sequence[Tuple[int, int, int]],
    need_drop: bool = False,
) -> Dict[str, torch.Tensor]:
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = kv_tuple
    kv_compact = kv_compact.clone()
    if need_drop and model.cond_drop_rate > 0:
        total = 0
        for length in lens:
            if torch.rand(1).item() < model.cond_drop_rate:
                kv_compact[total:total + length] = model.cfg_uncond[:length]
            total += length

    kv_compact = model.text_norm(kv_compact).contiguous()
    cond_BD = model.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()
    kv_proj = model.text_proj_for_ca(kv_compact).contiguous()
    ca_kv = (kv_proj, cu_seqlens_k, max_seqlen_k)
    cond_BD_or_gss = model.shared_ada_lin(cond_BD).contiguous()

    B = x_tokens_wo_prefix.shape[0]
    sos = cond_BD.unsqueeze(1).expand(B, 1, -1) + model.pos_start.expand(B, 1, -1)
    word_tokens = model.word_embed(model.norm0_ve(x_tokens_wo_prefix))
    x_BLC = torch.cat((sos, word_tokens), dim=1)

    l_end = x_BLC.shape[1]
    need_to_pad = (l_end + model.pad_to_multiplier - 1) // model.pad_to_multiplier * model.pad_to_multiplier - l_end

    d = torch.cat(
        [torch.full((int(np.prod(stage)),), idx, device=x_BLC.device, dtype=torch.float32)
         for idx, stage in enumerate(scale_schedule)],
        dim=0,
    ).view(1, l_end, 1)
    attn_bias = torch.where(d >= d.transpose(1, 2), 0.0, -torch.inf).reshape(1, 1, l_end, l_end)
    if need_to_pad:
        attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
        attn_bias[0, 0, l_end:, 0] = 0
        x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))

    return {
        "cond_BD": cond_BD,
        "cond_BD_or_gss": cond_BD_or_gss,
        "ca_kv": ca_kv,
        "sos": sos,
        "x_BLC": x_BLC,
        "attn_bias": attn_bias.to(x_BLC.dtype),
        "need_to_pad": need_to_pad,
    }


# ---------------------------------------------------------------------------
# CAR ↔ VAR cooperation (manual execution of forward)
# ---------------------------------------------------------------------------

def run_car_var_pipeline(
    model,
    conditioning: Dict[str, torch.Tensor],
    x_tokens_wo_prefix: torch.Tensor,
    scale_schedule: Sequence[Tuple[int, int, int]],
    scenario_name: str,
    control_mode: Optional[str] = None,
    control_tensors: Optional[Sequence[torch.Tensor]] = None,
    control_input_tokens: Optional[Sequence[Optional[torch.Tensor]]] = None,
    control_quant_tokens: Optional[Sequence[Optional[torch.Tensor]]] = None,
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    sos = conditioning["sos"]
    x_BLC = conditioning["x_BLC"]
    cond_BD = conditioning["cond_BD"]
    cond_BD_or_gss = conditioning["cond_BD_or_gss"]
    attn_bias = conditioning["attn_bias"]
    ca_kv = conditioning["ca_kv"]
    need_to_pad = conditioning["need_to_pad"]
    lvl_pos = conditioning["lvl_pos"]

    B = x_BLC.shape[0]
    pointer = 1
    car_segments: List[torch.Tensor] = []
    control_residual_f: List[torch.Tensor] = []

    shared_control_spatial: List[Optional[torch.Tensor]] = []
    if control_mode == "shared_vae":
        assert control_input_tokens is not None and control_quant_tokens is not None
        for si, stage in enumerate(scale_schedule):
            ph, pw = stage[1], stage[2]
            if si == 0:
                tokens = control_quant_tokens[0]
            else:
                tokens = control_input_tokens[si]
            if tokens is None:
                shared_control_spatial.append(None)
                continue
            emb = model.word_embed(model.norm0_ve(tokens))
            spatial = emb.transpose(1, 2).reshape(B, model.C, ph, pw)
            shared_control_spatial.append(spatial)

    if control_mode == "conv":
        assert control_tensors is not None and len(control_tensors) == len(scale_schedule)

    for si, stage in enumerate(scale_schedule):
        seq_len = int(np.prod(stage))
        ph, pw = stage[1], stage[2]
        if si == 0:
            var_x = sos.transpose(1, 2).reshape(B, model.C, ph, pw)
        else:
            tokens = x_BLC[:, pointer:pointer + seq_len]
            pointer += seq_len
            var_x = tokens.transpose(1, 2).reshape(B, model.C, ph, pw)

        var_after_conv = model.car_var_conv(var_x)
        stats[f"{scenario_name}/stage{si}_var_after_conv"] = summarize_tensor(var_after_conv)

        if control_mode is None:
            continue

        if control_mode == "conv":
            control_tensor = control_tensors[si].to(var_after_conv.device)
            control_f = model.car_control_convs(control_tensor)
            if control_f.shape[-2:] != (ph, pw):
                control_f = F.interpolate(control_f, size=(ph, pw), mode='bilinear', align_corners=False)
            control_tokens = control_f.flatten(2).transpose(1, 2)
            control_tokens = model.car_control_norm(control_tokens)
            control_f = control_tokens.transpose(1, 2).reshape(B, model.C, ph, pw)
        elif control_mode == "shared_vae":
            control_spatial = shared_control_spatial[si]
            if control_spatial is None:
                control_f = torch.zeros_like(var_after_conv)
            else:
                control_tokens = control_spatial.flatten(2).transpose(1, 2)
                control_tokens = model.car_control_norm(control_tokens)
                control_f = control_tokens.transpose(1, 2).reshape(B, model.C, ph, pw)
        else:
            raise ValueError(f"Unknown control_mode {control_mode}")

        stats[f"{scenario_name}/stage{si}_control_feat"] = summarize_tensor(control_f)
        summed = var_after_conv + control_f
        stats[f"{scenario_name}/stage{si}_combined"] = summarize_tensor(summed)

        combined_tokens = summed.view(B, model.C, -1).transpose(1, 2).contiguous()
        car_segments.append(combined_tokens)

    if control_mode is not None and car_segments:
        car_input = torch.cat(car_segments, dim=1)
        if need_to_pad:
            car_input = F.pad(car_input, (0, 0, 0, need_to_pad))
        car_state = model.add_lvl_embeding_for_x_BLC(car_input.clone(), scale_schedule, need_to_pad)
        stats[f"{scenario_name}/car_input_post_pos"] = summarize_tensor(car_state)

        for cb_idx, cb in enumerate(model.car_blocks):
            car_state = cb(
                x=car_state,
                cond_BD=cond_BD_or_gss,
                ca_kv=ca_kv,
                attn_bias_or_two_vector=attn_bias,
                attn_fn=None,
                scale_schedule=scale_schedule,
                rope2d_freqs_grid=model.rope2d_freqs_grid,
            )
            control_residual_f.append(car_state)
            stats[f"{scenario_name}/car_block{cb_idx}_out"] = summarize_tensor(car_state)

    x_with_pos = model.add_lvl_embeding_for_x_BLC(x_BLC.clone(), scale_schedule, need_to_pad)
    stats[f"{scenario_name}/var_input_post_pos"] = summarize_tensor(x_with_pos)

    half = len(model.blocks) - model.car_depth
    residual_stack = list(control_residual_f)

    x_state = x_with_pos
    for idx, block in enumerate(model.blocks):
        if control_mode is not None and idx >= half and residual_stack:
            skip_idx = idx - half
            con_f = residual_stack.pop()
            base_norm = model.car_skip_base_norm[skip_idx](x_state)
            ctrl_norm = model.car_skip_ctrl_norm[skip_idx](con_f)
            cat = torch.cat([base_norm, ctrl_norm], dim=-1)
            delta = model.car_skip_linear[skip_idx](cat)
            scale = model.car_skip_scale[skip_idx].view(1, 1, 1)
            x_state = x_state + scale * delta
            stats[f"{scenario_name}/skip{skip_idx}_cat"] = summarize_tensor(cat)
            stats[f"{scenario_name}/skip{skip_idx}_post_linear"] = summarize_tensor(x_state)

        x_state = block(
            x=x_state,
            cond_BD=cond_BD_or_gss,
            ca_kv=ca_kv,
            attn_bias_or_two_vector=attn_bias,
            attn_fn=None,
            scale_schedule=scale_schedule,
            rope2d_freqs_grid=model.rope2d_freqs_grid,
        )
        stats[f"{scenario_name}/var_block{idx}_out"] = summarize_tensor(x_state)

    logits = model.get_logits(x_state[:, : x_tokens_wo_prefix.shape[1] + 1], cond_BD)
    stats[f"{scenario_name}/final_logits"] = summarize_tensor(logits)
    return stats


# ---------------------------------------------------------------------------
# Utility for random image/control generation
# ---------------------------------------------------------------------------

def generate_random_images(
    schedule: Sequence[Tuple[int, int, int]],
    batch_size: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    torch.manual_seed(seed)
    ph, pw = schedule[-1][1], schedule[-1][2]
    height, width = ph * 16, pw * 16
    return torch.rand(batch_size, 3, height, width, device=device) * 2 - 1
