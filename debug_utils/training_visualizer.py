# Replacement visualizer that:
# - Uses the official token->codes->upsample->vae.decode flow (no manual decoder work)
# - Does NOT resize source images; composes them horizontally without changing original pixels
# - Provides robust shape checks and debug logs, plus safe fallbacks
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import traceback
from typing import Optional, List, Dict, Tuple
from infinity.utils import arg_util, misc, wandb_utils

# ----------------------------
# Helper: official-style decode
# ----------------------------
@torch.no_grad()
def decode_tokens_official(vae, tokens_per_scale: List[torch.Tensor], scale_schedule, vae_scale_schedule, apply_patchify: bool, label_type='bit_label', debug: bool = False):
    """
    Follow the official Infinity logic:
      tokens -> reshape -> (pixel_shuffle if patchify) -> indices_to_codes -> upsample -> summed_codes -> vae.decode
    Returns:
      img: tensor [B, C, H, W] (in [-1,1])
      summed_codes: tensor [B, d, 1, H, W] (before squeeze) for inspection
    """
    if not tokens_per_scale:
        raise ValueError("tokens_per_scale is empty")

    device = tokens_per_scale[0].device
    B = tokens_per_scale[0].shape[0]
    summed_codes = None
    last_codes = None

    for si, toks in enumerate(tokens_per_scale):
        pt, ph, pw = scale_schedule[si]
        toks = toks.to(device)
        D = toks.shape[-1]

        # reshape tokens -> [B, pt, ph, pw, D]
        if toks.shape[1] == pt * ph * pw:
            tokens_si = toks.reshape(B, pt, ph, pw, D)
        else:
            # try a tolerant reshape attempt then fail with helpful message
            try:
                tokens_si = toks.reshape(B, pt, ph, pw, D)
            except Exception as e:
                raise RuntimeError(f"Cannot reshape tokens at scale {si}: toks.shape={tuple(toks.shape)}, expected (B,{pt},{ph},{pw},D)") from e

        if debug:
            print(f"[decode_official] scale {si} tokens_si.shape={tuple(tokens_si.shape)} apply_patchify={apply_patchify}")

        # apply patchify handling consistent with official code
        if apply_patchify:
            if D % 4 != 0:
                raise RuntimeError(f"apply_spatial_patchify=True but token dim D={D} not divisible by 4")
            bpt = B * pt
            tokens_tmp = tokens_si.reshape(bpt, ph, pw, D).permute(0, 3, 1, 2)  # [B*pt, D, ph, pw]
            try:
                tokens_shuffled = torch.nn.functional.pixel_shuffle(tokens_tmp, 2)  # -> [B*pt, D/4, 2ph, 2pw]
            except Exception as e:
                raise RuntimeError(f"pixel_shuffle failed: {e}; tokens_tmp.shape={tokens_tmp.shape}")
            D2 = D // 4
            idx_Bld = tokens_shuffled.permute(0, 2, 3, 1).reshape(B, pt, ph * 2, pw * 2, D2)  # [B, pt, 2ph, 2pw, D2]
            idx_Bld = idx_Bld.unsqueeze(1)  # mimic official shape when needed
        else:
            # official code often passes tokens_si (maybe unsqueezed) into indices_to_codes
            idx_Bld = tokens_si.unsqueeze(1) if tokens_si.dim() == 4 else tokens_si

        # indices_to_codes: may return [B, d, H, W] or [B, d, 1, H, W]
        codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type=label_type)
        last_codes = codes
        if debug:
            print(f"[decode_official] scale {si} codes.shape (raw) = {tuple(codes.shape)} dtype={codes.dtype} device={codes.device}")

        # ensure 5D: [B, d, 1, H, W]
        if codes.dim() == 4:
            codes = codes.unsqueeze(2)
        if codes.dim() != 5:
            raise RuntimeError(f"Unexpected codes.dim()={codes.dim()} at scale {si}")

        # upsample codes to final VAE scale using quantizer preference (fallbacks handled)
        target_size = vae_scale_schedule[-1]
        if isinstance(target_size, int):
            tsize = (1, target_size, target_size)
        elif len(target_size) == 2:
            tsize = (1, target_size[0], target_size[1])
        else:
            tsize = tuple(int(x) for x in target_size)

        mode = getattr(vae.quantizer, 'z_interplote_up', None)
        tried = []
        upsampled = None
        for m in (mode, 'trilinear', 'nearest', 'bilinear'):
            if m is None:
                continue
            try:
                if m == 'trilinear':
                    upsampled = F.interpolate(codes, size=tsize, mode=m, align_corners=False)
                else:
                    upsampled = F.interpolate(codes, size=tsize, mode=m)
                if debug:
                    print(f"[decode_official] scale {si} upsample mode={m} result.shape={tuple(upsampled.shape)}")
                break
            except Exception as e:
                tried.append((m, str(e)))
                upsampled = None
        if upsampled is None:
            raise RuntimeError(f"All interpolation attempts failed for scale {si}. Tried: {tried}")

        summed_codes = upsampled if summed_codes is None else summed_codes + upsampled

    if summed_codes is None:
        raise RuntimeError("No summed_codes produced")

    # official: squeeze the time dim (dim -3) then call vae.decode(...)
    dec_in = summed_codes.squeeze(-3)  # [B, d, H, W]
    if debug:
        print(f"[decode_official] calling vae.decode with dec_in.shape={tuple(dec_in.shape)} dtype={dec_in.dtype} device={dec_in.device}")
    # prefer vae.decode if available
    if hasattr(vae, 'decode') and callable(getattr(vae, 'decode')):
        img = vae.decode(summed_codes.squeeze(-3))
    elif hasattr(vae, 'decoder') and callable(getattr(vae, 'decoder')):
        img = vae.decoder(dec_in)
    else:
        raise RuntimeError("VAE has no decode/decoder method")
    img = torch.clamp(img, -1, 1)
    return img, summed_codes, last_codes

# ----------------------------
# Helper: compose horizontal without resizing
# ----------------------------
def compose_horizontal_no_resize(imgs: List[torch.Tensor]) -> torch.Tensor:
    """
    imgs: list of tensors (C,H,W)
    Returns canvas tensor (C, max_H, sum_W) with imgs placed left-to-right, top-aligned, padded with 0 below as needed.
    """
    assert len(imgs) > 0
    device = imgs[0].device
    dtype = imgs[0].dtype
    # ensure same device/dtype
    imgs_fixed = []
    for im in imgs:
        im = im.to(device)
        if im.dtype != dtype:
            im = im.to(dtype)
        imgs_fixed.append(im)
    heights = [int(im.shape[1]) for im in imgs_fixed]
    widths = [int(im.shape[2]) for im in imgs_fixed]
    max_h = max(heights)
    total_w = sum(widths)
    canvas = torch.zeros(3, max_h, total_w, device=device, dtype=dtype)
    x = 0
    for im in imgs_fixed:
        c, h, w = im.shape
        canvas[:, 0:h, x:x+w] = im
        x += w
    return canvas

# ----------------------------
# Helper: normalize to 3 channels, DO NOT change H/W
# ----------------------------
def ensure_3ch_no_resize(img: torch.Tensor) -> torch.Tensor:
    """
    Accepts (H,W) or (C,H,W) or (1,H,W) and returns (3,H,W) without resizing.
    """
    if not isinstance(img, torch.Tensor):
        raise TypeError("ensure_3ch_no_resize expects a torch.Tensor")
    if img.dim() == 2:
        img = img.unsqueeze(0)
    if img.dim() == 3:
        c, h, w = img.shape
        if c == 1:
            return img.repeat(3, 1, 1)
        elif c >= 3:
            return img[:3]
    if img.dim() == 4 and img.shape[0] == 1:
        return ensure_3ch_no_resize(img.squeeze(0))
    raise ValueError(f"Unexpected img.dim()={img.dim()} in ensure_3ch_no_resize")

# ----------------------------
# Main visualizer (replacement)
# ----------------------------
@torch.no_grad()
def _generate_training_visualization(
    trainer_self,
    ep: int,
    it: int,
    g_it: int,
    inp_B3HW: torch.Tensor,
    raw_features_BdHW: Optional[torch.Tensor],
    condition_inputs: Optional[Dict[str, torch.Tensor]],
    gt_ms_idx_Bl: List[torch.Tensor],
    pred_ms_idx_Bl: List[torch.Tensor],
    scale_schedule,
    training_scales: int,
    training_seq_len: int,
    full_gt_ms_idx_Bl,
    full_scale_schedule,
    full_vae_scale_schedule,
):
    """
    Visualizer replacement:
      - Uses official decode flow via decode_tokens_official (calls vae.decode)
      - Does NOT resize images; composes originals horizontally
      - Logs to wandb or saves local files on failure
    """
    try:
        print(f'scale_schedule: {scale_schedule}')
        device = inp_B3HW.device
        if scale_schedule is None:
            raise ValueError("scale_schedule must be provided for visualization.")
        scale_schedule = list(scale_schedule)
        original_scale_schedule = list(scale_schedule)
        decode_scale_schedule = list(full_scale_schedule) if full_scale_schedule else list(original_scale_schedule)
        effective_scales = min(training_scales, len(scale_schedule), len(gt_ms_idx_Bl))
        if effective_scales == 0:
            return None
        scale_schedule = scale_schedule[:effective_scales]
        gt_ms_idx_Bl = gt_ms_idx_Bl[:effective_scales]

        vis_batch_size = min(1, inp_B3HW.shape[0])
        inp_vis = inp_B3HW[:vis_batch_size]
        if condition_inputs is not None:
            condition_vis = {k: v[:vis_batch_size] for k, v in condition_inputs.items() if v is not None}
            if len(condition_vis) == 0:
                condition_vis = None
        else:
            condition_vis = None

        raw_features_vis = None
        if raw_features_BdHW is not None:
            raw_features_vis = raw_features_BdHW[:vis_batch_size].clone()
            if raw_features_vis.dim() == 5 and raw_features_vis.shape[2] == 1:
                raw_features_vis = raw_features_vis.squeeze(2)

        gt_tokens_vis = [gt[:vis_batch_size] for gt in gt_ms_idx_Bl]
        full_gt_tokens_vis = None
        if full_gt_ms_idx_Bl is not None:
            full_gt_tokens_vis = [gt[:vis_batch_size] for gt in full_gt_ms_idx_Bl]
        pred_tokens_vis = [pred[:vis_batch_size] for pred in pred_ms_idx_Bl] if pred_ms_idx_Bl else None

        # determine vae schedules and patchify flag from trainer and args
        apply_patchify = getattr(trainer_self.bitwise_self_correction, 'apply_spatial_patchify', False)
        vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule] if apply_patchify else scale_schedule
        if full_vae_scale_schedule:
            decode_vae_scale_schedule = list(full_vae_scale_schedule)
        else:
            if apply_patchify:
                decode_vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in decode_scale_schedule]
            else:
                decode_vae_scale_schedule = decode_scale_schedule

        if full_gt_tokens_vis is not None and len(full_gt_tokens_vis) == len(decode_scale_schedule):
            gt_tokens_for_decode = full_gt_tokens_vis
            scale_schedule_for_decode = decode_scale_schedule
            vae_schedule_for_decode = decode_vae_scale_schedule
        else:
            gt_tokens_for_decode = gt_tokens_vis
            scale_schedule_for_decode = scale_schedule
            vae_schedule_for_decode = vae_scale_schedule

        # Decode GT tokens using official flow (safe)
        try:
            gt_images, gt_summed_codes, gt_last_codes = decode_tokens_official(
                trainer_self.vae_local,
                gt_tokens_for_decode,
                scale_schedule_for_decode,
                vae_schedule_for_decode,
                apply_patchify=apply_patchify,
                label_type='bit_label',
                debug=False
            )
        except Exception as e:
            print(f"[visualizer] decode_tokens_official failed for GT: {e}")
            traceback.print_exc()
            gt_images = None
            gt_summed_codes = None

        # Decode predicted tokens if available
        pred_images = None
        if pred_tokens_vis is not None:
            # build full pred list aligned to decode schedule (official expects same length)
            pred_tokens_for_decode = pred_tokens_vis
            scale_schedule_pred = scale_schedule
            vae_schedule_pred = vae_scale_schedule
            if decode_scale_schedule and len(pred_tokens_vis) <= len(decode_scale_schedule):
                pred_tokens_full = []
                for si in range(len(decode_scale_schedule)):
                    if si < len(pred_tokens_vis):
                        pred_tokens_full.append(pred_tokens_vis[si])
                    elif full_gt_tokens_vis is not None and si < len(full_gt_tokens_vis):
                        pred_tokens_full.append(full_gt_tokens_vis[si])
                if len(pred_tokens_full) == len(decode_scale_schedule):
                    pred_tokens_for_decode = pred_tokens_full
                    scale_schedule_pred = decode_scale_schedule
                    vae_schedule_pred = decode_vae_scale_schedule
            try:
                pred_images, pred_summed_codes, pred_last_codes = decode_tokens_official(
                    trainer_self.vae_local,
                    pred_tokens_for_decode,
                    scale_schedule_pred,
                    vae_schedule_pred,
                    apply_patchify=apply_patchify,
                    label_type='bit_label',
                    debug=False
                )
            except Exception as e:
                print(f"[visualizer] decode_tokens_official failed for pred: {e}")
                traceback.print_exc()
                pred_images = None

        # Decode raw (pre-quantized) VAE features to measure information loss.
        vae_reconstructed = None
        if raw_features_vis is not None:
            try:
                vae_reconstructed = trainer_self.vae_local.decode(raw_features_vis).clamp_(-1, 1)
            except Exception as e:
                print(f"[visualizer] VAE auto reconstruction failed: {e}")
                traceback.print_exc()
                vae_reconstructed = None

        # Compose visualization without resizing: compare original | condition(s) | gt | pred
        reconstruction_images = []
        for i in range(vis_batch_size):
            try:
                single_inp = inp_vis[i:i+1]
                single_condition = None
                if condition_vis:
                    single_condition = {k: v[i:i+1] for k, v in condition_vis.items()}

                comparison_images = []

                if vae_reconstructed is not None and vae_reconstructed.shape[0] > i:
                    auto_display = torch.clamp(vae_reconstructed[i], -1, 1).to(device)
                    comparison_images.append(ensure_3ch_no_resize(auto_display))

                if single_condition is not None:
                    if 'normal' in single_condition:
                        normal_img = torch.clamp(single_condition['normal'].squeeze(0), -1, 1).to(device)
                        comparison_images.append(ensure_3ch_no_resize(normal_img))
                    if 'mask' in single_condition:
                        mask_img = torch.clamp(single_condition['mask'].squeeze(0), -1, 1).to(device)
                        comparison_images.append(ensure_3ch_no_resize(mask_img))

                if gt_images is not None and gt_images.shape[0] > i:
                    comparison_images.append(ensure_3ch_no_resize(gt_images[i].to(device)))
                if pred_images is not None and pred_images.shape[0] > i:
                    comparison_images.append(ensure_3ch_no_resize(pred_images[i].to(device)))

                if not comparison_images:
                    h, w = single_inp.shape[-2:]
                    comparison_grid = torch.zeros(3, h, w, device=device, dtype=single_inp.dtype)
                else:
                    comparison_grid = compose_horizontal_no_resize(comparison_images)
                reconstruction_images.append(comparison_grid)

            except Exception:
                traceback.print_exc()
                # fallback: blank canvas with same size as input
                h, w = inp_vis.shape[-2:]
                num_images = max(1, 2 + (len(condition_vis) if condition_vis is not None else 0))
                placeholder = torch.zeros(3, h, w * max(1, num_images), device=device)
                reconstruction_images.append(placeholder)
                continue

        if len(reconstruction_images) > 0:
            final_grid = torch.stack(reconstruction_images, dim=0)  # [N, C, H, W_total]
            try:
                wandb_utils.log_image(f"training_reconstruction", final_grid, step=g_it)
            except Exception as e:
                print(f"Failed to log training visualization to wandb: {e}")
                try:
                    import os
                    save_dir = f"./training_visualizations"
                    os.makedirs(save_dir, exist_ok=True)
                    for idx, img in enumerate(final_grid):
                        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        cv2.imwrite(f"{save_dir}/train_ep{ep}_it{it}_g{g_it}_recon{idx}.jpg", img_np[:, :, ::-1])
                except Exception as e2:
                    print(f"Failed to save training visualization files: {e2}")
            finally:
                try:
                    del final_grid, reconstruction_images
                    if 'inp_vis' in locals():
                        del inp_vis
                    if 'condition_vis' in locals():
                        del condition_vis
                    if 'comparison_images' in locals():
                        del comparison_images
                    torch.cuda.empty_cache()
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")

    except Exception as e:
        print(f"Error in training visualization: {e}")
        traceback.print_exc()
