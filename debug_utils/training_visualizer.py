import torch
import torch.nn.functional as F
import numpy as np
import cv2
import traceback
from typing import Optional, List, Dict, Tuple
from infinity.utils import arg_util, misc, wandb_utils


@torch.no_grad()
def _generate_training_visualization(
    trainer_self,
    ep: int,
    it: int,
    g_it: int,
    inp_B3HW: torch.Tensor,
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
    """Generate visualization images during training using current training step results and log to wandb"""
    try:
        # Only generate visualizations occasionally to avoid overhead
        # if g_it % 100 == 0 and it != 0:  # Every 100 global iterations or first iteration
            # return
            
        # # print(f"Generating training visualization at epoch {ep}, iteration {it}, global_it {g_it}...")
        
        # Get device from input
        device = inp_B3HW.device
        effective_scales = min(training_scales, len(scale_schedule), len(gt_ms_idx_Bl))
        if effective_scales == 0:
            return None
        scale_schedule = scale_schedule[:effective_scales]
        gt_ms_idx_Bl = gt_ms_idx_Bl[:effective_scales]
        training_scales = effective_scales
        training_seq_len = np.array(scale_schedule).prod(axis=1).sum()
        
        # Limit to first sample to run a single inference pass
        vis_batch_size = min(1, inp_B3HW.shape[0])
        inp_vis = inp_B3HW[:vis_batch_size]
        if condition_inputs is not None:
            condition_vis = {k: v[:vis_batch_size] for k, v in condition_inputs.items() if v is not None}
            if len(condition_vis) == 0:
                condition_vis = None
        else:
            condition_vis = None
        
        gt_tokens_vis = [gt[:vis_batch_size] for gt in gt_ms_idx_Bl]
        seq_lengths = [gt.shape[1] for gt in gt_tokens_vis]
        full_gt_tokens_vis = (
            [gt[:vis_batch_size] for gt in full_gt_ms_idx_Bl]
            if full_gt_ms_idx_Bl is not None
            else gt_tokens_vis
        )

        pred_tokens_vis = None
        if pred_ms_idx_Bl:
            pred_tokens_vis = [pred[:vis_batch_size] for pred in pred_ms_idx_Bl]

        if trainer_self.bitwise_self_correction.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
            full_vae_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in full_scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
            full_vae_schedule = full_scale_schedule

        final_vae_scale = full_vae_scale_schedule[-1] if full_vae_scale_schedule else full_vae_schedule[-1]
        gt_images = _decode_tokens_to_image(
            trainer_self,
            full_gt_tokens_vis,
            full_scale_schedule,
            full_vae_schedule,
            final_vae_scale,
        )

        pred_images = None
        if pred_tokens_vis is not None:
            pred_tokens_full = []
            for si in range(len(full_scale_schedule)):
                if si < len(pred_tokens_vis):
                    pred_tokens_full.append(pred_tokens_vis[si])
                elif full_gt_tokens_vis is not None and si < len(full_gt_tokens_vis):
                    pred_tokens_full.append(full_gt_tokens_vis[si])
            if full_gt_tokens_vis is not None and len(pred_tokens_full) < len(full_scale_schedule):
                pred_tokens_full.extend(full_gt_tokens_vis[len(pred_tokens_full):len(full_scale_schedule)])
            if pred_tokens_full:
                pred_images = _decode_tokens_to_image(
                    trainer_self,
                    pred_tokens_full,
                    full_scale_schedule,
                    full_vae_schedule,
                    final_vae_scale,
                )

        reconstruction_images = []

        for i in range(vis_batch_size):
            try:
                single_inp = inp_vis[i:i+1]
                single_condition = None
                if condition_vis is not None:
                    single_condition = {k: v[i:i+1] for k, v in condition_vis.items()}

                orig_display = torch.clamp(single_inp.squeeze(0), -1, 1).to(device)
                comparison_images = [orig_display]

                if single_condition is not None:
                    if 'normal' in single_condition:
                        comparison_images.append(torch.clamp(single_condition['normal'].squeeze(0), -1, 1).to(device))
                    if 'mask' in single_condition:
                        mask_img = torch.clamp(single_condition['mask'].squeeze(0), -1, 1).to(device)
                        if mask_img.shape[0] == 1:
                            mask_img = mask_img.repeat(3, 1, 1)
                        comparison_images.append(mask_img)

                gt_display = torch.clamp(gt_images[i], -1, 1).to(device)
                comparison_images.append(gt_display)

                if pred_images is not None:
                    pred_display = torch.clamp(pred_images[i], -1, 1).to(device)
                    comparison_images.append(pred_display)

                target_h, target_w = orig_display.shape[-2:]
                resized_comparison = []
                for img in comparison_images:
                    if img.shape[-2:] != (target_h, target_w):
                        img_resized = F.interpolate(img.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0)
                        resized_comparison.append(img_resized)
                    else:
                        resized_comparison.append(img)

                comparison_grid = torch.cat(resized_comparison, dim=2)
                reconstruction_images.append(comparison_grid)

            except Exception:
                traceback.print_exc()
                h, w = inp_vis.shape[-2:]
                num_images = 3 + (len(condition_vis) if condition_vis is not None else 0)
                placeholder = torch.zeros(3, h, w * num_images, device=device)
                reconstruction_images.append(placeholder)
                continue
        
        if len(reconstruction_images) > 0:
            # Stack vertically for final display
            final_grid = torch.stack(reconstruction_images, dim=0)  # [N, C, H, W*num_images]
            
            # Log to wandb using existing utilities
            try:
                wandb_utils.log_image(f"training_reconstruction", final_grid, step=g_it)
                # print(f"✅ Logged training reconstruction to wandb at step {g_it}")
                
            except Exception as e:
                print(f"Failed to log training visualization to wandb: {e}")
                
                # Fallback: save as local files
                try:
                    import os
                    save_dir = f"./training_visualizations"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    for idx, img in enumerate(final_grid):
                        # Convert to numpy and save
                        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        cv2.imwrite(
                            f"{save_dir}/train_ep{ep}_it{it}_g{g_it}_recon{idx}.jpg", 
                            img_np[:, :, ::-1]  # RGB to BGR for cv2
                        )
                    # print(f"💾 Saved training reconstructions to {save_dir}")
                    
                except Exception as e2:
                    print(f"Failed to save training visualization files: {e2}")

            finally:
                try:
                    # 清理大型張量和中間結果
                    del final_grid, reconstruction_images
                    if 'inp_vis' in locals():
                        del inp_vis
                    if 'condition_vis' in locals():
                        del condition_vis
                    if 'comparison_images' in locals():
                        del comparison_images
                    if 'summed_codes' in locals():
                        del summed_codes
                    if 'reconstructed_img' in locals():
                        del reconstructed_img
                    
                    # 清理 CUDA 快取
                    torch.cuda.empty_cache()
                except Exception as cleanup_error:
                    print(f"Cleanup error: {cleanup_error}")

    except Exception as e:
        print(f"Error in training visualization: {e}")
        traceback.print_exc()


def _decode_tokens_to_image(
    trainer_self,
    tokens_per_scale: List[torch.Tensor],
    scale_schedule,
    vae_scale_schedule,
    final_vae_scale,
    reference_tokens: Optional[List[torch.Tensor]] = None,
):
    if not tokens_per_scale and not reference_tokens:
        return torch.zeros(0, device=trainer_self.vae_local.decoder.weight.device)

    vae = trainer_self.vae_local
    use_bit_label = getattr(trainer_self.gpt_wo_ddp, 'use_bit_label', False)
    apply_patchify = getattr(trainer_self.bitwise_self_correction, 'apply_spatial_patchify', False)

    if tokens_per_scale:
        B = tokens_per_scale[0].shape[0]
        device = tokens_per_scale[0].device
    else:
        B = reference_tokens[0].shape[0]
        device = reference_tokens[0].device

    prepared_tokens: List[torch.Tensor] = []
    for si, _ in enumerate(scale_schedule):
        if si < len(tokens_per_scale):
            prepared_tokens.append(tokens_per_scale[si])
        else:
            if reference_tokens is None or si >= len(reference_tokens):
                raise ValueError(f"Missing tokens for scale {si} and no reference provided")
            prepared_tokens.append(torch.zeros_like(reference_tokens[si]))

    summed_codes = None
    for si, tokens in enumerate(prepared_tokens):
        pt, ph, pw = scale_schedule[si]
        if use_bit_label:
            tokens_si = tokens.reshape(B, pt, ph, pw, -1)
            if apply_patchify:
                d = tokens_si.shape[-1]
                tokens_si = tokens_si.reshape(B * pt, ph, pw, d).permute(0, 3, 1, 2)
                tokens_si = torch.nn.functional.pixel_shuffle(tokens_si, 2)
                tokens_si = tokens_si.permute(0, 2, 3, 1).reshape(B, pt, ph * 2, pw * 2, d // 4)
            if tokens_si.dim() == 4:
                tokens_si = tokens_si.unsqueeze(1)
            codes = vae.quantizer.lfq.indices_to_codes(tokens_si, label_type='bit_label')
        else:
            raise NotImplementedError("Visualizer decode only implemented for bit-label mode")

        if codes.dim() == 4:
            codes = codes.unsqueeze(2)
        if codes.dim() != 5:
            raise ValueError(f"Unexpected code tensor rank {codes.dim()} while decoding visualization")
        target_size = final_vae_scale
        if isinstance(target_size, int):
            target_size = (1, target_size, target_size)
        elif len(target_size) == 2:
            target_size = (1, *target_size)
        target_size = tuple(int(x) for x in target_size)
        upsampled = F.interpolate(codes, size=target_size, mode=vae.quantizer.z_interplote_up)
        if summed_codes is None:
            summed_codes = upsampled
        else:
            summed_codes = summed_codes + upsampled

    img = vae.decoder(summed_codes.squeeze(-3))
    return torch.clamp(img, -1, 1)


def _visualize_tokens_as_image(gt_tokens_list: List[torch.Tensor], scale_schedule, device) -> torch.Tensor:
    """Create a simple visualization of token values as an image when VAE decode fails"""
    try:
        # Get the largest scale tokens for visualization
        if len(gt_tokens_list) == 0:
            return torch.zeros(3, 64, 64, device=device)
            
        largest_tokens = gt_tokens_list[-1] if len(gt_tokens_list) > 1 else gt_tokens_list[0]
        largest_tokens = largest_tokens.to(device)
        
        # print(f"Debug: Token visualization input shape={largest_tokens.shape}, dtype={largest_tokens.dtype}")
        
        # Handle different token formats
        if largest_tokens.dim() == 3:  # [B, L, D] - multi-dimensional tokens
            B, L, D = largest_tokens.shape
            # Convert to float before taking mean to avoid dtype issues
            if D > 1:
                largest_tokens = largest_tokens.float().mean(dim=-1)  # [B, L]
            else:
                largest_tokens = largest_tokens.squeeze(-1).float()  # [B, L]
        elif largest_tokens.dim() == 2:  # [B, L] - already flattened
            B, L = largest_tokens.shape
            largest_tokens = largest_tokens.float()  # Convert to float
        else:
            # Unexpected format, create placeholder
            # print(f"Debug: Unexpected token dimension {largest_tokens.dim()}, creating placeholder")
            return torch.zeros(3, 64, 64, device=device)
        
        # Flatten the tokens properly
        largest_tokens = largest_tokens.view(B, -1)  # Ensure [B, L] format
        L = largest_tokens.shape[1]
        
        # print(f"Debug: Processed tokens shape=[{B}, {L}], dtype={largest_tokens.dtype}")
        
        # Try to reshape tokens into a square-ish grid
        grid_size = int(np.ceil(np.sqrt(L)))
        
        # Pad tokens if needed
        tokens_padded = torch.zeros(B, grid_size * grid_size, device=device, dtype=largest_tokens.dtype)
        tokens_padded[:, :L] = largest_tokens
        
        # Reshape to image-like format
        token_img = tokens_padded.view(B, grid_size, grid_size)
        
        # Normalize to [0, 1] range
        token_min = token_img.min()
        token_max = token_img.max()
        if token_max > token_min:
            token_img = (token_img - token_min) / (token_max - token_min)
        else:
            token_img = torch.zeros_like(token_img)
        
        # Convert to 3-channel image by repeating grayscale
        token_img_3ch = token_img.unsqueeze(1).repeat(1, 3, 1, 1)  # [B, 3, H, W]
        
        # Resize to a reasonable size if too small
        if grid_size < 64:
            token_img_3ch = torch.nn.functional.interpolate(
                token_img_3ch, size=(64, 64), mode='nearest'
            )
        elif grid_size > 256:
            # If too large, resize down to avoid memory issues
            token_img_3ch = torch.nn.functional.interpolate(
                token_img_3ch, size=(256, 256), mode='nearest'
            )
        
        # print(f"Debug: Token visualization output shape={token_img_3ch.shape}")
        return token_img_3ch.squeeze(0)  # Remove batch dimension, return [3, H, W]
        
    except Exception as e:
        print(f"Token visualization failed: {e}")
        traceback.print_exc()
        # Return a simple placeholder
        return torch.zeros(3, 64, 64, device=device)
