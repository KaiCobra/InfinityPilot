import torch
import numpy as np
import cv2
import traceback
from typing import Optional, List
from infinity.utils import arg_util, misc, wandb_utils


@torch.no_grad()
def _generate_training_visualization(trainer_self, ep: int, it: int, g_it: int, 
                                   inp_B3HW: torch.Tensor, condition_B3HW: Optional[torch.Tensor], 
                                   gt_ms_idx_Bl: List[torch.Tensor], text_cond_tuple, 
                                   scale_schedule, training_scales: int, training_seq_len: int):
    """Generate visualization images during training using current training step results and log to wandb"""
    try:
        # Only generate visualizations occasionally to avoid overhead
        if g_it % 2000 != 0 and it != 0:  # Every 1000 global iterations or first iteration
            return
            
        # # print(f"Generating training visualization at epoch {ep}, iteration {it}, global_it {g_it}...")
        
        # Get device from input
        device = inp_B3HW.device
        
        # Limit to first 2 samples to avoid memory issues
        vis_batch_size = min(2, inp_B3HW.shape[0])
        inp_vis = inp_B3HW[:vis_batch_size]
        condition_vis = condition_B3HW[:vis_batch_size] if condition_B3HW is not None else None
        
        # Use current training step's ground truth tokens to reconstruct images
        reconstruction_images = []
        
        for i in range(vis_batch_size):
            try:
                # Get single sample
                single_inp = inp_vis[i:i+1]  # [1, 3, H, W] - original input
                single_condition = condition_vis[i:i+1] if condition_vis is not None else None
                
                # Get ground truth tokens for this sample
                single_gt_tokens = [gt_tokens[i:i+1] for gt_tokens in gt_ms_idx_Bl]
                
                # 參考 infinity_pilot.py 的 autoregressive_infer_cfg 方法進行重建
                try:
                    if hasattr(trainer_self, 'vae_local') and trainer_self.vae_local is not None:
                        vae = trainer_self.vae_local
                        
                        # 使用與 autoregressive_infer_cfg 相同的重建邏輯
                        if hasattr(trainer_self.gpt_wo_ddp, 'use_bit_label') and trainer_self.gpt_wo_ddp.use_bit_label:
                            # BSQ-VAE bit label 處理方式
                            # print(f"Debug: Using BSQ-VAE reconstruction with bit labels")
                            
                            # 將 gt tokens 轉換為正確格式
                            summed_codes = None
                            final_scale_h, final_scale_w = scale_schedule[training_scales-1][1:]  # 最終尺度
                            
                            # 處理每個尺度的 tokens
                            for si, (gt_tokens_scale, (pt, ph, pw)) in enumerate(zip(single_gt_tokens[:training_scales], scale_schedule[:training_scales])):
                                try:
                                    # 重新整形 tokens 以匹配 BSQ-VAE 的期望格式
                                    if gt_tokens_scale.dim() == 2:  # [1, L]
                                        seq_len = pt * ph * pw
                                        if gt_tokens_scale.shape[1] == seq_len:
                                            # 沒有 bit dimension，直接使用
                                            gt_tokens_reshaped = gt_tokens_scale.reshape(1, ph, pw, -1)
                                        else:
                                            # 有 bit dimension，需要重新整形
                                            bits_per_token = gt_tokens_scale.shape[1] // seq_len
                                            gt_tokens_reshaped = gt_tokens_scale.reshape(1, seq_len, bits_per_token)
                                            gt_tokens_reshaped = gt_tokens_reshaped.reshape(1, ph, pw, bits_per_token)
                                    elif gt_tokens_scale.dim() == 3:  # [1, L, D]
                                        gt_tokens_reshaped = gt_tokens_scale.reshape(1, ph, pw, -1)
                                    else:
                                        # print(f"Unexpected token dimension: {gt_tokens_scale.shape}")
                                        continue
                                    
                                    # 添加時間維度 [B, t, h, w, d] -> [1, 1, h, w, d]
                                    gt_tokens_with_time = gt_tokens_reshaped.unsqueeze(1)
                                    
                                    # print(f"Debug: Scale {si}, gt_tokens shape: {gt_tokens_with_time.shape}")
                                    
                                    # 使用 VAE 的 quantizer 將 indices 轉換為 codes
                                    if hasattr(vae, 'quantizer') and hasattr(vae.quantizer, 'lfq'):
                                        codes = vae.quantizer.lfq.indices_to_codes(gt_tokens_with_time, label_type='bit_label')
                                        # print(f"Debug: Scale {si}, codes shape: {codes.shape}")
                                        
                                        # 修復插值操作：正確處理 5D tensor
                                        if si != training_scales - 1:
                                            # 對於非最後一個尺度，插值到最終尺度並累加
                                            # codes 格式: [B, C, T, H, W] -> [1, 32, 1, h, w]
                                            # 需要先移除時間維度進行插值，然後再加回
                                            codes_2d = codes.squeeze(2)  # [1, 32, h, w] - 移除時間維度
                                            upsampled_2d = torch.nn.functional.interpolate(
                                                codes_2d, size=(final_scale_h, final_scale_w), mode='nearest'
                                            )  # [1, 32, final_h, final_w]
                                            upsampled = upsampled_2d.unsqueeze(2)  # [1, 32, 1, final_h, final_w] - 重新加入時間維度
                                            
                                            if summed_codes is None:
                                                summed_codes = upsampled
                                            else:
                                                summed_codes = summed_codes + upsampled
                                        else:
                                            # 最後一個尺度直接累加
                                            if summed_codes is None:
                                                summed_codes = codes
                                            else:
                                                summed_codes = summed_codes + codes
                                    else:
                                        # print(f"VAE quantizer not found or incompatible")
                                        break
                                        
                                except Exception as scale_error:
                                    print(f"Error processing scale {si}: {scale_error}")
                                    continue
                            
                            # 解碼最終的 summed_codes
                            if summed_codes is not None and summed_codes.numel() > 0:
                                # print(f"Debug: Final summed_codes shape: {summed_codes.shape}")
                                # 移除時間維度進行解碼: [1, 32, 1, h, w] -> [1, 32, h, w]
                                summed_codes_for_decode = summed_codes.squeeze(-3)
                                reconstructed_img = vae.decode(summed_codes_for_decode)
                                # print(f"Debug: Reconstructed image shape: {reconstructed_img.shape}")
                            else:
                                raise ValueError("Failed to accumulate codes")
                        else:
                            # 傳統 VAE 處理方式
                            # print(f"Debug: Using traditional VAE reconstruction")
                            # 使用最大尺度的 tokens
                            largest_tokens = single_gt_tokens[-1]
                            
                            # 嘗試直接解碼
                            if hasattr(vae, 'decode_from_indices'):
                                reconstructed_img = vae.decode_from_indices(largest_tokens)
                            else:
                                # fallback to token visualization
                                raise AttributeError("Traditional VAE decode not implemented")
                    else:
                        raise AttributeError("VAE not available")
                    
                    # 確保重建圖像在正確設備上
                    reconstructed_img = reconstructed_img.to(device)
                    
                except Exception as vae_error:
                    # print(f"VAE decode failed: {vae_error}, using simple token visualization")
                    traceback.print_exc()
                    # Fallback: Create a simple visualization of token values
                    reconstructed_img = _visualize_tokens_as_image(single_gt_tokens, scale_schedule[:training_scales], device)
                
                # Create comparison grid: [Original, Condition (if exists), GT Reconstruction]
                comparison_images = []
                
                # Original input image (normalize from [-1,1] to [0,1] and ensure on device)
                orig_display = torch.clamp((single_inp.squeeze(0) + 1) / 2, 0, 1).to(device)
                comparison_images.append(orig_display)
                
                # Condition image (if exists)
                if single_condition is not None:
                    condition_display = torch.clamp((single_condition.squeeze(0) + 1) / 2, 0, 1).to(device)
                    comparison_images.append(condition_display)
                
                # Ground truth reconstruction (normalize and ensure on device)
                if isinstance(reconstructed_img, torch.Tensor):
                    if reconstructed_img.dim() == 4:
                        reconstructed_img = reconstructed_img.squeeze(0)  # Remove batch dim
                    
                    # Ensure on correct device
                    reconstructed_img = reconstructed_img.to(device)
                    
                    # Normalize based on the range of values (參考 infinity_pilot.py line 1129-1130)
                    # img = (img + 1) / 2
                    if reconstructed_img.min() < 0:  # If in [-1,1] range
                        recon_display = torch.clamp((reconstructed_img + 1) / 2, 0, 1)
                    else:  # If in [0,1] or other range
                        recon_display = torch.clamp(reconstructed_img, 0, 1)
                else:
                    # Convert numpy to tensor if needed
                    recon_display = torch.clamp(
                        torch.from_numpy(reconstructed_img).permute(2, 0, 1).to(device) / 255.0, 
                        0, 1
                    )
                
                comparison_images.append(recon_display)
                
                # Ensure all images have the same spatial dimensions
                target_h, target_w = orig_display.shape[-2:]
                resized_comparison = []
                for img in comparison_images:
                    if img.shape[-2:] != (target_h, target_w):
                        img_resized = torch.nn.functional.interpolate(
                            img.unsqueeze(0), size=(target_h, target_w), mode='bilinear', align_corners=False
                        ).squeeze(0)
                        resized_comparison.append(img_resized)
                    else:
                        resized_comparison.append(img)
                
                # Stack horizontally: [Original | Condition | GT Reconstruction] or [Original | GT Reconstruction]
                comparison_grid = torch.cat(resized_comparison, dim=2)  # Concatenate along width
                reconstruction_images.append(comparison_grid)
                
            except Exception as e:
                # print(f"Error processing training sample {i}: {e}")
                traceback.print_exc()
                
                # Create a placeholder image if processing fails
                h, w = inp_vis.shape[-2:]
                num_images = 3 if condition_vis is not None else 2
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