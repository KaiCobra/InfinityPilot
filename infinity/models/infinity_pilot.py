"""
Definition of Infinity transformer model.(original version)
"""

import math
import random
import time
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model
from torch.utils.checkpoint import checkpoint
from PIL import Image
import numpy as np

import infinity.utils.dist as dist
from infinity.utils.dist import for_visualize
from infinity.models.basic import flash_attn_func, flash_fused_op_installed, AdaLNBeforeHead, CrossAttnBlock, SelfAttnBlock, CrossAttention, FastRMSNorm, precompute_rope2d_freqs_grid
from infinity.utils import misc
from infinity.models.flex_attn import FlexAttn
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
import inspect

try:
    from infinity.models.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None

from infinity.models.infinity import Infinity
from infinity.utils.control_data_utils import numpy_to_pt, pil_to_numpy



def get_control_for_each_scale(control_image, scale):
    """
    Get tensors for each scale from extracted tensor.
    """
    def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
        return x.add(x).add_(-1)
    c_tensors = []
    c_images = []
    for pn in scale:
        c_res = control_image.resize((pn * 16, pn * 16))
        c_images.append(c_res)
        c_tensors.append(normalize_01_into_pm1(numpy_to_pt(pil_to_numpy(c_res))))
    return c_images, c_tensors

class MultiInpIdentity(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class TextAttentivePool(nn.Module):
    def __init__(self, Ct5: int, D: int):
        super().__init__()
        self.Ct5, self.D = Ct5, D
        if D > 4096:
            self.head_dim = 64 
        else:
            self.head_dim = 128

        self.num_heads = Ct5 // self.head_dim
        self.ca = CrossAttention(for_attn_pool=True, embed_dim=self.D, kv_dim=Ct5, num_heads=self.num_heads)
    def forward(self, ca_kv):
        return self.ca(None, ca_kv).squeeze(1)

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).reshape(-1, 1, 6, C)   # B16C


class MultipleLayers(nn.Module):
    def __init__(self, ls, num_blocks_in_a_chunk, index):
        super().__init__()
        self.module = nn.ModuleList()
        for i in range(index, index+num_blocks_in_a_chunk):
            self.module.append(ls[i])

    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, checkpointing_full_block=False, rope2d_freqs_grid=None, fusion_callback=None, block_offset=0):
        h = x
        for local_idx, m in enumerate(self.module):
            if fusion_callback is not None:
                h = fusion_callback(block_offset + local_idx, h)
            if checkpointing_full_block:
                h = torch.utils.checkpoint.checkpoint(m, h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                h = m(h, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid)
        return h

class FP32_Layernorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(),
                            self.eps).to(origin_dtype)


class InfinityPilot(Infinity):
    """
    ## 🧑‍🚀InfinityPilot: surf beyond the infinity!🛰️
    This is a variant of Infinity that can refer to the condition image and prompt text to generate images that imply to the conditions.
    
    Like the ControlNet for diffusion models, it can be used to control the generation process.

    Args:
        vae_local: VAE model or module used for image encoding/decoding.
        text_channels (int, default=0): Number of text channels for text-conditioned generation.
        text_maxlen (int, default=0): Maximum length of text input.
        selecting_idx (Optional[int], default=None): Index for class-conditioned generation.
        embed_dim (int, default=1024): Embedding dimension of the model.
        depth (int, default=16): Number of transformer blocks (model depth).
        num_heads (int, default=16): Number of attention heads.
        mlp_ratio (float, default=4.0): Ratio of MLP hidden dimension to embedding dimension.
        drop_rate (float, default=0.0): Dropout rate for regularization.
        drop_path_rate (float, default=0.0): Drop path rate for stochastic depth.
        norm_eps (float, default=1e-6): Epsilon value for normalization layers.
        rms_norm (bool, default=False): If True, use RMSNorm instead of LayerNorm.
        shared_aln (bool, default=False): If True, use shared adaptive layer normalization.
        head_aln (bool, default=True): If True, use adaptive normalization in attention heads.
        cond_drop_rate (float, default=0.1): Drop rate for classifier-free guidance.
        rand_uncond (bool, default=False): If True, randomly use unconditional generation.
        cross_attn_layer_scale (float, default=-1.0): Scaling factor for cross-attention layers.
        nm0 (bool, default=False): Custom normalization flag.
        tau (float, default=1): Temperature parameter for attention or sampling.
        cos_attn (bool, default=True): If True, use cosine attention mechanism.
        swiglu (bool, default=False): If True, use SwiGLU activation in MLPs.
        raw_scale_schedule (tuple, default=(1,2,3,4,5,6,8,10,13,16)): Schedule for raw scaling across layers.
        head_depth (int, default=1): Depth of the attention head module.
        top_p (float, default=0.0): Nucleus sampling parameter (top-p).
        top_k (float, default=0.0): Top-k sampling parameter.
        customized_flash_attn (bool, default=False): If True, use customized FlashAttention.
        fused_mlp (bool, default=False): If True, use fused MLP implementation.
        fused_norm (bool, default=False): If True, use fused normalization implementation.
        block_chunks (int, default=1): Number of chunks to split blocks for memory efficiency.
        checkpointing (Optional[Any], default=None): Checkpointing strategy for memory saving.
        pad_to_multiplier (int, default=0): Pad input to a multiple of this value.
        use_flex_attn (bool, default=False): If True, use flexible attention mechanism.
        batch_size (int, default=2): Batch size for training or inference.
        add_lvl_embeding_only_first_block (int, default=1): If True, add level embedding only in the first block.
        use_bit_label (int, default=1): If True, use bit-level labels.
        rope2d_each_sa_layer (int, default=0): If True, apply 2D RoPE to each self-attention layer.
        rope2d_normalized_by_hw (int, default=0): If True, normalize 2D RoPE by height/width.
        pn (Optional[Any], default=None): Additional parameter, purpose defined by implementation.
        train_h_div_w_list (Optional[list], default=None): List of height/width ratios for training.
        video_frames (int, default=1): Number of video frames for video input.
        always_training_scales (int, default=20): Number of scales always used during training.
        apply_spatial_patchify (int, default=0): If True, apply spatial patchification.
        inference_mode (bool, default=False): If True, set model to inference mode.
    """
    def __init__(self, infinity_base_model: Optional['Infinity'] = None, init_car_modules=False, freeze_infinity=True, **kwargs):
        """
        Args:
            infinity_base_model: 預訓練的 Infinity 基礎模型，如果提供則會復制其參數
            init_car_modules: 是否初始化 CAR 模塊
            freeze_infinity: 是否凍結 Infinity 基礎模型的參數
            **kwargs: 其他參數傳遞給 Infinity 父類
        """
        # 檢查 checkpoint 架構並自動調整參數
        if infinity_base_model is not None:
            if isinstance(infinity_base_model, dict):
                state_dict = infinity_base_model
            else:
                state_dict = infinity_base_model.state_dict() if hasattr(infinity_base_model, 'state_dict') else infinity_base_model
            
            # 檢測參數格式來自動設置架構
            has_ada_gss = any('ada_gss' in k for k in state_dict.keys())
            has_shared_ada_lin = any('shared_ada_lin' in k for k in state_dict.keys())
            has_individual_ada_lin = any('ada_lin' in k and 'shared_ada_lin' not in k for k in state_dict.keys())
            
            print(f"[InfinityPilot] Checkpoint architecture detection:")
            print(f"  - ada_gss: {has_ada_gss}")
            print(f"  - shared_ada_lin: {has_shared_ada_lin}")
            print(f"  - individual ada_lin: {has_individual_ada_lin}")
            
            # 根據檢測結果自動設置 shared_aln
            if has_ada_gss or has_shared_ada_lin:
                kwargs['shared_aln'] = True
                print(f"[InfinityPilot] Auto-setting shared_aln=True to match checkpoint")
            elif has_individual_ada_lin:
                kwargs['shared_aln'] = False
                print(f"[InfinityPilot] Auto-setting shared_aln=False to match checkpoint")
            else:
                print(f"[InfinityPilot] Could not detect ada_lin format, using default shared_aln={kwargs.get('shared_aln', False)}")
        
        # 保存所需的參數以便在建立控制塊時使用
        self._init_kwargs = kwargs.copy()
        
        print(f"[InfinityPilot] Initializing with shared_aln={kwargs.get('shared_aln', False)}")

        # super().__init__(**kwargs)
        sig = inspect.signature(super().__init__)
        valid_keys = sig.parameters.keys()
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_keys}
        super().__init__(**filtered_kwargs)
        self.disable_car_fusion = kwargs.get('disable_car_fusion', False)
        self.disable_car_merge = kwargs.get('disable_car_merge', False)
        
        self.num_block_chunks = kwargs.get('block_chunks', 1)
        
        if infinity_base_model is not None:
            self.load_infinity_weights(infinity_base_model)
        
        # 凍結 Infinity 基礎模型參數
        if freeze_infinity:
            self.freeze_infinity_parameters()

        if init_car_modules:
            self._init_car_modules()

        print("🔍 NaN detector registered on all normalization layers")
    
    def load_infinity_weights(self, infinity_model_or_state_dict):
        """從預訓練的 Infinity 模型載入權重，只載入非CAR模塊"""
        if isinstance(infinity_model_or_state_dict, dict):
            infinity_state_dict = infinity_model_or_state_dict
        else:
            infinity_state_dict = infinity_model_or_state_dict.state_dict()
        
        # 檢查是否為 FSDP 格式並轉換
        if any(key.startswith('block_chunks.') for key in infinity_state_dict.keys()):
            print("[INFO] Detected FSDP format weights, converting to standard format...")
            infinity_state_dict = self._convert_fsdp_to_standard_format(infinity_state_dict)
        
        # 過濾出只有Infinity基礎模型的權重（排除CAR相關）
        filtered_dict = {}
        for name, param in infinity_state_dict.items():
            # 跳過CAR相關的參數
            if self._is_car_parameter(name):
                continue
            filtered_dict[name] = param
        
        # 獲取當前模型的非CAR參數名稱用於調試
        current_infinity_params = {name for name, _ in self.named_parameters() 
                                 if not self._is_car_parameter(name)}
        source_infinity_params = set(filtered_dict.keys())

        # sorted_source_infinity_params = sorted(source_infinity_params)
        # sorted_current_infinity_params = sorted(current_infinity_params)

        # 調試信息：檢查哪些參數缺失
        missing_in_target = source_infinity_params - current_infinity_params
        missing_in_source = current_infinity_params - source_infinity_params
        
        # sort the name of missing parameters for better readability
        missing_in_target = sorted(missing_in_target)
        missing_in_source = sorted(missing_in_source)

        if len(missing_in_source) > 10:  # 如果缺失太多，顯示詳細信息
            print(f"Debug: Parameters in source but not in target: {len(missing_in_target)}")
            print(f"Debug: Parameters in target but not in source: {len(missing_in_source)}")
            
            # 顯示一些範例
            if missing_in_source:
                print(f"Sample missing in source: {list(missing_in_source)[:]}")
            if missing_in_target:
                print(f"Sample unexpected in source: {list(missing_in_target)[:]}")
        
        # 載入基礎模型權重
        missing_keys, unexpected_keys = self.load_state_dict(filtered_dict, strict=False)
        
        # 過濾掉CAR相關的missing keys（這些是正常的）
        real_missing = [k for k in missing_keys if not self._is_car_parameter(k)]
        
        # 統計不同類型的缺失參數
        ada_lin_missing = [k for k in real_missing if 'ada_lin' in k]
        zero_bias_missing = [k for k in real_missing if 'zero_k_bias' in k or 'zero_v_bias' in k]
        other_missing = [k for k in real_missing if 'ada_lin' not in k and 'zero_k_bias' not in k and 'zero_v_bias' not in k]
        
        print(f"Loaded Infinity base weights: {len(filtered_dict) - len(real_missing)}/{len(filtered_dict)}")
        if real_missing:
            print(f"Missing base model keys: {len(real_missing)}")
            if ada_lin_missing:
                print(f"  - AdaLIN related: {len(ada_lin_missing)} (model version difference)")
            if zero_bias_missing:
                print(f"  - Zero bias related: {len(zero_bias_missing)} (attention config difference)")
            if other_missing:
                print(f"  - Other missing: {len(other_missing)}")
                if len(other_missing) <= 5:
                    print(f"    Details: {other_missing}")
        
        # 如果只是配置差異的參數缺失，這是正常的
        config_diff_missing = len(ada_lin_missing) + len(zero_bias_missing)
        if config_diff_missing == len(real_missing):
            print("✓ All missing parameters are due to model configuration differences - this is normal")
        elif len(real_missing) < len(filtered_dict) * 0.1:  # 少於10%缺失是可接受的
            print("✓ Missing parameters are within acceptable range")
        else:
            print("⚠ Significant parameter mismatch - please verify model compatibility")
            
        return real_missing, unexpected_keys

    def save_separated_weights(self, save_dir):
        """分別保存Infinity基礎權重和CAR權重，T5風格"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 分離權重
        infinity_weights = {}
        car_weights = {}
        
        for name, param in self.state_dict().items():
            if self._is_car_parameter(name):
                car_weights[name] = param
            else:
                infinity_weights[name] = param
        
        # 保存
        torch.save(infinity_weights, os.path.join(save_dir, 'infinity_base_weights.pth'))
        torch.save(car_weights, os.path.join(save_dir, 'car_weights.pth'))
        
        print(f"Saved separated weights:")
        print(f"  Infinity base: {len(infinity_weights)} parameters")
        print(f"  CAR modules: {len(car_weights)} parameters")
        return save_dir

    def _convert_fsdp_to_standard_format(self, fsdp_state_dict):
        """將 FSDP 格式的權重轉換為標準格式 (內存優化版)
        
        FSDP 格式: block_chunks.X.module.Y.xxx
        標準格式: blocks.Z.xxx (其中 Z = X * chunk_size + Y)
        """
        converted_dict = {}
        chunk_size = 4  # 假設每個 chunk 有 4 個 block（根據錯誤信息推測）
        fsdp_keys_to_process = []
        
        # 首先收集需要轉換的鍵，避免在迭代時修改字典
        for key in fsdp_state_dict.keys():
            if key.startswith('block_chunks.'):
                fsdp_keys_to_process.append(key)
            else:
                # 非 block_chunks 的鍵，直接保留（避免內存複製）
                converted_dict[key] = fsdp_state_dict[key]
        
        # 批量處理 FSDP 格式的鍵，節省內存
        conversion_count = 0
        for key in fsdp_keys_to_process:
            # 解析 FSDP 格式的鍵
            # 格式: block_chunks.X.module.Y.rest_of_path
            parts = key.split('.')
            if len(parts) >= 4 and parts[2] == 'module':
                try:
                    chunk_idx = int(parts[1])  # X
                    block_idx = int(parts[3])  # Y
                    rest_path = '.'.join(parts[4:])  # rest_of_path
                    
                    # 計算實際的 block 索引
                    actual_block_idx = chunk_idx * chunk_size + block_idx
                    
                    # 生成標準格式的鍵並移動權重（避免複製）
                    new_key = f'blocks.{actual_block_idx}.{rest_path}'
                    converted_dict[new_key] = fsdp_state_dict[key]
                    conversion_count += 1
                except (ValueError, IndexError) as e:
                    # 無法解析的鍵，保持原樣
                    print(f"Warning: Could not parse FSDP key {key}: {e}")
                    converted_dict[key] = fsdp_state_dict[key]
            else:
                # 無法解析的鍵，保持原樣
                converted_dict[key] = fsdp_state_dict[key]
        
        print(f"[INFO] Converted {conversion_count} FSDP block parameters to standard format")
        
        # 清理原始字典的引用以釋放內存
        fsdp_state_dict.clear()
        return converted_dict

    def _is_car_parameter(self, param_name):
        """判斷參數是否屬於CAR模塊"""
        car_prefixes = ['car_']
        return any(param_name.startswith(prefix) for prefix in car_prefixes)
    
    def _assert_finite(self, name: str, tensor: torch.Tensor):
        if tensor is None:
            return
        if torch.isfinite(tensor).all():
            return
        bad_mask = ~torch.isfinite(tensor)
        bad_index = bad_mask.nonzero(as_tuple=False)[0]
        sample = tensor[tuple(bad_index.tolist())].item()
        raise FloatingPointError(
            f"[NaN detector] {name} contains non-finite values "
            f"(first bad index {bad_index.tolist()}, sample={sample})"
        )
    
    def freeze_infinity_parameters(self):
        """凍結 Infinity 基礎模型的參數"""
        frozen_count = 0
        for name, param in self.named_parameters():
            # 不凍結 CAR 相關的參數
            if not any(car_prefix in name for car_prefix in ['car_', 'control_']):
                param.requires_grad = False
                frozen_count += 1
        print(f"Frozen {frozen_count} Infinity base model parameters")
    
    def unfreeze_infinity_parameters(self):
        """解凍 Infinity 基礎模型的參數"""
        unfrozen_count = 0
        for name, param in self.named_parameters():
            if not any(car_prefix in name for car_prefix in ['car_', 'control_']):
                param.requires_grad = True
                unfrozen_count += 1
        print(f"Unfrozen {unfrozen_count} Infinity base model parameters")

    def _init_car_parameters(self):
        """為 CAR 子模組執行顯式初始化，避免未初始化權重導致數值異常。"""
        def init_fn(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.LayerNorm, FP32_Layernorm)):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
            elif isinstance(module, FastRMSNorm):
                if module.elementwise_affine:
                    nn.init.zeros_(module.weight)
        
        # for block in self.car_blocks:
        #     block.apply(init_fn)
        if hasattr(self, 'car_control_proj'):
            self.car_control_proj.apply(init_fn)

    def _init_car_blocks_from_transformer(self):
        """
        用 Layer Merge 方式從 blocks 初始化 car_blocks
        """
        # methods = ['merge', 'copy']
        sel = 1

        if sel == 0:
            step = self.depth // self.car_depth
            
            print(f"[car_init] merging from {self.depth} transformer blocks -> {self.car_depth} car blocks")
            with torch.no_grad():
                for i in range(self.car_depth):
                    merged = {}
                    idx_a = i * step
                    idx_b = min((i + 1) * step - 1, self.depth - 1)
                    wa = self.blocks[idx_a].state_dict()
                    wb = self.blocks[idx_b].state_dict()

                    for k in wa.keys():
                        a = wa[k].float()
                        b = wb[k].float()

                        if torch.isnan(a).any() or torch.isinf(a).any():
                            raise ValueError(f"Warning: NaN or Inf detected in weights of block {idx_a}, key {k}")
                            a = torch.nan_to_num(a)
                            b = torch.nan_to_num(b)
                        # 對 gating / scaling 參數採用偏淺層權重
                        if "bias" in k:
                            merged_v = torch.zeros_like(a)

                        elif any(x in k for x in ["ada_gss", "scale_mul"]):
                            merged_v = 1.0 * (0.8 * a + 0.2 * b)
                        else:
                            merged_v = 0.5 * (a + b)
                        
                        merged_v = torch.nan_to_num(merged_v)
                        merged_v = torch.clamp(merged_v, -3.0, 3.0)
                        merged[k] = merged_v.to(wa[k].dtype)

                    self.car_blocks[i].load_state_dict(merged)
                    print(f"[car_init] initialized car block {i} from transformer blocks {idx_a} to {idx_b}")
        elif sel == 1:
            print(f"[car_init] copying from first {self.car_depth} transformer blocks -> {self.car_depth} car blocks")
            with torch.no_grad():
                for i in range(self.car_depth):
                    source_idx = i
                    source_state = self.blocks[source_idx].state_dict()
                    target_state = self.car_blocks[i].state_dict()

                    for k in target_state.keys():
                        if k in source_state:
                            v = source_state[k].float()
                            if torch.isnan(v).any() or torch.isinf(v).any():
                                raise ValueError(f"Warning: NaN or Inf detected in weights of block {source_idx}, key {k}")
                            # v = torch.nan_to_num(v)
                            # v = torch.clamp(v, -3.0, 3.0)
                            target_state[k] = v.to(target_state[k].dtype)
                        else:
                            print(f"[car_init] key {k} not found in source block {source_idx}, skipping")

                    self.car_blocks[i].load_state_dict(target_state)
                    print(f"[car_init] initialized car block {i} from transformer block {source_idx}")

    def _init_car_modules(self):
        """初始化 CAR 控制模塊"""
        # CAR control modules - 參考 CAR 的架構
        init_kwargs = getattr(self, '_init_kwargs', {})
        # 建立 CAR 控制塊 - 只建立 depth//2 個塊（與 CAR 一致）
        from functools import partial
        
        # Get parameters from kwargs or use defaults
        norm_layer = partial(FastRMSNorm if init_kwargs.get('rms_norm', False) else nn.LayerNorm, eps=init_kwargs.get('norm_eps', 1e-4))
        existing_depth = getattr(self, 'car_depth', None)
        print(f"[debug] _init_car_modules called with kwargs.car_depth={init_kwargs.get('car_depth', None)} existing car_depth attr={existing_depth}")

        def _to_positive_int(val):
            try:
                iv = int(val)
            except (TypeError, ValueError):
                return None
            return iv if iv > 0 else None

        init_depth = init_kwargs.get('car_depth', None)
        chosen_depth = _to_positive_int(init_depth) or _to_positive_int(existing_depth)
        if chosen_depth is None:
            depth_val = _to_positive_int(getattr(self, 'depth', None))
            chosen_depth = max(1, depth_val // 2) if depth_val is not None else 8

        self.car_depth = chosen_depth
        print(f"[debug] CAR depth set to {self.car_depth}")

        dpr = [x.item() for x in torch.linspace(0, init_kwargs.get('drop_path_rate', 0.0), self.car_depth)]

        # self.car_control_proj = nn.Linear(self.d_vae, self.C)
        self.car_control_proj = nn.Sequential(
            nn.Linear(self.d_vae, 2 * self.C),
            nn.GELU(),
            nn.Linear(2 * self.C, self.C),
        )

        first, second = self.car_control_proj[0], self.car_control_proj[2]  # 兩個 Linear
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)
        nn.init.xavier_uniform_(second.weight)
        second.weight.data.mul_(0.5)                  # 縮小第二層
        nn.init.zeros_(second.bias)

        # nn.init.xavier_uniform_(self.car_control_proj.weight)
        # if self.car_control_proj.bias is not None:
        #     nn.init.zeros_(self.car_control_proj.bias)

        self.car_blocks = nn.ModuleList([
            (CrossAttnBlock if self.t2i else SelfAttnBlock)(
                embed_dim=self.C, kv_dim=self.D, cross_attn_layer_scale=init_kwargs.get('cross_attn_layer_scale', -1.), 
                cond_dim=self.D, act=True, shared_aln=init_kwargs.get('shared_aln', False), norm_layer=norm_layer,
                num_heads=init_kwargs.get('num_heads', 16), mlp_ratio=init_kwargs.get('mlp_ratio', 4.), 
                drop=init_kwargs.get('drop_rate', 0.), drop_path=dpr[block_idx], tau=init_kwargs.get('tau', 1), 
                cos_attn=init_kwargs.get('cos_attn', True), swiglu=init_kwargs.get('swiglu', False), 
                customized_flash_attn=getattr(self, 'customized_flash_attn', False), fused_mlp=init_kwargs.get('fused_mlp', False), 
                fused_norm_func=getattr(self, '_fused_norm_func', None),
                checkpointing_sa_only=getattr(self, 'checkpointing', None) == 'self-attn',
                use_flex_attn=init_kwargs.get('use_flex_attn', False), batch_size=init_kwargs.get('batch_size', 2), 
                pad_to_multiplier=init_kwargs.get('pad_to_multiplier', 0), 
                rope2d_normalized_by_hw=init_kwargs.get('rope2d_normalized_by_hw', 0),
            )
            for block_idx in range(self.car_depth)
        ])

        self.car_fusion_norms = nn.ModuleList([
            norm_layer(2*self.C)  
            # nn.LayerNorm(self.C, eps=init_kwargs.get('norm_eps', 1e-4))
            for _ in range(self.car_depth)
        ])
        
        for norm in self.car_fusion_norms:
            if isinstance(norm, nn.LayerNorm) or isinstance(norm, FP32_Layernorm):
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)
            elif isinstance(norm, FastRMSNorm):
                nn.init.ones_(norm.weight)

        self.car_fusion_linears = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.C, 4 * self.C),
                nn.GELU(),
                nn.Linear(4 * self.C, self.C)
                )
            for _ in range(self.car_depth)
        ])

        for seq in self.car_fusion_linears:
            first, second = seq[0], seq[2]                # 兩個 Linear
            nn.init.xavier_uniform_(first.weight)
            nn.init.zeros_(first.bias)
            nn.init.xavier_uniform_(second.weight)
            second.weight.data.mul_(0.5)                  # 縮小第二層
            nn.init.zeros_(second.bias)

        # self.car_fusion_scales = nn.Parameter(torch.zeros(self.car_depth))

        # Initialize CAR blocks by borrowing weights from corresponding Infinity transformer blocks
        if not getattr(self, 'disable_car_merge', False) and hasattr(self, 'blocks') and len(self.blocks) >= self.car_depth:
            try:
                self._init_car_blocks_from_transformer()
            except Exception as e:
                print(f"[WARNING] Failed to initialize CAR blocks from Infinity transformer weights: {e}")
        else:

            print("[car_init] Skipping merge from transformer blocks; using default CAR initialization.")
            self._init_car_parameters()

        print(f"Initialized CAR modules with {len(self.car_blocks)} control blocks")
    
    def has_car_modules(self):
        """檢查是否已初始化 CAR 模塊"""
        return hasattr(self, 'car_control_proj') and hasattr(self, 'car_blocks')

    def _build_control_ca_kv(self, control_tokens: Optional[List[Optional[torch.Tensor]]], scale_schedule: List[Tuple[int, int, int]]):
        if control_tokens is None:
            raise ValueError("control_tokens cannot be None when building control CA KV")
            return None
        if not self.has_car_modules():
            raise RuntimeError("CAR modules not initialized; call `init_car_modules_if_needed()` before wrapping with FSDP.")

        per_scale = []
        for scale_idx, tokens in enumerate(control_tokens):
            if tokens is None:
                per_scale.append(None)
                continue
            if tokens.shape[-1] != self.car_control_proj[0].in_features:
                raise ValueError(f"Control tokens dim {tokens.shape[-1]} does not match expected {self.car_control_proj[0].in_features}")
            tok = self.car_control_proj(tokens)
            lvl_index = scale_idx % self.lvl_embed.num_embeddings
            lvl_vec = self.lvl_embed.weight[lvl_index].to(tok.dtype).view(1, 1, -1)
            tok = tok + lvl_vec

            B, L, C = tok.shape
            cu_seqlens_k = torch.arange(0, (B + 1) * L, L, dtype=torch.int32, device=tok.device)
            kv_compact = tok.reshape(B * L, C).contiguous()
            per_scale.append({
                'kv': (kv_compact, cu_seqlens_k, L),
                'proj': tok
            })

        return per_scale

    def _prepare_car_fusion_map(
        self,
        cond_BD_or_gss: torch.Tensor,
        scale_schedule: List[Tuple[int, int, int]],
        control_scale_info: Optional[List[Optional[Dict[str, Any]]]],
        seq_len: int,
        *,
        pad_to_multiplier: int = 0,
    ) -> Dict[int, Tuple[int, Tuple[int, int], torch.Tensor]]:
        """
        回傳 {block_index: (car_idx, (start, end), car_feat_segment)}
        car_feat_segment 只包含該 scale 的片段，shape=[B, (end-start), C]。
        """
        if control_scale_info is None or not control_scale_info or not self.has_car_modules():
            return {}

        bs, device, dtype = cond_BD_or_gss.shape[0], cond_BD_or_gss.device, cond_BD_or_gss.dtype

        # 計算各 scale 的 token 長度與序列 offset（含 SOS 跳過）
        scale_token_lengths = [int(np.prod(pn)) for pn in scale_schedule]
        total_tokens = 1 + sum(scale_token_lengths)
        if seq_len < total_tokens:
            raise ValueError(f"seq_len ({seq_len}) must cover all scheduled tokens ({total_tokens})")

        scale_offsets = []
        ptr = 1
        for length in scale_token_lengths:
            scale_offsets.append((ptr, ptr + length))  # [start, end)
            ptr += length

        # 逐 scale 跑 CAR，保留每層的局部輸出
        car_layer_outputs_per_scale = [
            [None for _ in range(len(scale_schedule))] for _ in range(self.car_depth)
        ]

        for scale_idx, pn in enumerate(scale_schedule):
            info = control_scale_info[scale_idx] if (control_scale_info and scale_idx < len(control_scale_info)) else None
            scale_len = int(np.prod(pn))

            if info is None:
                zero_feat = cond_BD_or_gss.new_zeros(bs, scale_len, self.C)
                for layer_idx in range(self.car_depth):
                    car_layer_outputs_per_scale[layer_idx][scale_idx] = zero_feat
                continue

            proj = info["proj"]  # [B_ctrl, L_scale, C]
            kv_compact, cu_seqlens_k, max_seqlen_k = info["kv"]

            ctrl_bs, ctrl_len, _ = proj.shape
            if ctrl_bs != bs:
                if bs % ctrl_bs != 0:
                    raise ValueError(f"Control tokens batch {ctrl_bs} cannot match inference batch {bs}")
                repeat_factor = bs // ctrl_bs
                proj = proj.repeat(repeat_factor, 1, 1)
                kv_shape = kv_compact.shape[-1]
                kv_compact = kv_compact.view(ctrl_bs, max_seqlen_k, kv_shape).repeat(repeat_factor, 1, 1)
                kv_compact = kv_compact.view(bs * max_seqlen_k, kv_shape).contiguous()
                cu_seqlens_k = torch.arange(0, (bs + 1) * max_seqlen_k, max_seqlen_k, device=proj.device, dtype=cu_seqlens_k.dtype)

            # 簡單用 causal mask；FlexAttn 也可以，但此處不必 pad 到整數倍
            ctrl_len = proj.shape[1]
            causal_mask = torch.triu(torch.ones(ctrl_len, ctrl_len, device=proj.device, dtype=torch.bool), diagonal=1)
            car_attn_bias = torch.zeros(1, 1, ctrl_len, ctrl_len, device=proj.device, dtype=torch.float32)
            car_attn_bias.masked_fill_(causal_mask, float('-inf'))

            for layer_idx, car_block in enumerate(self.car_blocks):
                with torch.amp.autocast('cuda', enabled=False):
                    h_fp32 = proj.float()
                    kv_fp32 = kv_compact.float()
                    ca_kv_fp32 = (kv_fp32, cu_seqlens_k, max_seqlen_k)
                    block_output = car_block(
                        x=h_fp32,
                        cond_BD=cond_BD_or_gss,
                        ca_kv=ca_kv_fp32,
                        attn_bias_or_two_vector=car_attn_bias,
                        scale_schedule=scale_schedule,
                        rope2d_freqs_grid=self.rope2d_freqs_grid,
                        scale_ind=scale_idx,
                    )
                block_output = block_output.to(dtype)
                residual = proj.to(dtype)
                if block_output.shape[1] != residual.shape[1]:
                    residual = residual[:, :block_output.shape[1], :]
                h_out = (block_output + residual)
                self._assert_finite(f"car_block[{layer_idx}] scale {scale_idx}", h_out)
                car_layer_outputs_per_scale[layer_idx][scale_idx] = h_out  # [B, scale_len, C]
                proj = h_out

        # 依 token 比例分配 main blocks 給各 scale（沿用你的策略）
        num_blocks = len(self.blocks)
        tokens_per_scale = torch.tensor(scale_token_lengths, dtype=torch.float32, device=device)
        ratios = (tokens_per_scale / tokens_per_scale.sum()).tolist()
        blocks_per_scale = [max(1, int(round(r * num_blocks))) for r in ratios]
        delta = sum(blocks_per_scale) - num_blocks
        if delta > 0:
            for i in range(len(blocks_per_scale)):
                if delta == 0: break
                if blocks_per_scale[i] > 1:
                    blocks_per_scale[i] -= 1
                    delta -= 1
        elif delta < 0:
            for i in reversed(range(len(blocks_per_scale))):
                if delta == 0: break
                blocks_per_scale[i] += 1
                delta += 1

        scale_block_ranges = []
        cursor = 0
        for cnt in blocks_per_scale:
            scale_block_ranges.append((cursor, cursor + cnt))
            cursor += cnt
        assert cursor == num_blocks

        # 建立 fusion_map：每個 block → (car_idx, (start,end), car_feat_segment)
        fusion_map: Dict[int, Tuple[int, Tuple[int, int], torch.Tensor]] = {}
        for scale_idx, (b_start, b_end) in enumerate(scale_block_ranges):
            seg_len = b_end - b_start
            if seg_len <= 0:
                continue
            (start, end) = scale_offsets[scale_idx]
            car_indices = torch.linspace(0, self.car_depth - 1, steps=seg_len, dtype=torch.int64).tolist()
            for k, bidx in enumerate(range(b_start, b_end)):
                car_idx = int(car_indices[k])
                feat_segment = car_layer_outputs_per_scale[car_idx][scale_idx].contiguous()  # [B, end-start, C]
                fusion_map[bidx] = (car_idx, (start, end), feat_segment)

        return fusion_map

    def _make_car_fusion_hook(self, fusion_map: Dict[int, Tuple[int, Tuple[int, int], torch.Tensor]]):
        if not fusion_map or getattr(self, "disable_car_fusion", False):
            return lambda _idx, seq: seq

        def _apply(block_index: int, seq: torch.Tensor) -> torch.Tensor:
            self._assert_finite(f"infinity_block[{block_index}] pre_fusion", seq)
            entry = fusion_map.get(block_index, None)
            if entry is None:
                return seq

            car_idx, (start, end), car_feat = entry  # car_feat: [B, end-start, C]
            seqlen = seq.size(1)
            # 安全裁切（避免 pad-to-multiplier 或其他情況）
            seg_start = min(start, seqlen)
            seg_end   = min(end,   seqlen)
            if seg_end <= seg_start:
                return seq  # 沒有可融合的片段

            seg_len = seg_end - seg_start
            seq_seg = seq[:, seg_start:seg_end, :]                          # [B, seg_len, C]
            car_seg = car_feat[:, :seg_len, :].to(seq.dtype).contiguous()   # [B, seg_len, C]

            # 若 CAR 段比 seg_len 短，做 0 補齊；通常不會，但保守處理
            if car_seg.size(1) < seg_len:
                pad_len = seg_len - car_seg.size(1)
                car_seg = F.pad(car_seg, (0, 0, 0, pad_len))

            fused_in = torch.cat([seq_seg, car_seg], dim=-1)                # [B, seg_len, 2C]
            fused_in = self.car_fusion_norms[car_idx](fused_in)
            fused_seg = self.car_fusion_linears[car_idx](fused_in)          # [B, seg_len, C]

            self._assert_finite(f"infinity_block[{block_index}] post_fusion", fused_seg)

            # 只覆蓋該片段；避免 in-place 影響 autograd，clone 一份
            seq_out = seq.clone()
            seq_out[:, seg_start:seg_end, :] = fused_seg
            return seq_out

        return _apply
 
    def init_car_modules_if_needed(self):
        """如果尚未初始化則初始化 CAR 模塊"""
        if not self.has_car_modules():
            self._init_car_modules()
    
    def load_car_weights(self, car_state_dict: dict, strict=False):
        """載入 CAR 模塊的權重
        Args:
            car_state_dict: CAR 模塊的 state_dict
            strict: 是否嚴格匹配參數名稱
        """
        if not self.has_car_modules():
            print("CAR modules not initialized, initializing now...")
            self._init_car_modules()
        
        # 規範 FSDP / module 前綴，確保鍵與當前模型一致
        if any(key.startswith('block_chunks.') for key in car_state_dict.keys()):
            print('[INFO] Detected FSDP block_chunks format for CAR weights, converting...')
            converted = {}
            for key, value in car_state_dict.items():
                if key.startswith('block_chunks.'):
                    parts = key.split('.')
                    if len(parts) >= 4 and parts[2] == 'module':
                        rest_path = '.'.join(parts[4:])
                        if rest_path:
                            converted[rest_path] = value
                            continue
                converted[key] = value
            car_state_dict = converted

        if any('._fsdp_wrapped_module.' in key or key.startswith('module.') for key in car_state_dict.keys()):
            print('[INFO] Detected FSDP-wrapped CAR weights, normalizing key prefixes...')

        normalized_state = {}
        for key, value in car_state_dict.items():
            normalized_key = None
            for marker in ('car_', 'control_'):
                idx = key.find(marker)
                if idx != -1:
                    normalized_key = key[idx:]
                    break
            if normalized_key is None:
                continue
            if normalized_key in normalized_state:
                print(f"[WARN] Duplicate CAR key after normalization: {normalized_key}, keeping latest copy")
            normalized_state[normalized_key] = value

        if normalized_state:
            car_state_dict = normalized_state

        # 篩選出 CAR 相關的參數
        car_params = {}
        current_state_dict = self.state_dict()
        
        loaded_count = 0
        skipped_count = 0
        
        for name, param in car_state_dict.items():
            if any(car_prefix in name for car_prefix in ['car_', 'control_']):
                if name in current_state_dict:
                    current_param = current_state_dict[name]
                    if current_param.shape == param.shape:
                        car_params[name] = param
                        loaded_count += 1
                    else:
                        print(f"Warning: Shape mismatch for CAR parameter {name}: expected {current_param.shape}, got {param.shape}")
                        skipped_count += 1
                        if strict:
                            raise RuntimeError(f"Shape mismatch for {name}")
                else:
                    print(f"Warning: CAR parameter {name} not found in current model")
                    skipped_count += 1
                    if strict:
                        raise RuntimeError(f"Parameter {name} not found")
        
        if car_params:
            missing_keys, unexpected_keys = self.load_state_dict(car_params, strict=False)
            print(f"Successfully loaded {loaded_count} CAR parameters")
            if missing_keys:
                print(f"Missing CAR keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected CAR keys: {len(unexpected_keys)}")
        else:
            print("No matching CAR parameters found to load, using random initialization")
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} CAR parameters due to mismatch")
    
    def save_car_weights(self):
        """保存 CAR 模塊的權重
        Returns:
            dict: 包含所有 CAR 參數的字典
        """
        car_state = {}
        for name, param in self.named_parameters():
            if any(car_prefix in name for car_prefix in ['car_', 'control_']):
                car_state[name] = param.detach().cpu()
        return car_state
    
    def save_infinity_weights(self):
        """保存 Infinity 基礎模型的權重
        Returns:
            dict: 包含所有 Infinity 基礎參數的字典
        """
        infinity_state = {}
        for name, param in self.named_parameters():
            if not any(car_prefix in name for car_prefix in ['car_', 'control_']):
                infinity_state[name] = param.detach().cpu()
        return infinity_state
    
    def get_car_parameters(self):
        """獲取所有 CAR 模塊的參數"""
        car_params = []
        for name, param in self.named_parameters():
            if any(car_prefix in name for car_prefix in ['car_', 'control_']):
                car_params.append(param)
        return car_params
    
    def get_infinity_parameters(self):
        """獲取所有 Infinity 基礎模型的參數"""
        infinity_params = []
        for name, param in self.named_parameters():
            if not any(car_prefix in name for car_prefix in ['car_', 'control_']):
                infinity_params.append(param)
        return infinity_params

    def forward(self, label_B_or_BLT: Union[torch.LongTensor, Tuple[torch.FloatTensor, torch.IntTensor, int]], x_BLC_wo_prefix: torch.Tensor, scale_schedule: List[Tuple[int]],
        cfg_infer=False, control_tokens: Optional[List[Optional[torch.Tensor]]] = None,  **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:  # returns logits_BLV


        if cfg_infer:
            return self.autoregressive_infer_cfg(label_B_or_BLT=label_B_or_BLT, scale_schedule=scale_schedule, control_tokens=control_tokens, **kwargs)
        
        x_BLC_wo_prefix = x_BLC_wo_prefix.float()       # input should be float32
        B = x_BLC_wo_prefix.shape[0]

        # [1. get input sequence x_BLC]
        with torch.amp.autocast('cuda', enabled=False):
            kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
            # drop cond
            total = 0
            for le in lens:
                if random.random() < self.cond_drop_rate:
                    kv_compact[total:total+le] = self.cfg_uncond[:le]
                total += le
            must_on_graph = self.cfg_uncond[0, 0] * 0
            kv_compact = self.text_norm(kv_compact).contiguous()
            sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)).float().contiguous()    # cond_BD should be float32
            kv_compact = self.text_proj_for_ca(kv_compact).contiguous()
            kv_compact[0, 0] += must_on_graph
            ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
            
            cond_BD_or_gss = self.shared_ada_lin(cond_BD).contiguous()  # gss: gamma, scale, shift; cond_BD_or_gss should be float32
            
            sos = sos.unsqueeze(1).expand(B, 1, -1) + self.pos_start.expand(B, 1, -1)
            x_BLC = torch.cat((sos, self.word_embed(self.norm0_ve(x_BLC_wo_prefix))), dim=1)
            self._assert_finite("x_BLC_after_embed", x_BLC)

            # [1.1. pad the seqlen dim]
            l_end = x_BLC.shape[1]
            need_to_pad = (l_end + self.pad_to_multiplier - 1) // self.pad_to_multiplier * self.pad_to_multiplier - l_end # 0
            
            if self.customized_flash_attn:
                Infinity_visible_kvlen = self.Infinity_visible_kvlen[:l_end]
                Infinity_invisible_qlen = self.Infinity_invisible_qlen[:l_end]
                attn_bias_or_two_vector = (Infinity_visible_kvlen, Infinity_invisible_qlen)
                # todo: solve need_to_pad here
            elif self.use_flex_attn:
                if need_to_pad:
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                assert x_BLC.shape[-1] % 128 == 0, 'x_BLC.shape[-1] % 128 != 0'
                attn_bias_or_two_vector = None
            else:
                d: torch.Tensor = torch.cat([torch.full((pn[0]*pn[1]*pn[2],), i) for i, pn in enumerate(scale_schedule)]).view(1, l_end, 1)
                dT = d.transpose(1, 2)    # dT: 11L
                attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, l_end, l_end)
                attn_bias = attn_bias_for_masking[:, :, :l_end, :l_end].contiguous()   # attn_bias: 11LL
                if need_to_pad:
                    attn_bias = F.pad(attn_bias, (0, need_to_pad, 0, need_to_pad), value=-torch.inf)
                    attn_bias[0, 0, l_end:, 0] = 0
                    x_BLC = F.pad(x_BLC, (0, 0, 0, need_to_pad))
                attn_bias_or_two_vector = attn_bias.type_as(x_BLC).to(x_BLC.device)
        
        if self.use_flex_attn:
            attn_fn = self.attn_fn_compile_dict[tuple(scale_schedule)]
        else:
            attn_fn = None

        control_scale_info = self._build_control_ca_kv(control_tokens, scale_schedule)
        fusion_map = self._prepare_car_fusion_map(
            cond_BD_or_gss=cond_BD_or_gss,
            scale_schedule=scale_schedule,
            control_scale_info=control_scale_info,
            seq_len=x_BLC.size(1),
            pad_to_multiplier=self.pad_to_multiplier if self.use_flex_attn else 0,
        )
        _apply_car_fusion = self._make_car_fusion_hook(fusion_map)

        # [2. block loop] - 修改以支援控制條件
        checkpointing_full_block = self.checkpointing == 'full-block' and self.training
        if self.num_block_chunks == 1:
            global_block_idx = 0
            for i, b in enumerate(self.blocks):
                # x_BLC = _apply_car_fusion(global_block_idx, x_BLC)
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                
                x_BLC = _apply_car_fusion(global_block_idx, x_BLC)
                if checkpointing_full_block:
                    x_BLC = torch.utils.checkpoint.checkpoint(b, x_BLC, cond_BD_or_gss, ca_kv, attn_bias_or_two_vector, attn_fn, scale_schedule, self.rope2d_freqs_grid, use_reentrant=False)
                else:
                    x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_bias_or_two_vector, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid)
                
                global_block_idx += 1
        else:
            print("[WARNING] Using block chunks in CAR mode; ensure this is intended.")
            global_block_idx = 0
            for i, chunk in enumerate(self.block_chunks):
                def fusion_cb(idx, seq):
                    return _apply_car_fusion(idx, seq)
                if self.add_lvl_embeding_only_first_block and i == 0:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)
                if not self.add_lvl_embeding_only_first_block:
                    x_BLC = self.add_lvl_embeding_for_x_BLC(x_BLC, scale_schedule, need_to_pad)

                # 在後半部分塊中融合控制特徵
                x_BLC = _apply_car_fusion(global_block_idx, x_BLC)
                x_BLC = chunk(
                    x=x_BLC,
                    cond_BD=cond_BD_or_gss,
                    ca_kv=ca_kv,
                    attn_bias_or_two_vector=attn_bias_or_two_vector,
                    attn_fn=attn_fn,
                    scale_schedule=scale_schedule,
                    checkpointing_full_block=checkpointing_full_block,
                    rope2d_freqs_grid=self.rope2d_freqs_grid,
                    fusion_callback=fusion_cb,
                    block_offset=global_block_idx,
                )
                global_block_idx += len(chunk.module)

        # [3. unpad the seqlen dim, and then get logits]
        return self.get_logits(x_BLC[:, :l_end], cond_BD)    # return logits BLV, V is vocab_size

    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,
        label_B_or_BLT=None,
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],
        vae_type=0, softmax_merge_topk=-1, ret_img=False,
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,
        save_img_path=None,
        sampling_per_bits=1,
        control_tokens: Optional[List[Optional[torch.Tensor]]] = None,
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule
        
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B

        kv_compact = self.text_norm(kv_compact)
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k)) # sos shape: [2, 4096]
        kv_compact = self.text_proj_for_ca(kv_compact) # kv_compact shape: [304, 4096]
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()
        accu_BChw, cur_L, ret = None, 0, []  # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []
        
        control_scale_info = self._build_control_ca_kv(control_tokens, scale_schedule)
        total_seq_tokens = 1 + sum(int(np.prod(pn)) for pn in scale_schedule)
        if self.use_flex_attn and self.pad_to_multiplier:
            seq_len_for_car = ((total_seq_tokens + self.pad_to_multiplier - 1) // self.pad_to_multiplier) * self.pad_to_multiplier
        else:
            seq_len_for_car = total_seq_tokens
        fusion_map = self._prepare_car_fusion_map(
            cond_BD_or_gss=cond_BD_or_gss,
            scale_schedule=scale_schedule,
            control_scale_info=control_scale_info,
            seq_len=seq_len_for_car,
            pad_to_multiplier=self.pad_to_multiplier if self.use_flex_attn else 0,
        )
        apply_car = self._make_car_fusion_hook(fusion_map)

        # define model blocks
        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
            for b in self.car_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)
            for block_chunk_ in self.car_blocks:
                if hasattr(block_chunk_, 'module'):
                    for module in block_chunk_.module.module:
                        (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)
                else:
                    (block_chunk_.sa if isinstance(block_chunk_, CrossAttnBlock) else block_chunk_.attn).kv_caching(True)
        
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        
        num_stages_minus_1 = len(scale_schedule)-1
        summed_codes = 0
        
        for si, pn in enumerate(scale_schedule):   # si: i-th segment
            cfg = cfg_list[si]
            if si >= trunk_scale:
                break
            cur_L += np.array(pn).prod()

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

            # 處理控制條件分支（參考 CAR 的做法）

            # 主分支處理
            layer_idx = 0
            global_block_idx = 0
            for block_idx, b in enumerate(self.block_chunks):
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                if not self.add_lvl_embeding_only_first_block: 
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                
                for m in b.module:
                    # 在後半部分融合控制特徵
                    last_stage = apply_car(global_block_idx, last_stage)
                    last_stage = m(x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, scale_ind=si)
                    
                    if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                        last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    
                    global_block_idx += 1
                    layer_idx += 1
            
            if (cfg != 1) and add_cfg_on_logits: # True
                logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])
                logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
            else:
                logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
            
            if self.use_bit_label: #True
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2] # si=0: [1, 1, 64] si=1: [1, 4, 64]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2) # si=0: [1, 32, 2] si=1: [1, 128, 2]
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1) # si=0: [1, 1, 32] si=1: [1, 4, 32]
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            
            if vae_type != 0:
                assert returns_vemb
                if si < gt_leak: # false
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d] si=3: [1,6,6,32]
                    if self.apply_spatial_patchify: # unpatchify operation
                        idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, 32] si=3: [1,1,6,6,32]

                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]

                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify: # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                else:
                    
                    summed_codes += codes
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)
            
            if si != num_stages_minus_1:
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)

        if inference_mode:
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
            for b in self.car_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)
            for block_chunk_ in self.car_blocks:
                if hasattr(block_chunk_, 'module'):
                    for module in block_chunk_.module.module:
                        (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)
                else:
                    (block_chunk_.sa if isinstance(block_chunk_, CrossAttnBlock) else block_chunk_.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        
        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img

def sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = logits_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = logits_BlV < logits_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
    if top_p > 0:
        sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(logits_BlV.softmax(dim=-1).view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)

def sampling_with_top_k_top_p_also_inplace_modifying_probs_(probs_BlV: torch.Tensor, top_k: int = 0, top_p: float = 0.0, rng=None, num_samples=1) -> torch.Tensor:  # return idx, shaped (B, l)
    B, l, V = probs_BlV.shape
    if top_k > 0:
        top_k = min(top_k, V)
        idx_to_remove = probs_BlV < probs_BlV.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
        probs_BlV.masked_fill_(idx_to_remove, 0)
    if top_p > 0:
        sorted_probs, sorted_idx = probs_BlV.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_probs.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        probs_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), 0)
    # sample (have to squeeze cuz multinomial can only be used on 2D tensor)
    probs_BlV = probs_BlV / probs_BlV.sum(-1, keepdims=True)
    replacement = num_samples >= 0
    num_samples = abs(num_samples)
    return torch.multinomial(probs_BlV.view(-1, V), num_samples=num_samples, replacement=replacement, generator=rng).view(B, l, num_samples)


def get_params_num(d, w, mlp):
    m = round(mlp * w / 256) * 256
    s = d * (w**2 * 8 + w*m * 2)    # sa+ca, mlp
    s += w**2 * 6       # saln
    s += 4096 * w       # pred
    s += 32 * w         # we
    
    Ct5 = 4096
    s += Ct5*w * 4      # T5 attn pool
    s += Ct5*w + w*w    # T5 mlp
    return f'{s/1e9:.2f}B'


TIMM_KEYS = {'img_size', 'pretrained', 'pretrained_cfg', 'pretrained_cfg_overlay', 'global_pool'}

# InfinityPilot model registrations
@register_model
def infinity_pilot_2b(depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, **kwargs): 
    return InfinityPilot(depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=4, drop_path_rate=drop_path_rate, **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS})
