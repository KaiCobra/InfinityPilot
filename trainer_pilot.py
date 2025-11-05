import random
import time
import gc
import os.path as osp
from functools import partial
from pprint import pformat
from typing import List, Optional, Tuple, Union, Dict
from collections import defaultdict, deque

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig, StateDictType
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import numpy as np
import torch.distributed as tdist
from torch.amp import autocast
import cv2

# Import visualization tools
try:
    import torchvision.transforms.functional as TF
    from PIL import Image
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Visualization tools not available. Install torchvision and PIL for image saving.")

# Import parameter visualizer
try:
    from debug_utils.parameter_visualizer import ParameterChangeVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    print("Parameter visualizer not available. Install matplotlib and seaborn for visualization.")

# Import NaN detector
try:
    from debug_utils.nan_detector import NaNDetector
    NAN_DETECTOR_AVAILABLE = True
except ImportError:
    NAN_DETECTOR_AVAILABLE = False
    print("NaN detector not available.")

import infinity.utils.dist as dist
from infinity.models.infinity_pilot import InfinityPilot
from infinity.models.ema import update_ema
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.amp_opt import AmpOptimizer
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w
from debug_utils.training_visualizer import _generate_training_visualization


Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

class DynamicScaleManager:
    def __init__(
        self,
        initial_scale: int,
        target_scale: int,
        patience_low: int,
        patience_high: int,
        transition_scale: int,
        loss_delta: float,
        loss_window: int,
    ) -> None:
        self.current_scale = max(1, int(initial_scale))
        self.target_scale = max(self.current_scale, int(target_scale))
        self.patience_low = max(1, int(patience_low))
        self.patience_high = max(1, int(patience_high))
        self.transition_scale = int(transition_scale)
        self.loss_delta = float(loss_delta)
        self.loss_window = max(1, int(loss_window))
        self.loss_buffer: deque = deque(maxlen=self.loss_window)
        self.best_smoothed_loss = float('inf')
        self.last_improvement_step = 0
        self.last_scale_step = 0

    @property
    def enabled(self) -> bool:
        return self.current_scale < self.target_scale

    def get_current_limit(self) -> int:
        return self.current_scale

    def _current_patience(self) -> int:
        return self.patience_low if self.current_scale < self.transition_scale else self.patience_high

    def register_step(self, g_it: int, loss_value: float) -> Optional[int]:
        if not self.enabled:
            return None
        self.loss_buffer.append(float(loss_value))
        if len(self.loss_buffer) < self.loss_window:
            return None
        current_avg = sum(self.loss_buffer) / len(self.loss_buffer)
        if current_avg < self.best_smoothed_loss - self.loss_delta:
            self.best_smoothed_loss = current_avg
            self.last_improvement_step = g_it
        patience = self._current_patience()
        if g_it - self.last_scale_step < patience:
            return None
        if g_it - self.last_improvement_step >= patience:
            self.current_scale = min(self.target_scale, self.current_scale + 1)
            self.last_scale_step = g_it
            self.last_improvement_step = g_it
            self.best_smoothed_loss = float('inf')
            self.loss_buffer.clear()
            return self.current_scale
        return None

    def get_state(self) -> Dict[str, Union[int, float, List[float]]]:
        return {
            'current_scale': self.current_scale,
            'target_scale': self.target_scale,
            'best_smoothed_loss': self.best_smoothed_loss,
            'last_improvement_step': self.last_improvement_step,
            'last_scale_step': self.last_scale_step,
            'loss_buffer': list(self.loss_buffer),
        }

    def load_state(self, state: Optional[Dict[str, Union[int, float, List[float]]]]) -> None:
        if not state:
            return
        self.current_scale = max(1, int(state.get('current_scale', self.current_scale)))
        self.target_scale = max(self.current_scale, int(state.get('target_scale', self.target_scale)))
        self.best_smoothed_loss = float(state.get('best_smoothed_loss', self.best_smoothed_loss))
        self.last_improvement_step = int(state.get('last_improvement_step', self.last_improvement_step))
        self.last_scale_step = int(state.get('last_scale_step', self.last_scale_step))
        buffer_values = state.get('loss_buffer', [])
        self.loss_buffer.clear()
        for value in buffer_values[-self.loss_window:]:
            self.loss_buffer.append(float(value))

class InfinityPilotTrainer(object):
    def __init__(
        self, is_visualizer: bool, device, raw_scale_schedule: Tuple[int, ...],
        vae_local, gpt_wo_ddp: InfinityPilot, gpt: DDP, ema_ratio: float, max_it: int,
        gpt_opt: AmpOptimizer, label_smooth: float, z_loss_ratio: float, eq_loss: int, xen: bool,
        dbg_unused=False,zero=0, vae_type=True, reweight_loss_by_scale=False,
        gpt_wo_ddp_ema=None, gpt_ema=None, use_fsdp_model_ema=False, other_args=None,
    ):
        super(InfinityPilotTrainer, self).__init__()
        self.dbg_unused = dbg_unused
        self._args_ref = other_args
        
        self.zero = zero
        self.vae_type = vae_type
        
        self.gpt: Union[DDP, FSDP, nn.Module]
        self.gpt, self.vae_local, self.quantize_local = gpt, vae_local, vae_local.quantize
        self.gpt_opt: AmpOptimizer = gpt_opt
        self.gpt_wo_ddp: Union[InfinityPilot, torch._dynamo.eval_frame.OptimizedModule] = gpt_wo_ddp  # after torch.compile
        self.gpt_wo_ddp_ema = gpt_wo_ddp_ema
        self.gpt_ema = gpt_ema
        self.bitwise_self_correction = BitwiseSelfCorrection(self.vae_local, other_args)
        self.use_fsdp_model_ema = use_fsdp_model_ema
        self.batch_size, self.seq_len = 0, 0
        self.seq_len_each = []
        self.reweight_loss_by_scale = reweight_loss_by_scale
        self.logging_warmup_steps = max(0, int(getattr(other_args, 'logging_warmup_steps', 10))) if other_args is not None else 10
        self.visualize_interval = max(1, int(getattr(other_args, 'visualize_interval', 100))) if other_args is not None else 100
        print(f'self.reweight_loss_by_scale: {self.reweight_loss_by_scale}')
        self.text_tokenizer = None
        self.text_encoder = None
        self.last_train_loss: Optional[float] = None
        base_scale_limit = getattr(other_args, 'always_training_scales', 1) if other_args is not None else 1
        self.base_training_scale_limit = max(1, int(base_scale_limit))
        initial_scales_cfg = getattr(other_args, 'initial_training_scales', 0) if other_args is not None else 0
        if initial_scales_cfg is None or int(initial_scales_cfg) <= 0:
            initial_scales_cfg = self.base_training_scale_limit
        initial_scales = max(1, int(initial_scales_cfg))
        initial_scales = min(initial_scales, self.base_training_scale_limit)
        self.active_training_scales = initial_scales
        self.last_training_scales = initial_scales
        self.dynamic_scale_manager: Optional[DynamicScaleManager] = None
        if other_args is not None and getattr(other_args, 'enable_dynamic_scales', False):
            target_scale_cfg = getattr(other_args, 'dynamic_scale_target', self.base_training_scale_limit)
            target_scale = max(1, int(target_scale_cfg)) if target_scale_cfg is not None else self.base_training_scale_limit
            target_scale = min(target_scale, self.base_training_scale_limit)
            target_scale = max(target_scale, self.active_training_scales)
            patience_transition = getattr(other_args, 'dynamic_scale_patience_transition', target_scale)
            patience_transition = max(1, int(patience_transition)) if patience_transition is not None else target_scale
            self.dynamic_scale_manager = DynamicScaleManager(
                initial_scale=self.active_training_scales,
                target_scale=target_scale,
                patience_low=getattr(other_args, 'dynamic_scale_patience_low', 3000),
                patience_high=getattr(other_args, 'dynamic_scale_patience_high', 4000),
                transition_scale=patience_transition,
                loss_delta=getattr(other_args, 'dynamic_scale_loss_delta', 1e-3),
                loss_window=getattr(other_args, 'dynamic_scale_loss_window', 200),
            )
        
        # Ensure CAR modules are initialized
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        if hasattr(gpt_uncompiled, 'has_car_modules'):
            assert gpt_uncompiled.has_car_modules(), "CAR modules must be initialized before constructing the trainer"
        
        self._car_gradient_setup = False
        self._setup_car_parameter_gradients()

        # Print parameter counts (after gradient setup to reflect actual trainable stats)
        car_params = sum(p.numel() for p in gpt_uncompiled.get_car_parameters() if p.requires_grad)
        infinity_params = sum(p.numel() for p in gpt_uncompiled.get_infinity_parameters() if p.requires_grad)
        print(f"Trainable CAR parameters: {car_params:,}")
        print(f"Trainable Infinity parameters: {infinity_params:,}")
        print(f"Total trainable parameters: {car_params + infinity_params:,}")
        
        self.using_ema = ema_ratio != 0 and self.zero == 0
        self.ema_ratio = abs(ema_ratio)
        self.ema_cpu = ema_ratio < 0
        self.is_visualizer = is_visualizer
        
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        del gpt_uncompiled.rng
        gpt_uncompiled.rng = torch.Generator(device=device)
        del gpt_uncompiled
        
        self.cached_state_not_ema = None
        if self.using_ema:
            self.pi_para_copy_for_parallel_ema = []
            all_tot = tot = 0
            for pi, para in enumerate(self.gpt_opt.paras):          # only learnable parameters need ema update
                if pi % dist.get_world_size() == dist.get_rank():   # model-parallel-style split
                    p_ema = para.data.cpu() if self.ema_cpu else para.data.clone()
                    self.pi_para_copy_for_parallel_ema.append((pi, p_ema))
                    tot += p_ema.numel()
                all_tot += para.numel()
            t = torch.zeros(dist.get_world_size())
            t[dist.get_rank()] = float(tot)
            dist.allreduce(t)
            t = [round(x) for x in t.tolist()]
            print(f'[ema tot #para] min={min(t)/1e6:.2f}, max={max(t)/1e6:.2f}, sum={sum(t)/1e6:.2f}, error={sum(t)-all_tot}')
            # lvl_1L, attn_bias_for_masking, zero_k_bias are never changed
            # check we only have these buffers so that we can skip buffer copy in ema update (only perform param update)
            assert all(any(s in name for s in ('lvl_1L', 'attn_bias_for_masking', 'zero_k_bias')) for name, _ in self.gpt_wo_ddp.named_buffers())
        else:
            self.pi_para_copy_for_parallel_ema = None
        
        self.label_smooth = label_smooth
        self.z_loss_ratio = z_loss_ratio
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='none')
        self.eq_loss = eq_loss
        
        # For raw_scale_schedule computation        
        self.raw_scale_schedule = raw_scale_schedule
        self.raw_L = sum(pn * pn for pn in raw_scale_schedule)
        self.raw_last_l = raw_scale_schedule[-1] * raw_scale_schedule[-1]
        
        if self.eq_loss:
            self.loss_eq_weight = torch.empty(1, self.raw_L, device=device)
            cur = 0
            for raw_pn in raw_scale_schedule:
                l = raw_pn*raw_pn
                self.loss_eq_weight[0, cur:cur+l] = 1./((raw_pn*raw_pn) if self.eq_loss == 2 else raw_pn)
                cur += l
            self.loss_eq_weight /= self.loss_eq_weight.sum()
        else:
            self.loss_eq_weight = 1.
        
        self.cmap_sim: ListedColormap = sns.color_palette('viridis', as_cmap=True)
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
        self.generator = np.random.default_rng(0)
        
        # Initialize parameter visualizer if available
        self.param_visualizer = None
        if False:
        # if VISUALIZER_AVAILABLE and dist.is_master() and False:
            try:
                # Get the underlying model without DDP/FSDP wrapper
                underlying_model = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
                self.param_visualizer = ParameterChangeVisualizer(
                    underlying_model, 
                    save_dir=f"./debug/param_visualizations_{time.strftime('%m%d_%H%M%S')}"
                )
                # print("✅ Parameter visualizer initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize parameter visualizer: {e}")
                self.param_visualizer = None
        
        # Initialize NaN detector if available
        self.nan_detector = None
        if NAN_DETECTOR_AVAILABLE and getattr(other_args, 'enable_nan_detector', True):
            try:
                underlying_model = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
                self.nan_detector = NaNDetector(underlying_model, check_backward=True, verbose=True)
                self.nan_detector.register_hooks()
                print("✅ NaN detector initialized with hooks on all normalization layers")
            except Exception as e:
                print(f"⚠️ Failed to initialize NaN detector: {e}")
                self.nan_detector = None

    def register_text_encoder(self, tokenizer, encoder):
        """Attach tokenizer and encoder for evaluation-time caption encoding."""
        self.text_tokenizer = tokenizer
        self.text_encoder = encoder

    def _setup_car_parameter_gradients(self):
        """Freeze Infinity parameters once so training only updates CAR components."""
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        if hasattr(gpt_uncompiled, 'has_car_modules'):
            assert gpt_uncompiled.has_car_modules(), "CAR modules must be initialized before setting gradients."
        if getattr(gpt_uncompiled, '_gradient_setup_done', False):
            self._car_gradient_setup = True
            return

        car_param_count = 0
        infinity_param_count = 0
        for name, param in gpt_uncompiled.named_parameters():
            if any(car_prefix in name for car_prefix in ['car_']):
                param.requires_grad = True
                car_param_count += 1
            else:
                param.requires_grad = False
                infinity_param_count += 1

        if dist.is_master():
            print(f"[Gradient Setup] CAR params (trainable): {car_param_count}, Infinity params (frozen): {infinity_param_count}")

        gpt_uncompiled._gradient_setup_done = True
        self._car_gradient_setup = True
    
    @torch.no_grad()
    def _generate_eval_visualization(self, ep: int, eval_images: List[torch.Tensor], 
                                   eval_conditions: List[Optional[Dict[str, torch.Tensor]]], eval_prompts: List[str],
                                   eval_text_conds: List[Optional[Tuple[torch.Tensor, List[int], torch.Tensor, int]]],
                                   args):
        """Generate visualization images during evaluation and log to wandb/tensorboard"""
        try:
            print(f"Generating evaluation visualization for epoch {ep}...")
            
            # Get underlying model without wrapper
            model = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
            
            visualization_images = []
            
            for i, (orig_img, condition, prompt, text_cond) in enumerate(zip(eval_images, eval_conditions, eval_prompts, eval_text_conds)):
                if i >= 4:  # Limit to 4 samples to avoid memory issues
                    break
                    
                if text_cond is None:
                    print(f"[warn] Missing text condition for visualization sample {i}, skipping.")
                    continue
                
                try:
                    device = torch.device(args.device) if not isinstance(args.device, torch.device) else args.device
                    # Prepare inputs
                    orig_img_cpu = orig_img
                    condition_cpu = condition
                    condition_gpu = None
                    if condition_cpu is not None:
                        condition_gpu = {k: v.to(device) for k, v in condition_cpu.items() if v is not None}
                    
                    # Determine scale schedule
                    h_div_w = orig_img.shape[-2] / orig_img.shape[-1]
                    h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
                    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
                    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                    if getattr(args, 'apply_spatial_patchify', False):
                        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
                    else:
                        vae_scale_schedule = scale_schedule
                    
                    text_kv, text_lens, text_cu, text_max = text_cond
                    text_tuple_device = (
                        text_kv.to(device),
                        text_lens,
                        text_cu.to(device),
                        text_max,
                    )

                    control_tokens = None
                    if condition_gpu:
                        control_tokens = self._build_control_tokens(condition_gpu, vae_scale_schedule, len(scale_schedule))
                        if control_tokens is not None:
                            control_tokens = [t.to(device) if t is not None else None for t in control_tokens]

                    cfg_list = [1.0] * len(scale_schedule)
                    tau_list = [1.0] * len(scale_schedule)
                    top_k = getattr(args, 'vis_top_k', getattr(model, 'top_k', 0))
                    top_p = getattr(args, 'vis_top_p', getattr(model, 'top_p', 0.0))

                    with torch.no_grad():
                        _, _, generated_imgs = model.autoregressive_infer_cfg(
                            vae=self.vae_local,
                            scale_schedule=scale_schedule,
                            label_B_or_BLT=text_tuple_device,
                            B=1,
                            cfg_list=cfg_list,
                            tau_list=tau_list,
                            top_k=top_k,
                            top_p=top_p,
                            ret_img=True,
                            inference_mode=True,
                            control_tokens=control_tokens,
                        )

                    if generated_imgs is None or generated_imgs.shape[0] == 0:
                        print(f"[warn] No generated image returned for sample {i}")
                        continue
                    generated_img = generated_imgs[0]
                    
                    # Create comparison grid: [Original, Condition (if exists), Generated]
                    comparison_images = []
                    
                    # Original image (normalize to [0,1])
                    orig_display = (orig_img_cpu.squeeze(0) + 1) / 2
                    comparison_images.append(orig_display.detach().cpu())
                    
                    # Condition image (if exists)
                    if condition_cpu is not None:
                        for key, cond_tensor in condition_cpu.items():
                            cond_display = (cond_tensor.squeeze(0) + 1) / 2
                            comparison_images.append(cond_display.detach().cpu())
                    
                    # Generated image (normalize to [0,1])
                    if isinstance(generated_img, torch.Tensor):
                        generated_display = generated_img.float().permute(2, 0, 1) / 255.0
                    else:
                        generated_display = torch.from_numpy(generated_img).permute(2, 0, 1).float() / 255.0
                    generated_display = generated_display.flip(0)
                    comparison_images.append(generated_display.clamp(0, 1).cpu())
                    
                    # Stack horizontally
                    comparison_grid = torch.cat(comparison_images, dim=2)  # Concatenate along width
                    visualization_images.append(comparison_grid)
                    
                except Exception as e:
                    print(f"Error generating sample {i}: {e}")
                    continue
            
            if len(visualization_images) > 0:
                # Create final grid (stack vertically)
                final_grid = torch.stack(visualization_images, dim=0)  # [N, C, H, W*3]
                
                # Log to wandb if available
                try:
                    if hasattr(wandb_utils, 'log_image'):
                        wandb_utils.log_image(f"eval_generation_epoch_{ep}", final_grid, step=ep)
                    elif hasattr(wandb_utils, 'log'):
                        # Convert to numpy for wandb
                        grid_np = final_grid.permute(0, 2, 3, 1).cpu().numpy()
                        wandb_utils.log({
                            f"eval_generation_epoch_{ep}": [wandb_utils.wandb.Image(img) for img in grid_np]
                        }, step=ep)
                except Exception as e:
                    print(f"Failed to log to wandb: {e}")
                
                # Log to tensorboard if available
                try:
                    if hasattr(self, 'tb_logger') and self.tb_logger is not None:
                        for idx, img in enumerate(final_grid):
                            self.tb_logger.log_image(f"eval_generation_epoch_{ep}_sample_{idx}", img, step=ep)
                except Exception as e:
                    print(f"Failed to log to tensorboard: {e}")
                
                # Save as image files
                try:
                    import os
                    save_dir = f"./eval_visualizations_epoch_{ep}"
                    os.makedirs(save_dir, exist_ok=True)
                    
                    for idx, img in enumerate(final_grid):
                        # Convert to PIL and save
                        img_pil = torch.clamp(img.permute(1, 2, 0), 0, 1) * 255
                        img_np = img_pil.cpu().numpy().astype(np.uint8)
                        cv2.imwrite(f"{save_dir}/sample_{idx}.jpg", img_np[:, :, ::-1])  # BGR for cv2
                    
                    print(f"Saved {len(final_grid)} evaluation visualizations to {save_dir}")
                    
                except Exception as e:
                    print(f"Failed to save images: {e}")
                    
        except Exception as e:
            print(f"Error in evaluation visualization: {e}")
            import traceback
            traceback.print_exc()

    @torch.no_grad()
    def eval_ep(self, ep: int, args: arg_util.Args, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.gpt_wo_ddp.training
        self.gpt_wo_ddp.eval()
        
        # For visualization 
        eval_images = []
        eval_conditions = []
        eval_prompts = []
        eval_text_conds = []
        max_vis_samples = 8  # Limit number of visualization samples

        for batch_idx, data in enumerate(ld_val):
            if batch_idx >= max_vis_samples:
                break # [DEBUG] limit eval batches for faster testing
            # Handle data format for pilot (with condition)
            condition_inputs = {}
            if len(data) == 4:
                inp, condition_mask, condition_normal, label_B = data
            elif len(data) == 3:
                inp, condition_normal, label_B = data
                condition_mask = None
            else:
                inp, label_B = data
                condition_mask = None
                condition_normal = None

            B = inp.shape[0]
            text_cond_tuple = label_B
            if isinstance(label_B, torch.Tensor):
                text_cond_tuple = label_B.to(args.device, non_blocking=True)
            elif isinstance(label_B, (list, tuple)):
                if len(label_B) == 4 and isinstance(label_B[0], torch.Tensor):
                    kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B
                    text_cond_tuple = (
                        kv_compact.to(args.device, non_blocking=True),
                        lens,
                        cu_seqlens_k.to(args.device, non_blocking=True),
                        max_seqlen_k,
                    )
                elif len(label_B) > 0 and isinstance(label_B[0], str):
                    if self.text_tokenizer is None or self.text_encoder is None:
                        raise RuntimeError('Text tokenizer/encoder must be registered before evaluation.')
                    captions = list(label_B)
                    tokens = self.text_tokenizer(
                        text=captions,
                        max_length=self.text_tokenizer.model_max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt',
                    )
                    input_ids = tokens.input_ids.to(args.device, non_blocking=True)
                    mask = tokens.attention_mask.to(args.device, non_blocking=True)
                    text_features = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=mask,
                    )['last_hidden_state'].float()
                    lens: List[int] = mask.sum(dim=-1).tolist()
                    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
                    max_seqlen_k = max(lens) if lens else 0
                    kv_pieces = [feat_i[:len_i] for len_i, feat_i in zip(lens, text_features.unbind(0))]
                    kv_compact = torch.cat(kv_pieces, dim=0) if kv_pieces else text_features.new_zeros(0, text_features.shape[-1])
                    text_cond_tuple = (kv_compact, lens, cu_seqlens_k, max_seqlen_k)
                else:
                    text_cond_tuple = label_B
            else:
                text_cond_tuple = label_B
            use_bit_label = getattr(self.gpt_wo_ddp, 'use_bit_label', False)
            V = getattr(self.gpt_wo_ddp, 'V', self.vae_local.vocab_size)
            inp = inp.to(args.device, non_blocking=True)
            if condition_normal is not None:
                condition_normal = condition_normal.to(args.device, non_blocking=True)
                condition_inputs['normal'] = condition_normal
            if condition_mask is not None:
                condition_mask = condition_mask.to(args.device, non_blocking=True)
                condition_inputs['mask'] = condition_mask
            
            h_div_w = inp.shape[-2] / inp.shape[-1]
            h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
            h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
            full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
            with torch.no_grad():
                if args.apply_spatial_patchify:
                    vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in full_scale_schedule]
                else:
                    vae_scale_schedule = full_scale_schedule
                raw_features, _, _ = self.vae_local.encode_for_raw_features(inp, scale_schedule=vae_scale_schedule)
                x_BLC_wo_prefix_full, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(
                    vae_scale_schedule=vae_scale_schedule,
                    inp_B3HW=inp,
                    raw_features=raw_features,
                    device=inp.device,
                )

            available_scales = min(len(full_scale_schedule), len(gt_ms_idx_Bl))
            if available_scales == 0:
                continue
            full_scale_schedule = full_scale_schedule[:available_scales]
            gt_ms_idx_Bl = gt_ms_idx_Bl[:available_scales]
            
            full_control_tokens = None
            raw_features_condition_mask = None
            raw_features_condition_normal = None
            if condition_inputs:
                with torch.no_grad():
                    if args.apply_spatial_patchify:
                        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in full_scale_schedule]
                    else:
                        vae_scale_schedule = full_scale_schedule
                    if condition_inputs.get('mask') is not None:
                        raw_features_condition_mask, _, _ = self.vae_local.encode_for_raw_features(condition_inputs['mask'], scale_schedule=vae_scale_schedule)
                    if condition_inputs.get('normal') is not None:
                        raw_features_condition_normal, _, _ = self.vae_local.encode_for_raw_features(condition_inputs['normal'], scale_schedule=vae_scale_schedule)
                control_tokens_by_type = {}
                if raw_features_condition_mask is not None:
                    control_tokens_by_type['mask'] = self.bitwise_self_correction.requant(vae_scale_schedule, raw_features_condition_mask)
                if raw_features_condition_normal is not None:
                    control_tokens_by_type['normal'] = self.bitwise_self_correction.requant(vae_scale_schedule, raw_features_condition_normal)
                if control_tokens_by_type:
                    full_control_tokens = self._combine_control_tokens(control_tokens_by_type, len(full_scale_schedule))
            
            scale_limit = min(self.base_training_scale_limit, available_scales)
            training_scales = min(scale_limit, self.active_training_scales)
            if training_scales <= 0:
                continue
            self.last_training_scales = training_scales
            scale_schedule = full_scale_schedule[:training_scales]
            gt_ms_idx_Bl = gt_ms_idx_Bl[:training_scales]
            control_tokens = None
            if full_control_tokens is not None:
                control_tokens = full_control_tokens[:training_scales]
            training_seq_len = np.array(scale_schedule).prod(axis=1).sum()
            first_scale_tokens = int(np.prod(scale_schedule[0]))
            x_BLC_wo_prefix = x_BLC_wo_prefix_full[:, :(training_seq_len-first_scale_tokens), :]
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1).to(x_BLC_wo_prefix.device, dtype=torch.long)

            self.gpt_wo_ddp.forward
            logits_BLV = self.gpt_wo_ddp(text_cond_tuple, x_BLC_wo_prefix, 
                                        scale_schedule=scale_schedule, control_tokens=control_tokens)
            
            last_scale_area = int(np.prod(scale_schedule[-1]))
            B, seq_len = logits_BLV.shape[:2]
            if use_bit_label:
                logits_bits = logits_BLV.reshape(B, seq_len, -1, 2)
                ce_bits = self.val_loss(logits_bits.permute(0, 3, 1, 2), gt_BL)
                bitloss_type = getattr(args, 'bitloss_type', 'mean')
                if bitloss_type == 'mean':
                    token_loss = ce_bits.mean(dim=-1)
                elif bitloss_type == 'sum':
                    token_loss = ce_bits.sum(dim=-1)
                else:
                    raise NotImplementedError(f'{bitloss_type=}')
                per_sample_loss = token_loss.mean(dim=-1)
                tail_loss = token_loss[:, -last_scale_area:].mean(dim=-1)
                L_mean += per_sample_loss.sum().item()
                L_tail += tail_loss.sum().item()

                bitwise_acc = (logits_bits.argmax(dim=-1) == gt_BL).float()
                token_bit_acc = bitwise_acc.mean(dim=-1)
                per_sample_bit_acc = token_bit_acc.mean(dim=-1)
                tail_bit_acc = token_bit_acc[:, -last_scale_area:].mean(dim=-1)
                acc_mean += per_sample_bit_acc.sum().item() * 100.0
                acc_tail += tail_bit_acc.sum().item() * 100.0
            else:
                logits_view = logits_BLV.reshape(-1, V)
                token_loss = self.val_loss(logits_view, gt_BL.reshape(-1)).view(B, -1)
                per_sample_loss = token_loss.mean(dim=-1)
                tail_loss = token_loss[:, -last_scale_area:].mean(dim=-1)
                L_mean += per_sample_loss.sum().item()
                L_tail += tail_loss.sum().item()

                pred_BL = logits_BLV.argmax(dim=-1)
                per_sample_acc = (pred_BL == gt_BL).float().mean(dim=-1)
                tail_acc = (pred_BL[:, -last_scale_area:] == gt_BL[:, -last_scale_area:]).float().mean(dim=-1)
                acc_mean += per_sample_acc.sum().item() * 100.0
                acc_tail += tail_acc.sum().item() * 100.0
            tot += B
            
            # Collect samples for visualization (only from master process and first few batches)
            if dist.is_master() and len(eval_images) < max_vis_samples and batch_idx < 4:
                # Store original images, conditions, and generate samples for visualization
                remaining_slots = max_vis_samples - len(eval_images)
                num_samples = min(B, remaining_slots)
                for i in range(num_samples):
                    eval_images.append(inp[i:i+1].cpu())  # Keep original image
                    if condition_inputs:
                        sample_condition = {k: v[i:i+1].cpu() for k, v in condition_inputs.items()}
                    else:
                        sample_condition = None
                    eval_conditions.append(sample_condition)
                    eval_prompts.append(f"val_sample_{len(eval_prompts)}")
                    eval_text_conds.append(self._slice_text_cond_for_eval(text_cond_tuple, i))
                    
        self.gpt_wo_ddp.train(training)
        
        device = torch.device(args.device)
        stats = torch.tensor([L_mean, L_tail, acc_mean, acc_tail, float(tot)], device=device)
        dist.allreduce(stats)
        tot_float = stats[-1].item()
        if tot_float == 0:
            return 0.0, 0.0, 0.0, 0.0, 0, time.time() - stt
        tot = int(round(tot_float))
        stats = stats[:-1] / tot
        L_mean, L_tail, acc_mean, acc_tail = stats.tolist()
        
        # Generate visualization images during evaluation
        if dist.is_master() and len(eval_images) > 0:
            self._generate_eval_visualization(ep, eval_images, eval_conditions, eval_prompts, eval_text_conds, args)
        
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt

    def _build_control_tokens(self, condition_inputs: Optional[Dict[str, torch.Tensor]], vae_scale_schedule, training_scales):
        if condition_inputs is None or len(condition_inputs) == 0:
            return None
        control_tokens_by_type = {}
        with torch.no_grad():
            if condition_inputs.get('mask') is not None:
                mask = condition_inputs['mask']
                raw_features_mask, _, _ = self.vae_local.encode_for_raw_features(mask, scale_schedule=vae_scale_schedule)
                control_tokens_by_type['mask'] = self.bitwise_self_correction.requant(vae_scale_schedule, raw_features_mask)
            if condition_inputs.get('normal') is not None:
                normal = condition_inputs['normal']
                raw_features_normal, _, _ = self.vae_local.encode_for_raw_features(normal, scale_schedule=vae_scale_schedule)
                control_tokens_by_type['normal'] = self.bitwise_self_correction.requant(vae_scale_schedule, raw_features_normal)
        if not control_tokens_by_type:
            return None
        return self._combine_control_tokens(control_tokens_by_type, training_scales)

    @staticmethod
    def _combine_control_tokens(control_tokens_by_type: Dict[str, List[torch.Tensor]], training_scales: int):
        if not control_tokens_by_type:
            return None
        control_tokens = []
        for si in range(training_scales):
            tokens_at_scale = []
            for tokens_list in control_tokens_by_type.values():
                if si < len(tokens_list):
                    tokens_at_scale.append(tokens_list[si])
            if tokens_at_scale:
                if len(tokens_at_scale) == 1:
                    control_tokens.append(tokens_at_scale[0])
                else:
                    stacked = torch.stack(tokens_at_scale, dim=0)
                    control_tokens.append(stacked.mean(dim=0))
            else:
                control_tokens.append(None)
        return control_tokens
    
    @staticmethod
    def _slice_text_cond_for_eval(text_tuple, index: int) -> Optional[Tuple[torch.Tensor, List[int], torch.Tensor, int]]:
        if not isinstance(text_tuple, tuple) or len(text_tuple) != 4:
            return None
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = text_tuple
        if isinstance(lens, torch.Tensor):
            lens_list = lens.tolist()
        else:
            lens_list = list(lens)
        if index >= len(lens_list):
            return None
        start = sum(int(l) for l in lens_list[:index])
        length = int(lens_list[index])
        kv_slice = kv_compact[start:start+length].detach().cpu()
        cu = torch.tensor([0, length], dtype=torch.int32)
        return (kv_slice, [length], cu, length)
    
    def train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float,
        metric_lg: misc.MetricLogger, logging_params: bool,
        inp_B3HW: FTen, condition_inputs: Optional[Dict[str, FTen]], text_cond_tuple: Union[ITen, FTen], args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        
        B = inp_B3HW.shape[0]  # if isinstance(inp_B3HW, torch.Tensor) else inp_B3HW[0].shape[0]
        T = 1 if inp_B3HW.dim() == 4 else inp_B3HW.shape[2]
        V = self.vae_local.vocab_size
        device = inp_B3HW.device
        warmup_steps = self.logging_warmup_steps
        should_visualize_batch = (it % 10 == 0 and dist.is_master())
        full_gt_ms_idx_Bl = None

        h_div_w = inp_B3HW.shape[-2] / inp_B3HW.shape[-1]
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        full_scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        full_scale_schedule = [(min(t, T//4+1), h, w) for (t, h, w) in full_scale_schedule]
        

        # [forward]
        
        # # Check input for NaN/Inf before forward pass
        # if g_it < 50 and torch.isnan(inp_B3HW).any():
        #     print(f"❌ [it={it}] NaN in input inp_B3HW!")
        #     print(f"   Shape: {inp_B3HW.shape}, Device: {inp_B3HW.device}")
        # if g_it < 50 and torch.isinf(inp_B3HW).any():
        #     print(f"❌ [it={it}] Inf in input inp_B3HW!")
        
        with self.gpt_opt.amp_ctx:
            # with torch.amp.autocast('cuda', enabled=False):
            raw_features_condition_mask = None
            raw_features_condition_normal = None
            with torch.no_grad():
                if args.apply_spatial_patchify:
                    full_vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in full_scale_schedule]
                else:
                    full_vae_scale_schedule = full_scale_schedule
                vae_scale_schedule = full_vae_scale_schedule
                raw_features, _, _ = self.vae_local.encode_for_raw_features(inp_B3HW, scale_schedule=vae_scale_schedule)
                # take out normal map and text mask condition if exists
                # print('condition_inputs:', condition_inputs)
                if condition_inputs is not None:
                    if condition_inputs.get('mask') is not None:
                        condition_mask = condition_inputs.get('mask')
                        raw_features_condition_mask, _, _ = self.vae_local.encode_for_raw_features(condition_mask, scale_schedule=vae_scale_schedule) 
                    if condition_inputs.get('normal') is not None:
                        condition_normal = condition_inputs.get('normal')
                        raw_features_condition_normal, _, _ = self.vae_local.encode_for_raw_features(condition_normal, scale_schedule=vae_scale_schedule)



            x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(vae_scale_schedule, inp_B3HW, raw_features, device)
            if should_visualize_batch:
                full_gt_ms_idx_Bl = [tensor.clone() for tensor in gt_ms_idx_Bl]
            control_tokens_by_type = {}
            if raw_features_condition_mask is not None:
                control_tokens_by_type['mask'] = self.bitwise_self_correction.requant(vae_scale_schedule, raw_features_condition_mask)
            if raw_features_condition_normal is not None:
                control_tokens_by_type['normal'] = self.bitwise_self_correction.requant(vae_scale_schedule, raw_features_condition_normal)
            full_control_tokens = None
            if control_tokens_by_type:
                full_control_tokens = self._combine_control_tokens(control_tokens_by_type, len(full_scale_schedule))

            # truncate scales
            available_scales = min(len(full_scale_schedule), len(gt_ms_idx_Bl))
            scale_limit = min(self.base_training_scale_limit, available_scales)
            training_scales = min(scale_limit, self.active_training_scales)
            if training_scales <= 0:
                raise RuntimeError("No valid scales available; check VAE outputs or dynamic resolution settings.")
            self.last_training_scales = training_scales
            scale_schedule = full_scale_schedule[:training_scales]
            gt_ms_idx_Bl = gt_ms_idx_Bl[:training_scales]
            control_tokens = None
            if full_control_tokens is not None:
                control_tokens = full_control_tokens[:training_scales]
            training_seq_len = np.array(scale_schedule).prod(axis=1).sum()
            x_BLC_wo_prefix = x_BLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()), :]


            # [forward]
            self.gpt_wo_ddp.forward  
            

            if not self._car_gradient_setup:
                self._setup_car_parameter_gradients()
            # print(f'scale_schedule: {scale_schedule}')
            logits_BLV = self.gpt(text_cond_tuple, x_BLC_wo_prefix, scale_schedule=scale_schedule, control_tokens=control_tokens) # [bs, 1*1+...+64*64, vocab_size or log2(vocab_size)*2]
            self.batch_size, self.seq_len = logits_BLV.shape[:2]

            self.seq_len_each = [idx_Bl.shape[1] for idx_Bl in gt_ms_idx_Bl]
            
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:,:training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]

            if should_visualize_batch and full_gt_ms_idx_Bl is not None and args.use_bit_label:
                pred_ms_idx_Bl: List[torch.Tensor] = []
                seq_len_vis = logits_BLV.shape[1]
                logits_bits = logits_BLV.reshape(B, seq_len_vis, -1, 2)
                pred_bits = logits_bits.argmax(dim=-1)

                cursor = 0
                for scale_tensor in gt_ms_idx_Bl:
                    scale_len = scale_tensor.shape[1]
                    pred_slice = pred_bits[:, cursor:cursor + scale_len]
                    pred_ms_idx_Bl.append(pred_slice.contiguous().reshape(B, scale_len, scale_tensor.shape[-1]))
                    cursor += scale_len

                self.generate_training_visualization(
                    ep,
                    it,
                    g_it,
                    inp_B3HW,
                    raw_features,
                    condition_inputs,
                    gt_ms_idx_Bl,
                    pred_ms_idx_Bl,
                    scale_schedule,
                    training_scales,
                    training_seq_len,
                    full_gt_ms_idx_Bl,
                    full_scale_schedule,
                    full_vae_scale_schedule,
                )

            
            if args.use_bit_label:
                tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
                raw_loss = self.train_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
                if args.bitloss_type == 'mean':
                    raw_loss = raw_loss.mean(dim=-1)
                elif args.bitloss_type == 'sum':
                    raw_loss = raw_loss.sum(dim=-1)
                else:
                    raise NotImplementedError(f'{args.bitloss_type=}')
            else:
                raw_loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1)


            if self.reweight_loss_by_scale:
                lw = []
                last_scale_area = np.sqrt(np.prod(scale_schedule[-1]))
                # print(f"last_scale_area: {last_scale_area}")
                # print(f"scale_schedule: {scale_schedule}")
                for (pt, ph, pw) in scale_schedule:
                    this_scale_area = np.sqrt(pt * ph * pw)
                    lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw)])
                lw = torch.tensor(lw, device=raw_loss.device, dtype=raw_loss.dtype).unsqueeze(0)
                
                total_w = lw.sum()
                
                lw = lw / total_w
                weighted_loss = raw_loss.mul(lw)
            else:
                print(f"else: {self.seq_len}")
                lw = 1. / self.seq_len
                weighted_loss = raw_loss * lw

            loss = weighted_loss.sum(dim=-1).mean()
            self.last_train_loss = float(loss.detach().item())

        # [backward]
        # Check loss before backward
        # if torch.isnan(loss).any() or torch.isinf(loss).any():
        #     print(f"❌ [it={it}] Loss is NaN/Inf before backward: {loss.item()}")
        #     if self.nan_detector:
        #         print(self.nan_detector.get_summary())
        #     raise RuntimeError(f"Loss is NaN/Inf at it={it}")
        
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=ep, it=it, g_it=g_it, stepping=stepping, logging_params=logging_params, loss=loss, clip_decay_ratio=clip_decay_ratio, stable=args.stable)
        
        # # Check if NaN detector caught anything
        # if self.nan_detector and self.nan_detector.nan_layers and g_it < 100:
        #     print(f"\n{'='*80}")
        #     print(f"🔴 NaN DETECTION SUMMARY at it={it}, g_it={g_it}")
        #     print(self.nan_detector.get_summary())
        #     print(f"{'='*80}\n")
        #     # Clear for next iteration
        #     self.nan_detector.clear()
        
        # update ema
        if args.use_fsdp_model_ema:
            update_ema(self.gpt_ema, self.gpt)

        # [zero_grad]
        if stepping:
            grad_stats = {}
             # 梯度監控
            if dist.is_master(): # 只在主進程執行
                import matplotlib.pyplot as plt
                import seaborn as sns
                import pandas as pd
                import io
                from PIL import Image
                model_to_check = self.gpt_wo_ddp
                if hasattr(model_to_check, '_orig_mod'):
                    model_to_check = model_to_check._orig_mod
                
                # 創建一個字典來收集所有梯度統計
                grad_data = []
                def collect_grad_stats(name, module_or_param):
                    if isinstance(module_or_param, torch.nn.Module):
                        params = module_or_param.parameters()
                    else:
                        params = [module_or_param]

                    grads = [p.grad for p in params if p.grad is not None]
                    
                    mean_abs_grad = 0.0
                    if grads:
                        all_grads = torch.cat([g.detach().abs().view(-1) for g in grads])
                        mean_abs_grad = all_grads.mean().item()
                    
                    grad_data.append({"module": name, "mean_abs_grad": mean_abs_grad})

                # 1. 收集所有 CAR 模塊的梯度數據
                if hasattr(model_to_check, 'car_control_proj'):
                    collect_grad_stats('ControlProj', model_to_check.car_control_proj)
                if hasattr(model_to_check, 'car_blocks'):
                    for i, car_block in enumerate(model_to_check.car_blocks):
                        collect_grad_stats(f'Block_{i:02d}', car_block)
                if hasattr(model_to_check, 'car_output_norms'):
                    for i, norm_layer in enumerate(model_to_check.car_output_norms):
                        collect_grad_stats(f'OutputNorm_{i:02d}', norm_layer)
                if hasattr(model_to_check, 'car_fusion_linears'):
                    for i, linear_layer in enumerate(model_to_check.car_fusion_linears):
                        collect_grad_stats(f'FusionLinear_{i:02d}', linear_layer)
                if hasattr(model_to_check, 'car_fusion_scales'):
                    collect_grad_stats('FusionScales', model_to_check.car_fusion_scales)
                if hasattr(model_to_check, 'car_fusion_norms'):
                    collect_grad_stats('FusionNorms', model_to_check.car_fusion_norms)

                # 2. 如果收集到了數據，則繪圖並上傳
                if grad_data:
                    # 將數據轉換為 pandas DataFrame 以便繪圖
                    df = pd.DataFrame(grad_data)
                    
                    # 創建圖表
                    plt.style.use('seaborn-v0_8-whitegrid')
                    # 根據模塊數量動態調整圖表高度
                    fig_height = max(8, len(grad_data) * 0.3)
                    fig, ax = plt.subplots(figsize=(12, fig_height))
                    
                    # 繪製水平條形圖，梯度值大的在上面
                    df_sorted = df.sort_values("mean_abs_grad", ascending=False)
                    sns.barplot(x="mean_abs_grad", y="module", data=df_sorted, ax=ax, palette="viridis_r")
                    
                    # 美化圖表
                    ax.set_title(f'CAR Modules Mean Absolute Gradient (Global Step: {g_it})', fontsize=16)
                    ax.set_xlabel('Mean Absolute Gradient (log scale)', fontsize=12)
                    ax.set_ylabel('Module', fontsize=12)
                    ax.set_xscale('log') # 使用對數尺度，以便清晰地比較不同數量級的梯度

                    # 在條形上顯示數值，使用科學記數法
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.2e', padding=5, fontsize=9)

                    plt.tight_layout()
                    
                    # 將圖表保存到內存緩衝區
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=120) # 提高 dpi 以獲得更清晰的圖像
                    buf.seek(0)
                    
                    # 使用 PIL 從緩衝區讀取圖像
                    img = Image.open(buf)
                    
                    # 上傳到 WandB
                    wandb_utils.log({"Chart/CAR_Gradient_Distribution": wandb_utils.wandb.Image(img)})
                    
                    # 關閉圖表以釋放內存
                    plt.close(fig)
                    buf.close()
                    
            if self.using_ema: pass # self.ema_update(g_it)
            if self.dbg_unused:
                ls = []
                for n, p in self.gpt_wo_ddp.named_parameters():
                    if p.grad is None:
                        ls.append(n)
                if len(ls):
                    raise AttributeError(f'unused param: {ls}')
        
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)
        
        # [metric logging]
        should_log_iter = metric_lg.log_every_iter or it in metric_lg.log_iters or it == warmup_steps
        if it >= warmup_steps and should_log_iter:
            B, seq_len = logits_BLV.shape[:2]
            if args.use_bit_label:
                res_loss = self.train_loss(logits_BLV.reshape(B, seq_len, -1, 2).permute(0,3,1,2), gt_BL).mean(dim=-1).mean(0)
                bitwise_acc = (logits_BLV.reshape(B, seq_len, -1, 2).argmax(dim=-1) == gt_BL).float() # shape: [bs, seq_len, codebook_dim]
            else:
                res_loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1).mean(0)
                pred_BL = logits_BLV.argmax(dim=-1)
                mask = self.vae_local.quantizer.lfq.mask
                pred_bits = ((pred_BL[..., None].int() & mask) != 0)
                gt_bits = ((gt_BL[..., None].int() & mask) != 0)
                bitwise_acc = (pred_bits == gt_bits).float() # shape: [bs, seq_len, codebook_dim]
            res_bit_acc = bitwise_acc.mean(-1).mean(0)
            res_token_acc = (bitwise_acc.sum(-1) == self.vae_local.codebook_dim).float().mean(0)
            
            loss_token_mean, acc_bit_mean, acc_token_mean = res_loss.mean().item(), res_bit_acc.mean().item() * 100., res_token_acc.mean().item() * 100.
            ptr = 0
            L_list, acc_bit_list, acc_token_list = [], [], []
            for scale_ind in range(min(training_scales, len(scale_schedule))):
                start, end = ptr, ptr + np.prod(scale_schedule[scale_ind])
                L_list.append(res_loss[start:end].mean().item())
                acc_bit_list.append(res_bit_acc[start:end].mean().item() * 100.)
                acc_token_list.append(res_token_acc[start:end].mean().item() * 100.)
                ptr = end
            
            metrics = torch.tensor(L_list + acc_bit_list + acc_token_list +[grad_norm_t.item(), loss_token_mean, acc_bit_mean, acc_token_mean], device=loss.device)
            tdist.all_reduce(metrics, op=tdist.ReduceOp.SUM)
            metrics = metrics.cpu().data.numpy() / dist.get_world_size()
            leng = len(L_list)
            L_list, acc_bit_list, acc_token_list, grad_norm_t, loss_token_mean, acc_bit_mean, acc_token_mean = metrics[:leng], \
                metrics[leng:2*leng], metrics[2*leng:3*leng], metrics[-4], metrics[-3], metrics[-2], metrics[-1]
            Lmean = loss_token_mean
            Ltail = L_list[-1]
            acc_mean = acc_bit_mean if args.use_bit_label else acc_token_mean
            acc_tail = acc_bit_list[-1] if args.use_bit_label else acc_token_list[-1]
            metric_lg.update(Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, tnm=grad_norm_t)    # todo: Accm, Acct
            wandb_log_dict = {"Overall/L_mean": Lmean, 'Overall/Acc_bit_mean': acc_bit_mean, 'Overall/Acc_token_mean': acc_token_mean, 'Overall/grad_norm_t': grad_norm_t}
            for si, (loss_si, acc_bit_si, acc_token_si) in enumerate(zip(L_list, acc_bit_list, acc_token_list)):
                wandb_log_dict[f'Detail/L_s{si+1:02d}'] = loss_si
                wandb_log_dict[f'Detail/Acc_bit_s{si+1:02d}'] = acc_bit_si
                wandb_log_dict[f'Detail/Acc_token_s{si+1:02d}'] = acc_token_si
            
            # Add parameter monitoring to WandB
            if stepping and self.param_visualizer is not None and False:
                try:
                    # Get current parameter changes for WandB logging
                    module_changes = defaultdict(list)
                    module_grads = defaultdict(list)
                    
                    gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
                    for name, param in gpt_uncompiled.named_parameters():
                        if param.requires_grad:
                            # Get parameter change from visualizer
                            change_mag = self.param_visualizer._get_param_change_magnitude(name, param.data)
                            module = self.param_visualizer._categorize_parameter(name)
                            module_changes[module].append(change_mag)
                            
                            # Get gradient magnitude
                            if param.grad is not None:
                                grad_norm = torch.norm(param.grad).item()
                                module_grads[module].append(grad_norm)
                    
                    # Log module-level statistics to WandB
                    for module in module_changes:
                        if module_changes[module]:
                            mean_change = np.mean(module_changes[module])
                            max_change = np.max(module_changes[module])
                            wandb_log_dict[f'ParamChange/{module}/mean'] = mean_change
                            wandb_log_dict[f'ParamChange/{module}/max'] = max_change
                            
                            # Check freeze status
                            is_frozen = all(change < 1e-8 for change in module_changes[module])
                            wandb_log_dict[f'FreezeStatus/{module}'] = 1.0 if is_frozen else 0.0
                        
                        if module in module_grads and module_grads[module]:
                            mean_grad = np.mean(module_grads[module])
                            wandb_log_dict[f'GradNorm/{module}/mean'] = mean_grad
                
                except Exception as e:
                    print(f"⚠️ WandB parameter logging error: {e}")
            
            # wandb_utils.log(wandb_log_dict, step=g_it)
            # 如果收集到了統計數據，就用 wandb 記錄
            if grad_stats:
                wandb_utils.log(grad_stats, step=g_it) # 使用全局步數 g_it
            wandb_utils.log(wandb_log_dict)

        return grad_norm_t, scale_log2_t

    def maybe_update_training_scales(self, g_it: int, args: Optional[arg_util.Args] = None) -> Optional[int]:
        if self.dynamic_scale_manager is None:
            return None
        if self.last_train_loss is None:
            return None
        new_scale = self.dynamic_scale_manager.register_step(g_it, self.last_train_loss)
        if new_scale is None:
            return None
        clamped_scale = min(self.base_training_scale_limit, max(1, int(new_scale)))
        self.active_training_scales = clamped_scale
        self.last_training_scales = clamped_scale
        return clamped_scale

    def __repr__(self):
        return (
            f'\n'
            f'[VGPTTr.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[VGPTTr.structure]: {super(InfinityPilotTrainer, self).__repr__().replace(InfinityPilotTrainer.__name__, "")}'
        )
    
    def ema_load(self):
        self.cached_state_not_ema = {k: v.cpu() for k, v in self.gpt_wo_ddp.state_dict().items()}
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            self.gpt_opt.paras[pi].data.copy_(p_ema)
        for pi, para in enumerate(self.gpt_opt.paras):
            dist.broadcast(para, src_rank=pi % dist.get_world_size())
    
    def ema_recover(self):
        self.gpt_wo_ddp.load_state_dict(self.cached_state_not_ema)
        del self.cached_state_not_ema
        self.cached_state_not_ema = None
    
    # p_ema = p_ema*0.9 + p*0.1 <==> p_ema.lerp_(p, 0.1)
    # p_ema.mul_(self.ema_ratio).add_(p.mul(self.ema_ratio_1))
    # @profile(precision=4, stream=open('ema_update.log', 'w+'))
    def ema_update(self, g_it): # todo: 将来再用离线ema
        # if self.using_ema and (g_it + 1) in self.ema_upd_it:
        stt = time.time()
        for pi, p_ema in self.pi_para_copy_for_parallel_ema:
            p = self.gpt_opt.paras[pi]
            p_ema.data.mul_(self.ema_ratio).add_(p.data.to(p_ema.device), alpha=1-self.ema_ratio)
        # ii = self.ema_upd_it.index(g_it + 1)
        ii = g_it
        if ii < 3:
            print(f'[ema upd {self.ema_ratio}, cpu={self.ema_cpu}, @ g_it={g_it}] cost: {time.time()-stt:.2f}s')
    
    def get_config(self):
        config = {
            'dynamic_resolution_h_w': dynamic_resolution_h_w,
            'label_smooth': self.label_smooth, 'eq_loss': self.eq_loss,
            'ema_ratio':    self.ema_ratio,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
        if self.dynamic_scale_manager is not None:
            config['dynamic_scale_state'] = self.dynamic_scale_manager.get_state()
        return config
    
    def state_dict(self):
        m = self.vae_local
        if hasattr(m, '_orig_mod'):
            m = m._orig_mod
        state = {'config': self.get_config(), 'vae_local': m.state_dict()}
        
        if self.zero:   # FSDP 模式
            state['gpt_fsdp'] = None
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                # 分離 Infinity 和 CAR 的權重
                full_state = self.gpt.state_dict()
                infinity_state = {}
                car_state = {}
                
                for key, value in full_state.items():
                    if any(car_prefix in key for car_prefix in ['car_blocks', 'car_control_proj', 'car_fusion_linears', 'car_fusion_gates']):
                        car_state[key] = value
                    else:
                        infinity_state[key] = value
                
                state['gpt_fsdp'] = infinity_state  # 只存 Infinity 部分
                state['car_fsdp'] = car_state       # 單獨存 CAR 部分
                
                if self.use_fsdp_model_ema:
                    ema_full_state = self.gpt_ema.state_dict()
                    ema_infinity_state = {}
                    ema_car_state = {}
                    
                    for key, value in ema_full_state.items():
                        if any(car_prefix in key for car_prefix in ['car_blocks', 'car_control_proj', 'car_fusion_linears', 'car_fusion_gates']):
                            ema_car_state[key] = value
                        else:
                            ema_infinity_state[key] = value
                    
                    state['gpt_ema_fsdp'] = ema_infinity_state
                    state['car_ema_fsdp'] = ema_car_state
                
                state['gpt_fsdp_opt'] = FSDP.optim_state_dict(model=self.gpt, optim=self.gpt_opt.optimizer, optim_state_dict=self.gpt_opt.optimizer.state_dict())
            
            if self.gpt_opt.scaler is not None:
                state['gpt_opt_scaler'] = self.gpt_opt.scaler.state_dict()
        
        else:  # DDP 模式
            if self.using_ema:
                self.ema_load()
                full_ema_state = self.gpt_wo_ddp.state_dict()
                ema_infinity_state = {}
                ema_car_state = {}
                
                for key, value in full_ema_state.items():
                    if any(car_prefix in key for car_prefix in ['car_blocks', 'car_control_proj', 'car_fusion_linears', 'car_fusion_gates']):
                        ema_car_state[key] = value.cpu()
                    else:
                        ema_infinity_state[key] = value.cpu()
                
                state['gpt_ema_for_vis'] = ema_infinity_state
                state['car_ema_for_vis'] = ema_car_state
                self.ema_recover()
            
            # 分離當前狀態
            gpt_wo_ddp = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
            full_state = gpt_wo_ddp.state_dict()
            infinity_state = {}
            car_state = {}
            
            for key, value in full_state.items():
                if any(car_prefix in key for car_prefix in ['car_blocks', 'car_control_proj', 'car_fusion_linears', 'car_fusion_gates']):
                    car_state[key] = value
                else:
                    infinity_state[key] = value
            
            state['gpt_wo_ddp'] = infinity_state  # 只存 Infinity 部分
            state['car_wo_ddp'] = car_state       # 單獨存 CAR 部分
            
            # 優化器狀態保持完整
            gpt_opt = self.gpt_opt._orig_mod if hasattr(self.gpt_opt, '_orig_mod') else self.gpt_opt
            state['gpt_opt'] = gpt_opt.state_dict()
        
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False, car_ckpt_path=None):
        # Ensure CAR modules are initialized before loading
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        if hasattr(gpt_uncompiled, 'has_car_modules'):
            assert gpt_uncompiled.has_car_modules(), "CAR modules must be initialized before loading state"
        
        if self.zero:  # FSDP 模式
            with FSDP.state_dict_type(self.gpt, StateDictType.FULL_STATE_DICT, fullstate_save_policy, fulloptstate_save_policy):
                # 載入完整的 state_dict（包含 Infinity + CAR）
                if 'gpt_fsdp' in state:
                    # 分離 Infinity 和 CAR 參數
                    full_state = state['gpt_fsdp']
                    infinity_state = {k: v for k, v in full_state.items() if not any(car_prefix in k for car_prefix in ['car_', 'control_'])}
                    car_state = {k: v for k, v in full_state.items() if any(car_prefix in k for car_prefix in ['car_', 'control_'])}
                    
                    # 載入 Infinity 部分（強制載入）
                    infinity_missing, infinity_unexpected = self.gpt.load_state_dict(infinity_state, strict=False)
                    print(f'[Infinity] missing: {len(infinity_missing)}, unexpected: {len(infinity_unexpected)}')
                    
                    # 載入 CAR 部分（如果存在）
                    if car_state:
                        car_missing, car_unexpected = self.gpt.load_state_dict(car_state, strict=False)
                        print(f'[CAR] missing: {len(car_missing)}, unexpected: {len(car_unexpected)}')
                    else:
                        print('[CAR] No CAR weights in checkpoint, using random initialization')
                
                # 如果提供了單獨的 CAR checkpoint
                if car_ckpt_path and osp.exists(car_ckpt_path):
                    print(f'[CAR] Loading from separate file: {car_ckpt_path}')
                    car_checkpoint = torch.load(car_ckpt_path, map_location='cpu')
                    if hasattr(gpt_uncompiled, 'load_car_weights'):
                        gpt_uncompiled.load_car_weights(car_checkpoint, strict=False)
                
                # EMA 處理
                if self.use_fsdp_model_ema and 'gpt_ema_fsdp' in state:
                    ema_state = state['gpt_ema_fsdp']
                    infinity_ema_state = {k: v for k, v in ema_state.items() if not any(car_prefix in k for car_prefix in ['car_', 'control_'])}
                    car_ema_state = {k: v for k, v in ema_state.items() if any(car_prefix in k for car_prefix in ['car_', 'control_'])}
                    
                    self.gpt_ema.load_state_dict(infinity_ema_state, strict=False)
                    if car_ema_state:
                        self.gpt_ema.load_state_dict(car_ema_state, strict=False)
                
                # 優化器
                if 'gpt_fsdp_opt' in state:
                    optim_state_dict = FSDP.optim_state_dict_to_load(
                        model=self.gpt, 
                        optim=self.gpt_opt.optimizer, 
                        optim_state_dict=state['gpt_fsdp_opt']
                    )
                    self.gpt_opt.optimizer.load_state_dict(optim_state_dict)
            
            if self.gpt_opt.scaler is not None and 'gpt_opt_scaler' in state:
                try: 
                    self.gpt_opt.scaler.load_state_dict(state['gpt_opt_scaler'])
                except Exception as e: 
                    print(f'[fp16 load_state_dict err] {e}')
            self.gpt._reset_lazy_init()
        else:  # DDP 模式
            # 載入完整的 state_dict
            if 'gpt_wo_ddp' in state:
                full_state = state['gpt_wo_ddp']
                infinity_state = {k: v for k, v in full_state.items() if not any(car_prefix in k for car_prefix in ['car_', 'control_'])}
                car_state = {k: v for k, v in full_state.items() if any(car_prefix in k for car_prefix in ['car_', 'control_'])}
                
                # 載入 Infinity 部分
                infinity_missing, infinity_unexpected = gpt_uncompiled.load_state_dict(infinity_state, strict=False)
                print(f'[Infinity] missing: {len(infinity_missing)}, unexpected: {len(infinity_unexpected)}')
                
                # 載入 CAR 部分
                if car_state:
                    car_missing, car_unexpected = gpt_uncompiled.load_state_dict(car_state, strict=False)
                    print(f'[CAR] missing: {len(car_missing)}, unexpected: {len(car_unexpected)}')
                else:
                    print('[CAR] No CAR weights in checkpoint, using random initialization')
            
            # 如果提供了單獨的 CAR checkpoint
            if car_ckpt_path and osp.exists(car_ckpt_path):
                print(f'[CAR] Loading from separate file: {car_ckpt_path}')
                car_checkpoint = torch.load(car_ckpt_path, map_location='cpu')
                if hasattr(gpt_uncompiled, 'load_car_weights'):
                    gpt_uncompiled.load_car_weights(car_checkpoint, strict=False)
            
            # 載入優化器
            if 'gpt_opt' in state:
                gpt_opt = self.gpt_opt._orig_mod if hasattr(self.gpt_opt, '_orig_mod') else self.gpt_opt
                gpt_opt.load_state_dict(state['gpt_opt'], strict=strict)
            
            # EMA 處理
            if self.using_ema and 'gpt_ema_for_vis' in state:
                for pi, para in self.pi_para_copy_for_parallel_ema:
                    para_name = self.gpt_opt.names[pi]
                    if para_name in state['gpt_ema_for_vis']:
                        para.copy_(state['gpt_ema_for_vis'][para_name])
                print(f'[EMA] load succeed')
        
        # 載入配置
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0) if config else 0
        self.last_prog_si = config.get('last_prog_si', -1) if config else -1
        self.first_prog = config.get('first_prog', True) if config else True
        if config is not None and self.dynamic_scale_manager is not None:
            self.dynamic_scale_manager.load_state(config.get('dynamic_scale_state'))
            restored_limit = min(self.base_training_scale_limit, self.dynamic_scale_manager.get_current_limit())
            self.active_training_scales = max(1, int(restored_limit))
            self.last_training_scales = self.active_training_scales
        
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VGPT.load_state_dict] config mismatch: this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err)
    
    def _debug_parameter_freeze_status(self, iteration: int):
        """调试参数冻结状态并验证optimizer内容"""
        
        # 获取底层模型
        model = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        
        # 获取所有模型参数
        model_params = {name: param for name, param in model.named_parameters()}
        
        # 获取 optimizer 中的参数ID集合
        optimizer_param_ids = set()
        optimizer_param_mapping = {}
        
        for group_idx, param_group in enumerate(self.gpt_opt.optimizer.param_groups):
            for param in param_group['params']:
                param_id = id(param)
                optimizer_param_ids.add(param_id)
                
                # 查找参数名称
                for name, model_param in model_params.items():
                    if id(model_param) == param_id:
                        optimizer_param_mapping[name] = {
                            'group': group_idx,
                            'lr': param_group.get('lr', 'N/A'),
                            'param_id': param_id
                        }
                        break
        
        # 分析参数状态
        infinity_trainable = []
        infinity_frozen = []
        car_trainable = []
        car_frozen = []
        infinity_in_optimizer = []
        car_in_optimizer = []
        
        for name, param in model_params.items():
            param_id = id(param)
            is_car = any(prefix in name for prefix in ['car_', 'control_'])
            is_in_optimizer = param_id in optimizer_param_ids
            
            if is_car:
                if param.requires_grad:
                    car_trainable.append(name)
                    if is_in_optimizer:
                        car_in_optimizer.append(name)
                else:
                    car_frozen.append(name)
            else:  # Infinity参数
                if param.requires_grad:
                    infinity_trainable.append(name)
                    if is_in_optimizer:
                        infinity_in_optimizer.append(name)
                else:
                    infinity_frozen.append(name)
        verbose = False
        # 打印详细统计
        if verbose:
            print(f"📊 PARAMETER ANALYSIS:")
            print(f"   Total model parameters: {len(model_params)}")
            print(f"   Total params in optimizer: {len(optimizer_param_ids)}")
            print()
            
            print(f"🔥 CAR MODULE STATUS:")
            print(f"   CAR trainable: {len(car_trainable)}")
            print(f"   CAR frozen: {len(car_frozen)}")
            print(f"   CAR in optimizer: {len(car_in_optimizer)}")
            if car_frozen:
                print(f"   ❌ CAR frozen params (should be trainable): {car_frozen[:5]}")
            if len(car_trainable) != len(car_in_optimizer):
                print(f"   ❌ CAR trainable/optimizer mismatch!")
            
            print(f"\n❄️  INFINITY MODULE STATUS:")
            print(f"   Infinity trainable: {len(infinity_trainable)}")
            print(f"   Infinity frozen: {len(infinity_frozen)}")
            print(f"   Infinity in optimizer: {len(infinity_in_optimizer)}")
        
        if infinity_trainable:
            print(f"   ❌ PROBLEM: Infinity trainable params (should be frozen):")
            for name in infinity_trainable[:10]:
                print(f"      - {name}")
            raise RuntimeError(f"Infinity parameters are trainable: {infinity_trainable}")
            
        
        if infinity_in_optimizer:
            print(f"   ❌ CRITICAL: Infinity params in optimizer:")
            for name in infinity_in_optimizer[:10]:
                opt_info = optimizer_param_mapping.get(name, {})
                print(f"      - {name} [Group: {opt_info.get('group', 'N/A')}]")
            raise RuntimeError(f"Infinity parameters are in optimizer: {infinity_in_optimizer}")
        

        # =================================================
        # Check params change
        # =================================================
        
        if not hasattr(self, '_param_snapshots'):
            self._param_snapshots = {}
        
        # save current snapshot of selected parameters
        current_snapshot = {}
        for name, param in model_params.items():
            if 'word_embed' in name or 'blocks.0.' in name:  # 采样一些Infinity参数
                current_snapshot[name] = param.data.clone().detach()
        
        # Compare with previous snapshot
        if iteration > 0 and hasattr(self, '_param_snapshots'):
            
            for name, current_tensor in current_snapshot.items():
                if name in self._param_snapshots:
                    prev_tensor = self._param_snapshots[name]
                    diff = torch.abs(current_tensor - prev_tensor).max().item()
                    
                    if diff > 1e-8:  # 检测到变化
                        is_car = any(prefix in name for prefix in ['car_', 'control_'])
                        status = "✅ Expected" if is_car else "❌ UNEXPECTED"
                        # print(f"   {status}: {name} changed by {diff:.2e}")
                        if not is_car:
                            raise RuntimeError(f"Infinity parameter '{name}' changed during training!   {status}: {name} changed by {diff:.2e}")
        
        # update snapshot
        self._param_snapshots = current_snapshot

        # =================================================
        # Check gradient status
        # =================================================
        has_grad_infinity = 0
        has_grad_car = 0
        no_grad_infinity = 0
        no_grad_car = 0
        
        for name, param in model_params.items():
            is_car = any(prefix in name for prefix in ['car_', 'control_'])
            has_grad = param.grad is not None and param.grad.abs().sum() > 0
            
            if is_car:
                if has_grad:
                    has_grad_car += 1
                else:
                    no_grad_car += 1
            else:
                if has_grad:
                    has_grad_infinity += 1
                else:
                    no_grad_infinity += 1
        
        
        if has_grad_infinity > 0:
            raise RuntimeError(f"Infinity parameters have gradients! Count: {has_grad_infinity}")

    def generate_training_visualization(
        self,
        ep: int,
        it: int,
        g_it: int,
        inp_B3HW: torch.Tensor,
        raw_features_BdHW: torch.Tensor,
        condition_inputs: Optional[Dict[str, torch.Tensor]],
        gt_ms_idx_Bl: List[torch.Tensor],
        pred_ms_idx_Bl: List[torch.Tensor],
        scale_schedule: List[Tuple[int, int, int]],
        training_scales: int,
        training_seq_len: int,
        full_gt_ms_idx_Bl: List[torch.Tensor],
        full_scale_schedule: List[Tuple[int, int, int]],
        full_vae_scale_schedule: List[Tuple[int, int, int]],
    ):
        """Generate training visualization using current step results"""
        # Only run on master process to avoid duplicate visualizations
        if not dist.is_master():
            return None
        if not getattr(self.gpt_wo_ddp, 'use_bit_label', False):
            return None
            
        try:
            return _generate_training_visualization(
                self, ep, it, g_it, 
                inp_B3HW, raw_features_BdHW, condition_inputs, 
                gt_ms_idx_Bl,
                pred_ms_idx_Bl,
                scale_schedule,
                training_scales,
                training_seq_len,
                full_gt_ms_idx_Bl,
                full_scale_schedule,
                full_vae_scale_schedule,
            )
        except Exception as e:
            print(f"⚠️ Visualization generation error at iteration {g_it}: {e}")
            import traceback
            traceback.print_exc()
        return None
