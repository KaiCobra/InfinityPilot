import random
import time
import gc
import os.path as osp
from functools import partial
from pprint import pformat
from typing import List, Optional, Tuple, Union
from collections import defaultdict

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
    from parameter_visualizer import ParameterChangeVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    print("Parameter visualizer not available. Install matplotlib and seaborn for visualization.")

import infinity.utils.dist as dist
from infinity.models.infinity_pilot import InfinityPilot
from infinity.models.ema import update_ema
from infinity.models.bitwise_self_correction import BitwiseSelfCorrection
from infinity.utils import arg_util, misc, wandb_utils
from infinity.utils.amp_opt import AmpOptimizer
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
fulloptstate_save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)

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
        print(f'self.reweight_loss_by_scale: {self.reweight_loss_by_scale}')
        
        # Ensure CAR modules are initialized
        gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
        if hasattr(gpt_uncompiled, 'init_car_modules_if_needed'):
            gpt_uncompiled.init_car_modules_if_needed()
            print("CAR modules initialized in trainer")
        
        # Print parameter counts
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
        if VISUALIZER_AVAILABLE and dist.is_master():
            try:
                # Get the underlying model without DDP/FSDP wrapper
                underlying_model = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
                self.param_visualizer = ParameterChangeVisualizer(
                    underlying_model, 
                    save_dir=f"./debug/param_visualizations_{time.strftime('%Y%m%d_%H%M%S')}"
                )
                print("✅ Parameter visualizer initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize parameter visualizer: {e}")
                self.param_visualizer = None
    
        if hasattr(self.gpt_wo_ddp, 'init_car_modules_if_needed'):
            self.gpt_wo_ddp.init_car_modules_if_needed()

    @torch.no_grad()
    def _generate_eval_visualization(self, ep: int, eval_images: List[torch.Tensor], 
                                   eval_conditions: List[torch.Tensor], eval_prompts: List[str], args):
        """Generate visualization images during evaluation and log to wandb/tensorboard"""
        try:
            print(f"Generating evaluation visualization for epoch {ep}...")
            
            # Import generation utilities
            from tools.run_infinity_pilot import gen_one_img_with_condition
            
            # Get underlying model without wrapper
            model = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
            
            visualization_images = []
            
            for i, (orig_img, condition, prompt) in enumerate(zip(eval_images, eval_conditions, eval_prompts)):
                if i >= 4:  # Limit to 4 samples to avoid memory issues
                    break
                    
                try:
                    # Prepare inputs
                    orig_img = orig_img.to(args.device)
                    if condition is not None:
                        condition = condition.to(args.device)
                    
                    # Determine scale schedule
                    h_div_w = orig_img.shape[-2] / orig_img.shape[-1]
                    h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
                    h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
                    scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                    
                    # Generate with control
                    if condition is not None:
                        control_tensors = model.prepare_control_for_scales(condition, scale_schedule)
                        generated_img = model.autoregressive_infer_cfg(
                            control_tensors=control_tensors,
                            scale_schedule=scale_schedule,
                            cfg=2.0,
                            tau_list=[0.9],
                            vae=self.vae_local,
                            g_seed=42 + i
                        )
                    else:
                        # Generate without control (standard generation)
                        generated_img = model.autoregressive_infer_cfg(
                            scale_schedule=scale_schedule,
                            cfg=2.0,
                            tau_list=[0.9],
                            vae=self.vae_local,
                            g_seed=42 + i
                        )
                    
                    # Create comparison grid: [Original, Condition (if exists), Generated]
                    comparison_images = []
                    
                    # Original image (normalize to [0,1])
                    orig_display = (orig_img.squeeze(0) + 1) / 2
                    comparison_images.append(orig_display)
                    
                    # Condition image (if exists)
                    if condition is not None:
                        condition_display = (condition.squeeze(0) + 1) / 2
                        comparison_images.append(condition_display)
                    
                    # Generated image (normalize to [0,1])
                    if isinstance(generated_img, torch.Tensor):
                        if generated_img.max() > 1.0:  # If in [0,255] range
                            generated_display = generated_img.float() / 255.0
                        else:  # If in [-1,1] or [0,1] range
                            generated_display = (generated_img + 1) / 2 if generated_img.min() < 0 else generated_img
                    else:
                        # Convert numpy to tensor if needed
                        generated_display = torch.from_numpy(generated_img).permute(2, 0, 1) / 255.0
                    
                    if generated_display.dim() == 4:
                        generated_display = generated_display.squeeze(0)
                    comparison_images.append(generated_display)
                    
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
        max_vis_samples = 8  # Limit number of visualization samples
        
        for batch_idx, data in enumerate(ld_val):
            # Handle data format for pilot (with condition)
            if len(data) == 3:
                inp, condition, label_B = data
            else:
                inp, label_B = data
                condition = None
            
            B = label_B.shape[0] if isinstance(label_B, torch.Tensor) else inp.shape[0]
            if isinstance(label_B, torch.Tensor):
                label_B = label_B.to(args.device, non_blocking=True)
            V = self.vae_local.vocab_size
            inp = inp.to(args.device, non_blocking=True)
            if condition is not None:
                condition = condition.to(args.device, non_blocking=True)
            
            gt_ms_idx_Bl: List[Ten] = self.vae_local.get_GPT_ground_truth(inp)
            
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)
            
            # Prepare control tensors for InfinityPilot
            if condition is not None:
                h_div_w = inp.shape[-2] / inp.shape[-1]
                h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
                h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
                scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
                control_tensors = self.gpt_wo_ddp.prepare_control_for_scales(condition, scale_schedule)
            else:
                control_tensors = None
                # Use default scale schedule if no condition
                h_div_w = inp.shape[-2] / inp.shape[-1]
                h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
                h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
                scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
            
            self.gpt_wo_ddp.forward
            logits_BLV = self.gpt_wo_ddp(label_B, self.quantize_local.fuse_multiscale_idx_as_gpt_inp_BL(gt_ms_idx_Bl), 
                                        scale_schedule=scale_schedule, control_tensors=control_tensors)
            
            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.raw_last_l:].reshape(-1, V), gt_BL[:, -self.raw_last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.raw_last_l:].argmax(dim=-1) == gt_BL[:, -self.raw_last_l:]).sum() * (100/self.raw_last_l)
            tot += B
            
            # Collect samples for visualization (only from master process and first few batches)
            if dist.is_master() and len(eval_images) < max_vis_samples and batch_idx < 4:
                # Store original images, conditions, and generate samples for visualization
                for i in range(min(B, max_vis_samples - len(eval_images))):
                    eval_images.append(inp[i:i+1].cpu())  # Keep original image
                    if condition is not None:
                        eval_conditions.append(condition[i:i+1].cpu())
                    else:
                        eval_conditions.append(None)
                    eval_prompts.append(f"Sample_{len(eval_images)}")
                    
        self.gpt_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        
        # Generate visualization images during evaluation
        if dist.is_master() and len(eval_images) > 0:
            self._generate_eval_visualization(ep, eval_images, eval_conditions, eval_prompts, args)
        
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, ep: int, it: int, g_it: int, stepping: bool, clip_decay_ratio: float,
        metric_lg: misc.MetricLogger, logging_params: bool,
        inp_B3HW: FTen, condition_B3HW: Optional[FTen], text_cond_tuple: Union[ITen, FTen], args: arg_util.Args,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        
        # Debug parameter freeze status at the beginning of training steps
        if it <= 2 and ep == 0:  # Only check first few iterations of first epoch
            self._debug_parameter_freeze_status(it)
        
        B = inp_B3HW.shape[0]  # if isinstance(inp_B3HW, torch.Tensor) else inp_B3HW[0].shape[0]
        T = 1 if inp_B3HW.dim() == 4 else inp_B3HW.shape[2]
        V = self.vae_local.vocab_size
        device = inp_B3HW.device

        h_div_w = inp_B3HW.shape[-2] / inp_B3HW.shape[-1]
        h_div_w_templates = np.array(list(dynamic_resolution_h_w.keys()))
        h_div_w_template = h_div_w_templates[np.argmin(np.abs(h_div_w-h_div_w_templates))]
        scale_schedule = dynamic_resolution_h_w[h_div_w_template][args.pn]['scales']
        scale_schedule = [ (min(t, T//4+1), h, w) for (t,h, w) in scale_schedule]
        
        # [forward]
        with self.gpt_opt.amp_ctx:
            with torch.amp.autocast('cuda', enabled=False):
                with torch.no_grad():
                    if args.apply_spatial_patchify:
                        vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
                    else:
                        vae_scale_schedule = scale_schedule
                    raw_features, _, _ = self.vae_local.encode_for_raw_features(inp_B3HW, scale_schedule=vae_scale_schedule)
            
            x_BLC_wo_prefix, gt_ms_idx_Bl = self.bitwise_self_correction.flip_requant(vae_scale_schedule, inp_B3HW, raw_features, device)
            # x_BLC_wo_prefix: torch.Size([bs, 2*2+3*3+...+64*64, d or 4d])

            # truncate scales
            training_scales = args.always_training_scales
            training_seq_len = np.array(scale_schedule)[:training_scales].prod(axis=1).sum()
            x_BLC_wo_prefix = x_BLC_wo_prefix[:, :(training_seq_len-np.array(scale_schedule[0]).prod()), :]

            # Prepare control tensors for InfinityPilot
            control_tensors = None
            if condition_B3HW is not None:
                control_tensors = self.gpt_wo_ddp.prepare_control_for_scales(condition_B3HW, scale_schedule[:training_scales])

            self.gpt_wo_ddp.forward  
            
            # Auto-fix any NaN/Inf parameters before forward pass
            nan_params = []
            inf_params = []
            for name, param in self.gpt.named_parameters():
                if torch.isnan(param).any():
                    nan_params.append(name)
                if torch.isinf(param).any():
                    inf_params.append(name)
            
            if nan_params or inf_params:
                print(f"[WARNING] Found problematic parameters - attempting to fix...")
                if nan_params:
                    print(f"[WARNING] NaN parameters: {nan_params}")
                if inf_params:
                    print(f"[WARNING] Inf parameters: {inf_params}")
                
                # Auto-fix NaN/Inf parameters by reinitializing them
                for name, param in self.gpt.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"[FIX] Reinitializing parameter: {name}")
                        if 'bias' in name:
                            nn.init.zeros_(param)
                        elif 'weight' in name:
                            if param.dim() >= 2:
                                nn.init.xavier_uniform_(param)
                            else:
                                nn.init.normal_(param, 0, 0.02)
                        else:
                            nn.init.normal_(param, 0, 0.02)
                
                print(f"[FIX] Parameter reinitialization completed")

            # Check CAR module initialization and ensure proper gradient flow
            gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
            has_car = hasattr(gpt_uncompiled, 'has_car_modules') and gpt_uncompiled.has_car_modules()
            if not has_car:
                gpt_uncompiled.init_car_modules_if_needed()
                print("[WARNING] CAR modules were not initialized, initializing now")
            
            # Ensure only CAR parameters require grad for DDP
            if not hasattr(gpt_uncompiled, '_gradient_setup_done'):
                car_param_count = 0
                infinity_param_count = 0
                for name, param in gpt_uncompiled.named_parameters():
                    if any(car_prefix in name for car_prefix in ['car_control_convs', 'car_var_conv', 'car_blocks', 'car_skip_norm', 'car_skip_linear']):
                        param.requires_grad = True
                        car_param_count += 1
                    else:
                        param.requires_grad = False  # Freeze Infinity parameters
                        infinity_param_count += 1
                print(f"[Gradient Setup] CAR params (trainable): {car_param_count}, Infinity params (frozen): {infinity_param_count}")
                gpt_uncompiled._gradient_setup_done = True
            
            logits_BLV = self.gpt(text_cond_tuple, x_BLC_wo_prefix, scale_schedule=scale_schedule[:training_scales], control_tensors=control_tensors) # [bs, 1*1+...+64*64, vocab_size or log2(vocab_size)*2]
            self.batch_size, self.seq_len = logits_BLV.shape[:2]

            self.seq_len_each = [idx_Bl.shape[1] for idx_Bl in gt_ms_idx_Bl]
            
            gt_BL = torch.cat(gt_ms_idx_Bl, dim=1)[:,:training_seq_len].contiguous().type(torch.long) # [bs, 1*1+...+64*64, 16] or [bs, 1*1+...+64*64]
            
            # 檢查 logits 是否包含 NaN 或 inf
            if not torch.isfinite(logits_BLV).all():
                print(f"[ERROR] Non-finite values in logits_BLV")
                print(f"[DEBUG] logits_BLV contains nan: {torch.isnan(logits_BLV).any()}")
                print(f"[DEBUG] logits_BLV contains inf: {torch.isinf(logits_BLV).any()}")
                print(f"[DEBUG] logits_BLV min: {logits_BLV.min()}")
                print(f"[DEBUG] logits_BLV max: {logits_BLV.max()}")
                raise RuntimeError("Non-finite values in logits before loss calculation")
            
            # 檢查 logits 的數值範圍，如果過大可能導致數值不穩定
            logits_max = logits_BLV.max().item()
            logits_min = logits_BLV.min().item()
            if abs(logits_max) > 100 or abs(logits_min) > 100:
                print(f"[WARNING] Large logits values detected: min={logits_min:.2f}, max={logits_max:.2f}")
            
            if args.use_bit_label:
                tmp_bs, tmp_seq_len, tmp_channel = logits_BLV.shape
                loss = self.train_loss(logits_BLV.reshape(tmp_bs, tmp_seq_len, -1, 2).permute(0,3,1,2), gt_BL)
                if args.bitloss_type == 'mean':
                    loss = loss.mean(dim=-1)
                elif args.bitloss_type == 'sum':
                    loss = loss.sum(dim=-1)
                else:
                    raise NotImplementedError(f'{args.bitloss_type=}')
            else:
                loss = self.train_loss(logits_BLV.reshape(-1, V), gt_BL.reshape(-1)).reshape(B, -1)

            if self.reweight_loss_by_scale:
                lw = []
                last_scale_area = np.sqrt(np.prod(scale_schedule[-1]))
                for (pt, ph, pw) in scale_schedule[:training_scales]:
                    this_scale_area = np.sqrt(pt * ph * pw)
                    lw.extend([last_scale_area / this_scale_area for _ in range(pt * ph * pw)])
                lw = torch.tensor(lw, device=loss.device)[None, ...]
                lw = lw / lw.sum()
            else:
                lw = 1. / self.seq_len
            loss = loss.mul(lw).sum(dim=-1).mean()
        
        # [backward]
        grad_norm_t, scale_log2_t = self.gpt_opt.backward_clip_step(ep=ep, it=it, g_it=g_it, stepping=stepping, logging_params=logging_params, loss=loss, clip_decay_ratio=clip_decay_ratio, stable=args.stable)
        
        # Update parameter visualizer
        if self.param_visualizer is not None and stepping:
            self.param_visualizer.update(g_it)
            
            # Generate visualization every 10 steps or at specific milestones
            if g_it % 10 == 0 or it < 5:
                try:
                    self.param_visualizer.plot_module_heatmap(g_it)
                    if it < 3:  # Generate architecture diagram for first few iterations
                        self.param_visualizer.plot_architecture_diagram(g_it)
                except Exception as e:
                    print(f"⚠️ Visualization error at iteration {g_it}: {e}")
        
        # Debug parameter freeze status for first few iterations
        if it <= 2 and ep == 0 and dist.is_master():
            self._debug_parameter_freeze_status(it)
        
        # Debug: Check parameter gradient status for DDP
        if stepping and args.dbg and it < 3:  # Only for first few iterations
            grad_params = []
            no_grad_params = []
            gpt_uncompiled = self.gpt_wo_ddp._orig_mod if hasattr(self.gpt_wo_ddp, '_orig_mod') else self.gpt_wo_ddp
            for name, param in gpt_uncompiled.named_parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        grad_params.append(name)
                    else:
                        no_grad_params.append(name)
            print(f"[DEBUG] Iteration {it}: {len(grad_params)} params with grad, {len(no_grad_params)} params without grad")
            if no_grad_params:
                print(f"[DEBUG] No grad params: {no_grad_params[:5]}...")  # Show first 5
        
        # update ema
        if args.use_fsdp_model_ema:
            update_ema(self.gpt_ema, self.gpt)

        # [zero_grad]
        if stepping:
            if self.using_ema: self.ema_update(g_it)
            if self.dbg_unused:
                ls = []
                for n, p in self.gpt_wo_ddp.named_parameters():
                    if p.grad is None:
                        ls.append(n)
                if len(ls):
                    raise AttributeError(f'unused param: {ls}')
        
            self.gpt_opt.optimizer.zero_grad(set_to_none=True)
        
        # [metric logging]
        if metric_lg.log_every_iter or it == 0 or it in metric_lg.log_iters:
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
            if stepping and self.param_visualizer is not None:
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
            
            wandb_utils.log(wandb_log_dict, step=g_it)
        
        return grad_norm_t, scale_log2_t
    
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
        return {
            'dynamic_resolution_h_w': dynamic_resolution_h_w,
            'label_smooth': self.label_smooth, 'eq_loss': self.eq_loss,
            'ema_ratio':    self.ema_ratio,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
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
                    if any(car_prefix in key for car_prefix in ['car_control_convs', 'car_var_conv', 'car_blocks', 'car_skip_norm', 'car_skip_linear']):
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
                        if any(car_prefix in key for car_prefix in ['car_control_convs', 'car_var_conv', 'car_blocks', 'car_skip_norm', 'car_skip_linear']):
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
                    if any(car_prefix in key for car_prefix in ['car_control_convs', 'car_var_conv', 'car_blocks', 'car_skip_norm', 'car_skip_linear']):
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
                if any(car_prefix in key for car_prefix in ['car_control_convs', 'car_var_conv', 'car_blocks', 'car_skip_norm', 'car_skip_linear']):
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
        if hasattr(gpt_uncompiled, 'init_car_modules_if_needed'):
            gpt_uncompiled.init_car_modules_if_needed()
        
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
        print(f"\n{'='*80}")
        print(f"PARAMETER FREEZE & OPTIMIZER DEBUG - Iteration {iteration}")
        print(f"{'='*80}")
        
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
        
        # 打印详细统计
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
        
        if infinity_in_optimizer:
            print(f"   ❌ CRITICAL: Infinity params in optimizer:")
            for name in infinity_in_optimizer[:10]:
                opt_info = optimizer_param_mapping.get(name, {})
                print(f"      - {name} [Group: {opt_info.get('group', 'N/A')}]")
        
        # 特别检查：追踪参数变化
        if not hasattr(self, '_param_snapshots'):
            self._param_snapshots = {}
        
        # 保存当前参数快照
        current_snapshot = {}
        for name, param in model_params.items():
            if 'word_embed' in name or 'blocks.0.' in name:  # 采样一些Infinity参数
                current_snapshot[name] = param.data.clone().detach()
        
        # 比较参数变化
        if iteration > 0 and hasattr(self, '_param_snapshots'):
            print(f"\n📈 PARAMETER CHANGE DETECTION (since last check):")
            changes_detected = False
            
            for name, current_tensor in current_snapshot.items():
                if name in self._param_snapshots:
                    prev_tensor = self._param_snapshots[name]
                    diff = torch.abs(current_tensor - prev_tensor).max().item()
                    
                    if diff > 1e-8:  # 检测到变化
                        is_car = any(prefix in name for prefix in ['car_', 'control_'])
                        status = "✅ Expected" if is_car else "❌ UNEXPECTED"
                        print(f"   {status}: {name} changed by {diff:.2e}")
                        changes_detected = True
            
            if not changes_detected:
                print(f"   ✅ No parameter changes detected in sampled Infinity params")
        
        # 更新快照
        self._param_snapshots = current_snapshot
        
        # 检查EMA状态
        if hasattr(self, 'ema') and self.ema is not None:
            print(f"\n🔄 EMA STATUS:")
            print(f"   EMA enabled: Yes")
            print(f"   ⚠️  EMA updates ALL parameters (including frozen ones)")
            print(f"   This is normal and doesn't affect training gradients")
        
        # 检查gradient状态
        print(f"\n🎯 GRADIENT VERIFICATION:")
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
        
        print(f"   CAR params with gradients: {has_grad_car}")
        print(f"   CAR params without gradients: {no_grad_car}")
        print(f"   Infinity params with gradients: {has_grad_infinity}")
        print(f"   Infinity params without gradients: {no_grad_infinity}")
        
        if has_grad_infinity > 0:
            print(f"   ❌ CRITICAL: Infinity parameters have gradients!")
        else:
            print(f"   ✅ CORRECT: No gradients on Infinity parameters")
        
        print("="*80)