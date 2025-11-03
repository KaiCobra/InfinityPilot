import wandb
import torch
from torchvision.utils import make_grid
from typing import Dict, Any, Optional
import torch.distributed as dist
from PIL import Image
import os
import argparse
import hashlib
import math


def is_main_process():
    return dist.get_rank() == 0

def namespace_to_dict(namespace):
    return {
        k: namespace_to_dict(v) if isinstance(v, argparse.Namespace) else v
        for k, v in vars(namespace).items()
    }


def generate_run_id(exp_name):
    # https://stackoverflow.com/questions/16008670/how-to-hash-a-string-into-8-digits
    return str(int(hashlib.sha256(exp_name.encode('utf-8')).hexdigest(), 16) % 10 ** 8)


# def initialize(args, entity, exp_name, project_name):
#     config_dict = namespace_to_dict(args)
#     wandb.login(key='318c6ab88c39d7e5761e620e142c77ed7b75541e')
#     wandb.init(
#         entity=entity,
#         project=project_name,
#         name=exp_name,
#         config=config_dict,
#         id=generate_run_id(exp_name),
#         resume="allow",
#     )
def initialize(args, exp_name, project_name,):
    """初始化 W&B 運行"""
    print("[DEBUG] initialize wandb, rank=", dist.get_rank() if dist.is_initialized() else 'N/A')
    
    # 檢查是否為離線模式
    wandb_mode = os.environ.get('WANDB_MODE', 'online')
    print("[DEBUG] WANDB_MODE:", wandb_mode)
    
    config_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
    
    if wandb_mode == 'offline':
        print("[DEBUG] Running in offline mode")
        wandb.init(
            settings=wandb.Settings(mode=wandb_mode),
            project=project_name,
            name=exp_name,
            config=config_dict
        )
        return
    
        # 在線模式：嘗試登錄並處理錯誤
    try:
        # 檢查環境變量
        api_key = os.environ.get('WANDB_API_KEY')
        entity = os.environ.get('WANDB_ENTITY')
        # print(f"[DEBUG] API Key present: {bool(api_key)}")
        # print(f"[DEBUG] Entity: {entity}")
        
        # print("[DEBUG] Attempting online login...")
        wandb.login(key=api_key, relogin=True)
        # print("[DEBUG] Successfully logged in to wandb")
        
        # print("[DEBUG] Attempting wandb.init...")
        wandb.init(
            project=project_name,
            name=exp_name,
            config=config_dict,
            entity=entity,
            sync_tensorboard=args.sync_tensorboard
        )
        # print("[DEBUG] Successfully initialized wandb in online mode")
        # print(f"[DEBUG] Run URL: {wandb.run.url if wandb.run else 'N/A'}")
        
    except Exception as e:
        print(f"[WARNING] Failed to initialize wandb in online mode: {e}")
        print(f"[DEBUG] Exception type: {type(e).__name__}")
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")
        print("[INFO] Falling back to offline mode")
        
        # 確保清理之前的狀態
        if wandb.run is not None:
            wandb.finish()
            
        wandb.init(
            mode='offline',
            project=project_name,
            name=exp_name,
            config=config_dict
        )
        print("[DEBUG] Successfully initialized wandb in offline mode")

def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    # print("[DEBUG] log_metrics called, metrics:", metrics)
    if wandb.run is None:
        print("W&B not initialized, skipping logging")
        return
    if step is not None:
        metrics['global_step'] = step
    wandb.log(metrics)

def finish():
    """結束 W&B 運行"""
    if wandb.run is not None:
        wandb.finish()

def log(stats, step=None):
    if is_main_process():
        wandb.log({k: v for k, v in stats.items()}, step=step)


def log_image(name, sample, step=None):
    if is_main_process():
        sample = array2grid(sample)
        wandb.log({f"{name}": wandb.Image(sample), "train_step": step})


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x, nrow=nrow, normalize=True, value_range=(-1,1))
    x = x.mul(255).add_(0.5).clamp_(0,255).permute(1,2,0).to('cpu', torch.uint8).numpy()
    return x