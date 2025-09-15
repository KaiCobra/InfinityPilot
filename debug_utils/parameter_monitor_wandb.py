"""
Wandb 參數監控整合
將參數變化監控整合到 Weights & Biases 中
"""

import torch
import numpy as np
import wandb
from collections import defaultdict

def log_parameter_stats_to_wandb(model, step, prefix="param_monitor"):
    """
    將參數統計信息記錄到 Wandb
    
    Args:
        model: 模型
        step: 當前步數
        prefix: 日誌前綴
    """
    
    # 分類參數
    car_params = []
    infinity_params = []
    other_params = []
    
    car_grad_norms = []
    infinity_grad_norms = []
    car_param_norms = []
    infinity_param_norms = []
    
    frozen_with_grad_count = 0
    trainable_without_grad_count = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            param_norm = torch.norm(param).item()
            
            # 分類參數
            if any(car_prefix in name for car_prefix in ['car_control_convs', 'car_var_conv', 'car_blocks', 'car_skip_norm', 'car_skip_linear']):
                car_params.append(name)
                car_param_norms.append(param_norm)
                
                if param.grad is not None:
                    car_grad_norms.append(torch.norm(param.grad).item())
                elif param.requires_grad:
                    trainable_without_grad_count += 1
                    
            elif any(inf_prefix in name for inf_prefix in ['blocks', 'word_embed', 'pos_start', 'lvl_embed']):
                infinity_params.append(name)
                infinity_param_norms.append(param_norm)
                
                if param.grad is not None:
                    infinity_grad_norms.append(torch.norm(param.grad).item())
                    if not param.requires_grad:
                        frozen_with_grad_count += 1
            else:
                other_params.append(name)
    
    # 計算統計信息
    stats = {
        f"{prefix}/car_param_count": len(car_params),
        f"{prefix}/infinity_param_count": len(infinity_params),
        f"{prefix}/other_param_count": len(other_params),
        
        f"{prefix}/car_avg_param_norm": np.mean(car_param_norms) if car_param_norms else 0,
        f"{prefix}/infinity_avg_param_norm": np.mean(infinity_param_norms) if infinity_param_norms else 0,
        
        f"{prefix}/car_avg_grad_norm": np.mean(car_grad_norms) if car_grad_norms else 0,
        f"{prefix}/infinity_avg_grad_norm": np.mean(infinity_grad_norms) if infinity_grad_norms else 0,
        
        f"{prefix}/frozen_with_grad_count": frozen_with_grad_count,
        f"{prefix}/trainable_without_grad_count": trainable_without_grad_count,
        
        f"{prefix}/car_grad_count": len(car_grad_norms),
        f"{prefix}/infinity_grad_count": len(infinity_grad_norms),
    }
    
    # 記錄到 Wandb
    wandb.log(stats, step=step)
    
    # 返回問題計數以供打印
    return frozen_with_grad_count, trainable_without_grad_count

def create_parameter_change_table(model, baseline_params, step):
    """
    創建參數變化表格並記錄到 Wandb
    
    Args:
        model: 當前模型
        baseline_params: 基線參數
        step: 當前步數
    """
    
    changes_data = []
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in baseline_params:
                # 計算變化
                diff = torch.norm(param - baseline_params[name]).item()
                param_norm = torch.norm(param).item()
                relative_change = diff / (param_norm + 1e-8)
                
                # 確定參數類型
                if any(car_prefix in name for car_prefix in ['car_control_convs', 'car_var_conv', 'car_blocks']):
                    param_type = "CAR"
                elif any(inf_prefix in name for inf_prefix in ['blocks', 'word_embed', 'pos_start']):
                    param_type = "Infinity"
                else:
                    param_type = "Other"
                
                changes_data.append([
                    name.split('.')[-1][:30],  # 簡化的參數名
                    param_type,
                    param.requires_grad,
                    f"{diff:.2e}",
                    f"{relative_change:.2e}",
                    f"{param_norm:.2e}"
                ])
    
    # 創建表格
    table = wandb.Table(
        columns=["Parameter", "Type", "Trainable", "Abs Change", "Rel Change", "Norm"],
        data=changes_data
    )
    
    wandb.log({"parameter_changes": table}, step=step)

def log_freeze_violations_to_wandb(model, step, threshold=1e-8):
    """
    記錄凍結違規情況到 Wandb
    
    Args:
        model: 模型
        step: 當前步數
        threshold: 變化閾值
    """
    
    violations = []
    
    for name, param in model.named_parameters():
        # 檢查凍結參數是否有梯度
        if not param.requires_grad and param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            if grad_norm > threshold:
                param_type = "CAR" if "car_" in name else ("Infinity" if any(x in name for x in ['blocks', 'word_embed']) else "Other")
                violations.append([
                    name.split('.')[-1][:30],
                    param_type,
                    f"{grad_norm:.2e}",
                    "Frozen param has gradient"
                ])
        
        # 檢查可訓練參數是否沒有梯度
        elif param.requires_grad and param.grad is None:
            param_type = "CAR" if "car_" in name else ("Infinity" if any(x in name for x in ['blocks', 'word_embed']) else "Other")
            violations.append([
                name.split('.')[-1][:30],
                param_type,
                "0.0e+00",
                "Trainable param has no gradient"
            ])
    
    if violations:
        violation_table = wandb.Table(
            columns=["Parameter", "Type", "Gradient Norm", "Issue"],
            data=violations
        )
        wandb.log({"freeze_violations": violation_table}, step=step)
        
        # 也記錄違規計數
        wandb.log({
            "param_monitor/total_violations": len(violations),
            "param_monitor/has_violations": 1
        }, step=step)
    else:
        wandb.log({
            "param_monitor/total_violations": 0,
            "param_monitor/has_violations": 0
        }, step=step)
    
    return len(violations)

# 使用範例
def integrate_wandb_monitoring():
    """展示如何整合到現有的 Wandb 日誌中"""
    example_code = '''
# 在訓練循環中添加（已有 wandb.log 的地方）

# 保存基線參數（訓練開始時）
baseline_params = {}
with torch.no_grad():
    for name, param in model.named_parameters():
        baseline_params[name] = param.clone().cpu()

# 在訓練循環中（例如每50個iteration）
if it % 50 == 0:
    from parameter_monitor_wandb import log_parameter_stats_to_wandb, log_freeze_violations_to_wandb
    
    # 記錄參數統計
    frozen_grad_count, trainable_no_grad_count = log_parameter_stats_to_wandb(model, step=g_it)
    
    # 記錄凍結違規
    violation_count = log_freeze_violations_to_wandb(model, step=g_it)
    
    # 每200個iteration創建變化表格
    if it % 200 == 0:
        create_parameter_change_table(model, baseline_params, step=g_it)
    
    # 簡單的控制台輸出
    if violation_count > 0:
        print(f"⚠️ Found {violation_count} parameter violations at iteration {it}")
    '''
    
    print("Wandb Integration Example:")
    print(example_code)

if __name__ == "__main__":
    print("Wandb Parameter Monitor Integration")
    integrate_wandb_monitoring()
