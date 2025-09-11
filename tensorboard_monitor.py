import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class TensorBoardParameterMonitor:
    """
    使用 TensorBoard 監控參數變化的工具
    """
    def __init__(self, model, log_dir="./tensorboard_logs"):
        self.model = model
        self.writer = SummaryWriter(log_dir)
        self.initial_params = {}
        self.step_count = 0
        
        # 保存初始參數
        self._save_initial_params()
        
        # 定義模組分組
        self.module_groups = {
            'Infinity_Embedding': ['word_embed', 'pos_start', 'pos_1LC', 'lvl_embed'],
            'Infinity_Blocks': ['blocks'],
            'Infinity_Output': ['head_nm', 'head'],
            'CAR_Control': ['car_control_convs'],
            'CAR_VAR': ['car_var_conv'],
            'CAR_Blocks': ['car_blocks'],
            'CAR_Skip': ['car_skip_norm', 'car_skip_linear'],
        }
    
    def _save_initial_params(self):
        """保存初始參數狀態"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.initial_params[name] = param.data.clone().detach()
    
    def _categorize_parameter(self, param_name: str) -> str:
        """將參數分類"""
        for group_name, keywords in self.module_groups.items():
            for keyword in keywords:
                if keyword in param_name:
                    return group_name
        return 'Other'
    
    def log_parameter_changes(self, step: int):
        """記錄參數變化到 TensorBoard"""
        module_changes = defaultdict(list)
        module_grads = defaultdict(list)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.initial_params:
                # 計算參數變化
                initial = self.initial_params[name]
                change = torch.norm(param.data - initial).item()
                param_norm = torch.norm(initial).item()
                relative_change = change / param_norm if param_norm > 1e-8 else change
                
                # 記錄梯度
                grad_norm = torch.norm(param.grad).item() if param.grad is not None else 0.0
                
                # 分類到模組
                module = self._categorize_parameter(name)
                module_changes[module].append(relative_change)
                module_grads[module].append(grad_norm)
                
                # 記錄個別參數
                self.writer.add_scalar(f'Parameters/Change/{name}', relative_change, step)
                self.writer.add_scalar(f'Gradients/{name}', grad_norm, step)
        
        # 記錄模組級別統計
        for module in module_changes:
            if module_changes[module]:
                mean_change = np.mean(module_changes[module])
                max_change = np.max(module_changes[module])
                mean_grad = np.mean(module_grads[module])
                max_grad = np.max(module_grads[module])
                
                self.writer.add_scalar(f'Modules/ParameterChange/Mean/{module}', mean_change, step)
                self.writer.add_scalar(f'Modules/ParameterChange/Max/{module}', max_change, step)
                self.writer.add_scalar(f'Modules/Gradients/Mean/{module}', mean_grad, step)
                self.writer.add_scalar(f'Modules/Gradients/Max/{module}', max_grad, step)
        
        # 創建凍結狀態檢查
        freeze_status = {}
        for module in module_changes:
            changes = module_changes[module]
            is_frozen = all(change < 1e-7 for change in changes) if changes else True
            freeze_status[module] = 1.0 if is_frozen else 0.0
            self.writer.add_scalar(f'FreezeStatus/{module}', freeze_status[module], step)
        
        self.step_count += 1
    
    def create_parameter_heatmap(self, step: int):
        """創建參數變化熱度圖"""
        module_data = defaultdict(list)
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.initial_params:
                initial = self.initial_params[name]
                change = torch.norm(param.data - initial).item()
                param_norm = torch.norm(initial).item()
                relative_change = change / param_norm if param_norm > 1e-8 else change
                
                module = self._categorize_parameter(name)
                module_data[module].append(relative_change)
        
        # 創建熱度圖
        modules = list(module_data.keys())
        if modules:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 準備數據
            max_params_per_module = max(len(module_data[m]) for m in modules)
            heatmap_data = []
            
            for module in modules:
                row = module_data[module] + [0] * (max_params_per_module - len(module_data[module]))
                heatmap_data.append(row[:max_params_per_module])
            
            # 繪製熱度圖
            sns.heatmap(heatmap_data, 
                       yticklabels=modules,
                       xticklabels=[f'Param_{i}' for i in range(max_params_per_module)],
                       cmap='RdYlBu_r',
                       ax=ax,
                       cbar_kws={'label': 'Parameter Change Magnitude'})
            
            ax.set_title(f'Parameter Changes at Step {step}')
            plt.tight_layout()
            
            # 保存到 TensorBoard
            self.writer.add_figure('ParameterHeatmap', fig, step)
            plt.close(fig)
    
    def log_architecture_summary(self, step: int):
        """記錄架構摘要"""
        total_params = 0
        trainable_params = 0
        car_params = 0
        infinity_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
                if any(car_key in name for car_key in ['car_control', 'car_var', 'car_blocks', 'car_skip']):
                    car_params += param.numel()
                else:
                    infinity_params += param.numel()
        
        # 記錄參數統計
        self.writer.add_scalar('Architecture/TotalParams', total_params, step)
        self.writer.add_scalar('Architecture/TrainableParams', trainable_params, step)
        self.writer.add_scalar('Architecture/CARParams', car_params, step)
        self.writer.add_scalar('Architecture/InfinityParams', infinity_params, step)
        
        # 計算比例
        if trainable_params > 0:
            car_ratio = car_params / trainable_params
            infinity_ratio = infinity_params / trainable_params
            self.writer.add_scalar('Architecture/CARRatio', car_ratio, step)
            self.writer.add_scalar('Architecture/InfinityRatio', infinity_ratio, step)
    
    def close(self):
        """關閉 TensorBoard writer"""
        if hasattr(self, 'writer'):
            self.writer.close()

# 使用示例
def create_tensorboard_monitor(model, log_dir="./tensorboard_logs"):
    """創建 TensorBoard 監控器"""
    return TensorBoardParameterMonitor(model, log_dir)
