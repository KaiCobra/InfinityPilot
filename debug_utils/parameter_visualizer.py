import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, OrderedDict
import time
from typing import Dict, List, Tuple
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

class ParameterChangeVisualizer:
    """
    可視化模型參數變化的工具，特別針對 InfinityPilot control 訓練
    """
    def __init__(self, model, save_dir="./param_visualizations"):
        self.model = model
        self.save_dir = save_dir
        self.initial_params = {}
        self.param_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        self.iteration_count = 0
        
        # 獲取初始參數快照
        self._capture_initial_params()
        
        # 創建保存目錄
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 定義模組分組
        self.module_groups = {
            'Infinity_Embedding': ['word_embed', 'pos_start', 'pos_1LC', 'lvl_embed'],
            'Control_Encoder': ['control_encoder', 'control_token_norm', 'car_control_encoder', 'car_control_norm'],
            'Control_Gates': ['control_scale_gate_mlp', 'control_scale_gate_bias', 'control_block_gates', 'car_scale_gate_mlp', 'car_scale_gate_bias', 'car_block_fusion_gates'],
            'Control_Others': ['control_', 'car_'],
            'Infinity_Blocks': ['blocks'],
            'Infinity_Input': ['head_nm', 'head'],
            
        }
        
        # 創建顏色映射
        self.cmap = LinearSegmentedColormap.from_list(
            'param_change', 
            ['blue', 'white', 'red'], 
            N=256
        )
    
    def _capture_initial_params(self):
        """捕獲初始參數狀態 - 包含所有参数，不管是否可训练"""
        for name, param in self.model.named_parameters():
            self.initial_params[name] = param.data.clone().detach()

    
    def _get_param_change_magnitude(self, name: str, current_param: torch.Tensor) -> float:
        """計算參數變化幅度"""
        if name not in self.initial_params:
            return 0.0
        
        change = torch.norm(current_param - self.initial_params[name]).item()
        param_norm = torch.norm(self.initial_params[name]).item()
        
        # 相對變化率
        if param_norm > 1e-8:
            return change / param_norm
        else:
            return change
    
    def _categorize_parameter(self, param_name: str) -> str:
        """將參數分類到對應的模組"""
        for group_name, keywords in self.module_groups.items():
            for keyword in keywords:
                if keyword in param_name:
                    return group_name
        return 'Other'
    
    def update(self, iteration: int = None):
        """更新參數變化記錄 - 只记录可训练参数的变化"""
        if iteration is None:
            iteration = self.iteration_count
        
        current_changes = {}
        current_grads = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:  # 只记录可训练参数的变化
                # 記錄參數變化
                change_mag = self._get_param_change_magnitude(name, param.data)
                current_changes[name] = change_mag
                
                # 記錄梯度大小
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    current_grads[name] = grad_norm
                else:
                    current_grads[name] = 0.0
        
        self.param_history[iteration] = current_changes
        self.gradient_history[iteration] = current_grads
        self.iteration_count += 1

        # 清理中間變量以節省內存
        del current_changes
        del current_grads
        torch.cuda.empty_cache()
    
    def plot_module_heatmap(self, iteration: int = None, figsize=(15, 10)):
        """繪製模組級別的參數變化熱度圖"""
        if iteration is None:
            iteration = max(self.param_history.keys()) if self.param_history else 0
        
        if iteration not in self.param_history:
            print(f"No data for iteration {iteration}")
            return
        
        # 按模組聚合數據 - 包含所有参数
        module_changes = defaultdict(list)
        module_grads = defaultdict(list)
        module_requires_grad = defaultdict(list)
        
        # 遍历所有参数
        for name, param in self.model.named_parameters():
            module = self._categorize_parameter(name)
            
            # 只有可训练参数才有变化记录
            if param.requires_grad and name in self.param_history[iteration]:
                change = self.param_history[iteration][name]
                module_changes[module].append(change)
                
                if name in self.gradient_history[iteration]:
                    module_grads[module].append(self.gradient_history[iteration][name])
            else:
                # 冻结参数的变化为0
                module_changes[module].append(0.0)
                module_grads[module].append(0.0)
            
            # 记录参数的 requires_grad 状态
            module_requires_grad[module].append(param.requires_grad)
        
        # 計算模組級別的統計
        module_stats = {}
        for module in module_changes:
            changes = module_changes[module]
            grads = module_grads[module]
            requires_grad_list = module_requires_grad[module]
            
            # 判断模块冻结状态
            all_frozen = not any(requires_grad_list)
            
            module_stats[module] = {
                'param_change_mean': np.mean(changes) if changes else 0,
                'param_change_max': np.max(changes) if changes else 0,
                'grad_mean': np.mean(grads) if grads else 0,
                'grad_max': np.max(grads) if grads else 0,
                'param_count': len(changes),
                'frozen': all_frozen
            }
        
        # 創建可視化
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        modules = list(module_stats.keys())
        
        # 1. 參數變化均值
        param_changes = [module_stats[m]['param_change_mean'] for m in modules]
        bars1 = ax1.bar(range(len(modules)), param_changes, 
                       color=['red' if 'CAR' in m else 'blue' for m in modules])
        ax1.set_title('Parameter Change Magnitude (Mean)')
        ax1.set_ylabel('Relative Change')
        ax1.set_xticks(range(len(modules)))
        ax1.set_xticklabels(modules, rotation=45, ha='right')
        
        # 添加數值標籤
        for i, (bar, val) in enumerate(zip(bars1, param_changes)):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_changes)*0.01,
                        f'{val:.2e}', ha='center', va='bottom', fontsize=8)
        
        # 2. 梯度大小
        grad_norms = [module_stats[m]['grad_mean'] for m in modules]
        bars2 = ax2.bar(range(len(modules)), grad_norms,
                       color=['red' if 'CAR' in m else 'blue' for m in modules])
        ax2.set_title('Gradient Magnitude (Mean)')
        ax2.set_ylabel('Gradient Norm')
        ax2.set_xticks(range(len(modules)))
        ax2.set_xticklabels(modules, rotation=45, ha='right')
        
        # 3. 參數數量
        param_counts = [module_stats[m]['param_count'] for m in modules]
        bars3 = ax3.bar(range(len(modules)), param_counts,
                       color=['red' if 'CAR' in m else 'blue' for m in modules])
        ax3.set_title('Parameter Count per Module')
        ax3.set_ylabel('Number of Parameters')
        ax3.set_xticks(range(len(modules)))
        ax3.set_xticklabels(modules, rotation=45, ha='right')
        
        # 4. 凍結狀態檢查 - 使用 requires_grad 状态
        frozen_status = [1 if module_stats[m]['frozen'] else 0 for m in modules]
        
        bars4 = ax4.bar(range(len(modules)), frozen_status,
                       color=['green' if status else 'orange' for status in frozen_status])
        ax4.set_title('Freeze Status (1=Frozen, 0=Training)')
        ax4.set_ylabel('Frozen Status')
        ax4.set_xticks(range(len(modules)))
        ax4.set_xticklabels(modules, rotation=45, ha='right')
        ax4.set_ylim(0, 1.2)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/module_analysis_iter.png', dpi=300, bbox_inches='tight')
        # plt.show()

                # 清理中間變量以節省內存
        del module_changes
        del module_grads
        del module_requires_grad
        del module_stats
        torch.cuda.empty_cache()

    def plot_architecture_diagram(self, iteration: int = None):
        """繪製架構圖與參數變化"""
        if iteration is None:
            iteration = max(self.param_history.keys()) if self.param_history else 0
        
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        
        # 定義架構布局
        architecture_layout = {
            'Input': {'pos': (1, 9), 'size': (2, 1), 'color': 'lightgray'},
            'Text_Encoder': {'pos': (1, 7), 'size': (2, 1), 'color': 'lightblue'},
            'VAE_Encoder': {'pos': (1, 5), 'size': (2, 1), 'color': 'lightgreen'},
            
            'Infinity_Embedding': {'pos': (5, 9), 'size': (3, 1), 'color': 'blue'},
            'Infinity_Blocks': {'pos': (5, 7), 'size': (3, 2), 'color': 'blue'},
            'Infinity_Output': {'pos': (5, 4), 'size': (3, 1), 'color': 'blue'},
            
            'CAR_Control': {'pos': (10, 9), 'size': (2, 1), 'color': 'red'},
            'CAR_VAR': {'pos': (10, 7), 'size': (2, 1), 'color': 'red'},
            'CAR_Blocks': {'pos': (10, 5), 'size': (2, 2), 'color': 'red'},
            'CAR_Skip': {'pos': (10, 3), 'size': (2, 1), 'color': 'red'},
            
            'Output': {'pos': (14, 6), 'size': (2, 1), 'color': 'lightgray'}
        }
        
        # 獲取模組參數變化
        module_changes = defaultdict(list)
        if iteration in self.param_history:
            for param_name, change in self.param_history[iteration].items():
                module = self._categorize_parameter(param_name)
                module_changes[module].append(change)
        
        # 繪製架構圖
        for module_name, layout in architecture_layout.items():
            x, y = layout['pos']
            w, h = layout['size']
            base_color = layout['color']
            
            # 計算參數變化強度
            if module_name in module_changes and module_changes[module_name]:
                change_intensity = np.mean(module_changes[module_name])
                # 正規化強度到 0-1
                max_change = max([np.mean(changes) for changes in module_changes.values() if changes] + [1e-8])
                normalized_intensity = min(change_intensity / max_change, 1.0) if max_change > 0 else 0
                
                # 根據變化強度調整顏色
                if 'CAR' in module_name:
                    if normalized_intensity > 0.1:
                        color = plt.cm.Reds(0.3 + 0.7 * normalized_intensity)
                    else:
                        color = 'lightcoral'  # 淺紅色表示應該訓練但變化很小
                elif 'Infinity' in module_name:
                    if normalized_intensity > 0.01:
                        color = 'orange'  # 橙色表示應該凍結但有變化
                    else:
                        color = plt.cm.Blues(0.3)  # 藍色表示正確凍結
                else:
                    color = base_color
            else:
                color = base_color
            
            # 繪製模組方塊
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                   edgecolor='black', facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            
            # 添加文字標籤
            ax.text(x + w/2, y + h/2, module_name, ha='center', va='center',
                   fontsize=10, fontweight='bold')
            
            # 添加變化數值
            if module_name in module_changes and module_changes[module_name]:
                change_val = np.mean(module_changes[module_name])
                ax.text(x + w/2, y + h/4, f'{change_val:.2e}', ha='center', va='center',
                       fontsize=8, style='italic')
        
        # 繪製連接線
        connections = [
            ('Input', 'Text_Encoder'),
            ('Input', 'VAE_Encoder'),
            ('Text_Encoder', 'Infinity_Embedding'),
            ('VAE_Encoder', 'Infinity_Embedding'),
            ('Infinity_Embedding', 'Infinity_Blocks'),
            ('Infinity_Blocks', 'Infinity_Output'),
            ('Infinity_Blocks', 'CAR_Control'),
            ('Infinity_Blocks', 'CAR_VAR'),
            ('CAR_Control', 'CAR_Blocks'),
            ('CAR_VAR', 'CAR_Blocks'),
            ('CAR_Blocks', 'CAR_Skip'),
            ('Infinity_Output', 'Output'),
            ('CAR_Skip', 'Output')
        ]
        
        for start, end in connections:
            if start in architecture_layout and end in architecture_layout:
                start_pos = architecture_layout[start]['pos']
                start_size = architecture_layout[start]['size']
                end_pos = architecture_layout[end]['pos']
                end_size = architecture_layout[end]['size']
                
                # 計算連接點
                start_x = start_pos[0] + start_size[0]
                start_y = start_pos[1] + start_size[1]/2
                end_x = end_pos[0]
                end_y = end_pos[1] + end_size[1]/2
                
                ax.arrow(start_x, start_y, end_x - start_x - 0.1, end_y - start_y,
                        head_width=0.1, head_length=0.1, fc='gray', ec='gray', alpha=0.6)
        
        ax.set_xlim(0, 17)
        ax.set_ylim(2, 11)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'InfinityPilot Architecture - Parameter Changes at Iteration {iteration}', 
                    fontsize=16, fontweight='bold')
        
        # 添加圖例
        legend_elements = [
            patches.Patch(color='blue', alpha=0.7, label='Infinity (Frozen)'),
            patches.Patch(color='red', alpha=0.7, label='CAR (Training)'),
            patches.Patch(color='orange', alpha=0.7, label='Unexpected Change'),
            patches.Patch(color='lightgray', alpha=0.7, label='External Modules')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/architecture_iter_{iteration}.png', dpi=300, bbox_inches='tight')
        # plt.show()
    
    def save_report(self, iteration: int = None):
        """生成詳細報告"""
        if iteration is None:
            iteration = max(self.param_history.keys()) if self.param_history else 0
        
        report_path = f'{self.save_dir}/parameter_report_iter_{iteration}.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"=== InfinityPilot Parameter Analysis Report ===\n")
            f.write(f"Iteration: {iteration}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if iteration in self.param_history:
                module_changes = defaultdict(list)
                module_grads = defaultdict(list)
                
                for param_name, change in self.param_history[iteration].items():
                    module = self._categorize_parameter(param_name)
                    module_changes[module].append((param_name, change))
                    
                    if param_name in self.gradient_history[iteration]:
                        module_grads[module].append((param_name, self.gradient_history[iteration][param_name]))
                
                for module in sorted(module_changes.keys()):
                    f.write(f"\n--- {module} ---\n")
                    
                    changes = [change for _, change in module_changes[module]]
                    grads = [grad for _, grad in module_grads.get(module, [])]
                    
                    f.write(f"Parameter count: {len(changes)}\n")
                    f.write(f"Mean change: {np.mean(changes):.2e}\n")
                    f.write(f"Max change: {np.max(changes):.2e}\n")
                    f.write(f"Mean gradient: {np.mean(grads):.2e if grads else 0:.2e}\n")
                    
                    # 檢查是否正確凍結
                    is_car = 'CAR' in module
                    has_significant_change = any(change > 1e-6 for change in changes)
                    
                    if is_car and not has_significant_change:
                        f.write("⚠️  WARNING: CAR module not updating!\n")
                    elif not is_car and has_significant_change:
                        f.write("⚠️  WARNING: Infinity module updating!\n")
                    else:
                        f.write("✅ Status: OK\n")
        
        print(f"Report saved to: {report_path}")

# 使用示例函數
def create_parameter_monitor(model, save_dir="./param_visualizations"):
    """創建參數監控器的便捷函數"""
    return ParameterChangeVisualizer(model, save_dir)
