import torch
import torch.nn as nn
from typing import Dict, List, Optional
from collections import defaultdict

class NaNDetector:
    """Detect which layer produces NaN in forward/backward pass"""
    
    def __init__(self, model: nn.Module, check_backward: bool = True, verbose: bool = True):
        self.model = model
        self.check_backward = check_backward
        self.verbose = verbose
        self.hooks = []
        self.nan_layers: List[str] = []
        self.nan_info: Dict[str, List[str]] = defaultdict(list)
        self.forward_count = 0
        self.backward_count = 0
        
    def register_hooks(self):
        """Register forward and backward hooks on all LayerNorm layers"""
        norm_count = 0
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                # Forward hook
                hook = module.register_forward_hook(
                    self._make_forward_hook(name)
                )
                self.hooks.append(hook)
                
                # Backward hook
                if self.check_backward:
                    hook = module.register_full_backward_hook(
                        self._make_backward_hook(name)
                    )
                    self.hooks.append(hook)
                norm_count += 1
        
        if self.verbose:
            print(f"🔍 NaN detector registered on {norm_count} normalization layers")
    
    def _make_forward_hook(self, name: str):
        def hook(module, input, output):
            self.forward_count += 1
            
            # Check input
            if isinstance(input, tuple):
                for i, inp in enumerate(input):
                    if torch.is_tensor(inp):
                        if torch.isnan(inp).any():
                            msg = f"❌ NaN in {name} forward INPUT[{i}] | shape={inp.shape}"
                            print(msg)
                            self.nan_layers.append(f"{name}_input_{i}")
                            self.nan_info[name].append(f"forward_input_{i}")
                        elif torch.isinf(inp).any():
                            msg = f"❌ Inf in {name} forward INPUT[{i}] | shape={inp.shape}"
                            print(msg)
                            self.nan_info[name].append(f"forward_input_{i}_inf")
                        else:
                            # Check for extreme values
                            max_val = inp.abs().max().item()
                            if max_val > 1e4:
                                if self.verbose and self.forward_count < 100:  # Only log first 100
                                    print(f"⚠️  Extreme in {name} INPUT[{i}] | max={max_val:.2e}")
            
            # Check output
            if torch.is_tensor(output):
                if torch.isnan(output).any():
                    msg = f"❌ NaN in {name} forward OUTPUT | shape={output.shape}"
                    print(msg)
                    self.nan_layers.append(f"{name}_output")
                    self.nan_info[name].append("forward_output")
                elif torch.isinf(output).any():
                    msg = f"❌ Inf in {name} forward OUTPUT | shape={output.shape}"
                    print(msg)
                    self.nan_info[name].append("forward_output_inf")
        return hook
    
    def _make_backward_hook(self, name: str):
        def hook(module, grad_input, grad_output):
            self.backward_count += 1
            
            # Check grad_output (upstream gradient)
            if grad_output is not None:
                for i, grad in enumerate(grad_output):
                    if torch.is_tensor(grad):
                        if torch.isnan(grad).any():
                            msg = f"❌ NaN in {name} backward GRAD_OUTPUT[{i}] (from upstream)"
                            print(msg)
                            self.nan_layers.append(f"{name}_grad_output_{i}")
                            self.nan_info[name].append(f"backward_grad_output_{i}")
                        elif torch.isinf(grad).any():
                            print(f"❌ Inf in {name} backward GRAD_OUTPUT[{i}]")
                            self.nan_info[name].append(f"backward_grad_output_{i}_inf")
            
            # Check grad_input (gradient to pass down) - THIS IS THE KEY!
            if grad_input is not None:
                for i, grad in enumerate(grad_input):
                    if torch.is_tensor(grad):
                        if torch.isnan(grad).any():
                            msg = f"🔴 NaN in {name} backward GRAD_INPUT[{i}] - CULPRIT FOUND!"
                            print(msg)
                            print(f"   Module type: {type(module).__name__}")
                            print(f"   Module eps: {getattr(module, 'eps', 'N/A')}")
                            print(f"   Weight shape: {module.weight.shape if hasattr(module, 'weight') else 'N/A'}")
                            self.nan_layers.append(f"{name}_grad_input_{i}")
                            self.nan_info[name].append(f"backward_grad_input_{i}_CULPRIT")
                        elif torch.isinf(grad).any():
                            print(f"❌ Inf in {name} backward GRAD_INPUT[{i}]")
                            self.nan_info[name].append(f"backward_grad_input_{i}_inf")
        return hook
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if self.verbose:
            print(f"🧹 Removed NaN detector hooks")
    
    def get_nan_layers(self) -> List[str]:
        return self.nan_layers
    
    def get_summary(self) -> str:
        """Get summary of NaN detections"""
        if not self.nan_info:
            return "✅ No NaN detected"
        
        summary = [f"❌ NaN detected in {len(self.nan_info)} layers:"]
        for layer_name, issues in self.nan_info.items():
            summary.append(f"  - {layer_name}: {', '.join(issues)}")
        return "\n".join(summary)
    
    def clear(self):
        """Clear detection history"""
        self.nan_info.clear()
        self.nan_layers.clear()
        self.forward_count = 0
        self.backward_count = 0