# SceneTxtVAR AI Development Guide

## Project Overview
SceneTxtVAR is a controllable autoregressive image generation framework built on **Infinity** and **CAR** (Controllable AutoRegressive) architectures. The project combines VAR-style visual autoregressive modeling with ControlNet-style conditional control for precise image generation.

## Core Architecture

### InfinityPilot: The Central Model
The `InfinityPilot` class (`infinity/models/infinity_pilot.py`) extends the base `Infinity` transformer with CAR control modules:

```python
# Model instantiation pattern
model = InfinityPilot(
    infinity_base_model=pretrained_infinity,  # Load base weights
    init_car_modules=True,                    # Initialize control modules  
    freeze_infinity=True,                     # Freeze base, train only CAR
    car_depth=8,                             # Control block depth (≤ depth//2)
    **infinity_kwargs
)
```

### Weight Management Strategy
The architecture uses **separated weight loading** similar to T5's encoder-decoder pattern:

- **Infinity Base**: Frozen pretrained transformer (`infinity_base_weights.pth`)
- **CAR Modules**: Trainable control components (`car_weights.pth`)
- **Parameter Detection**: Auto-detects checkpoint architecture (`shared_aln`, `ada_gss`) and adjusts model accordingly

### VAE Integration
The project supports multiple VAE backends:

- **BSQ-VAE**: Modern multi-scale quantization (`infinity/models/bsq_vae/`)
- **VQ-VAE**: Legacy vector quantization (`CAR/models/vqvae.py`)
- **Scale Schedules**: Multi-resolution generation with format `[(pn, h, w), ...]`

## Development Workflows

### Training InfinityPilot
```bash
# Use trainer_pilot.py - the main training script
python trainer_pilot.py --config configs/config_infinity_pilot.py
```

Key training concepts:
- **Parameter Freezing**: Only CAR modules (`car_*`, `control_*`) are trainable
- **Control Tensors**: Pass condition images as `control_tensors` list
- **Memory Optimization**: Use `block_chunks` for FSDP, gradient checkpointing for large models

### Model Registry Pattern
Models are registered using timm's `@register_model` decorator:

```python
@register_model  
def infinity_pilot_layer16(depth=16, embed_dim=1152, **kwargs):
    return InfinityPilot(depth=depth, embed_dim=embed_dim, ...)
```

Available sizes: `layer12/16/24/32/40/48`, `2b`, `20b`

### Inference Workflows
```python
# Generation pipeline
from tools.run_infinity_pilot import *

# Load model with auto-architecture detection
vae, model = load_model_and_vae(args)

# Generate with control
images = model.autoregressive_infer_cfg(
    control_tensors=control_conditions,  # List of condition tensors per scale
    scale_schedule=[(1,16,16), (1,32,32), ...],
    cfg=4.0  # Classifier-free guidance strength
)
```

## Critical File Organization

### Core Model Files
- `infinity/models/infinity_pilot.py` - Main InfinityPilot implementation
- `infinity/models/infinity_new.py` - Base Infinity transformer 
- `infinity/utils/load.py` - Model/VAE loading utilities with architecture detection

### Training Infrastructure  
- `trainer_pilot.py` - Primary training script with CAR parameter tracking
- `configs/config_infinity_pilot.py` - Centralized training configuration
- `infinity/utils/amp_opt.py` - Mixed precision optimization wrapper

### Generation Tools
- `tools/run_infinity_pilot.py` - Inference pipeline and utilities
- `gen_infinity_pilot.py` - High-level generation script
- `scripts/infer.sh` - Shell script templates for batch inference

## Debugging and Memory Management

### Architecture Mismatch Resolution
The model auto-detects checkpoint architectures and adjusts parameters:

```python
# Common mismatch patterns (handled automatically)
has_ada_gss = any('ada_gss' in k for k in checkpoint.keys())
has_shared_ada_lin = any('shared_ada_lin' in k for k in checkpoint.keys())
# Auto-sets shared_aln=True/False accordingly
```

### Memory Optimization Patterns
- **FSDP Support**: Use `block_chunks` parameter for distributed training
- **Gradient Checkpointing**: Set `checkpointing='full-block'` for memory efficiency  
- **VAE Memory**: Use `encode_for_raw_features()` for multi-scale encoding without full quantization

### Parameter Monitoring
Use the built-in parameter visualizer:

```python
from parameter_visualizer import ParameterChangeVisualizer
visualizer = ParameterChangeVisualizer(model)
# Tracks CAR vs Infinity parameter changes during training
```

## Data Flow and Integration Points

### Multi-Scale Generation Pipeline
1. **Condition Processing**: `ControlConditionEmbedding` processes control images 
2. **VAE Encoding**: Multi-scale feature extraction via `encode_for_raw_features()`
3. **Autoregressive Generation**: Scale-by-scale token prediction with control injection
4. **Skip Connections**: CAR features merged with Infinity layers via `car_skip_norm/linear`

### Control Tensor Format
Control tensors must match scale schedule:
```python
control_tensors = [
    torch.randn(B, 3, 16, 16),   # Scale 1: 16x16
    torch.randn(B, 3, 32, 32),   # Scale 2: 32x32  
    # ... one tensor per scale in schedule
]
```

## Common Pitfalls and Solutions

- **Missing CAR Modules**: Call `init_car_modules_if_needed()` before training
- **FSDP Format Conversion**: Automatic conversion from `block_chunks.X.module.Y` to `blocks.Z` format
- **Control Tensor Mismatch**: Ensure control_tensors length matches scale_schedule length
- **Memory Issues**: Use smaller `car_depth` (4-8) and enable gradient checkpointing
- **Architecture Detection**: Let the model auto-detect `shared_aln` from checkpoints rather than forcing it

Focus on understanding the **dual-weight system** (Infinity + CAR) and **multi-scale control injection** - these are the key innovations that differentiate this from standard VAR/diffusion approaches.
