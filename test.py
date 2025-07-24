import torch
print(f"--- PyTorch CUDA Diagnostic ---")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available according to PyTorch: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version PyTorch was compiled with: {torch.version.cuda}")
    # 通常，PyTorch 使用的 CUDA 版本 (torch.version.cuda) 應該小於或等於您系統驅動支持的 CUDA 版本。
    # 例如，如果您的驅動支持 CUDA 12.6，PyTorch 用 CUDA 12.1 或 12.4 編譯是正常的。

    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs available: {torch.cuda.device_count()}")

    current_device_id = torch.cuda.current_device()
    print(f"Current GPU ID: {current_device_id}")

    gpu_name = torch.cuda.get_device_name(current_device_id)
    print(f"GPU Name: {gpu_name}")

    major, minor = torch.cuda.get_device_capability(current_device_id)
    print(f"GPU Compute Capability: {major}.{minor}") # RTX A6000 應該是 8.6

    print(f"\n--- Testing Simple GPU Operations ---")
    try:
        # Test tensor creation on GPU
        print("Creating tensor on GPU...")
        a = torch.randn(3, 3, device=f'cuda:{current_device_id}')
        print(f"Tensor 'a' created on device: {a.device}")

        # Test another tensor and basic operation
        print("Creating another tensor and performing matrix multiplication...")
        b = torch.randn(3, 3).to(a.device) # Move to the same device as 'a'
        c = a @ b
        print(f"Matrix multiplication successful. Result 'c' is on device: {c.device}")

        if c.is_cuda:
            print("Test operation confirmed on CUDA.")
        else:
            print(f"WARNING: Test tensor 'c' is on {c.device} instead of CUDA after operation.")

        # Test model movement to GPU (simple dummy model)
        print("Testing model movement to GPU...")
        model = torch.nn.Linear(3,3).to(a.device)
        print(f"Simple nn.Linear model moved to device: {next(model.parameters()).device}")
        print(f"--- GPU Diagnostics Complete ---")

    except Exception as e:
        print(f"ERROR during simple GPU test: {e}")
        print(f"This indicates a problem with basic PyTorch CUDA functionality.")
else:
    print("CUDA is NOT available to PyTorch. Inductor will not be able to use GPU backends.")
    print(f"--- GPU Diagnostics Complete (CUDA Not Available) ---")