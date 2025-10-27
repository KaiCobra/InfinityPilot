
def debug_parameter_freeze_status(gpt_wo_ddp):
    """Debug utility to print out the freeze status of model parameters."""
    print("[DEBUG] Verifying freeze status after initial freeze:")
    infinity_trainable = sum(1 for name, param in gpt_wo_ddp.named_parameters() 
                        if param.requires_grad and not any(car_prefix in name for car_prefix in ['car_', 'control_']))
    car_trainable = sum(1 for name, param in gpt_wo_ddp.named_parameters() 
                    if param.requires_grad and any(car_prefix in name for car_prefix in ['car_', 'control_']))
    print(f"  Infinity parameters with requires_grad=True: {infinity_trainable}")
    print(f"  CAR parameters with requires_grad=True: {car_trainable}")

    if infinity_trainable > 0:
        print(f"  [WARNING] Found {infinity_trainable} Infinity parameters still trainable after DDP!")
        # 打印这些参数的名称
        for name, param in gpt_wo_ddp.named_parameters():
            if param.requires_grad and not any(car_prefix in name for car_prefix in ['car_', 'control_']):
                print(f"    - {name}")

    del infinity_trainable, car_trainable

def debug_parameter_number(gpt_wo_ddp):
    """Debug utility to print out the total number of parameters."""
    infinity_params = sum(p.numel() for name, p in gpt_wo_ddp.named_parameters() 
                         if not any(car_prefix in name for car_prefix in ['car_', 'control_']))
    car_params = sum(p.numel() for name, p in gpt_wo_ddp.named_parameters() 
                    if any(car_prefix in name for car_prefix in ['car_', 'control_']))
    trainable_infinity = sum(p.numel() for name, p in gpt_wo_ddp.named_parameters() 
                           if not any(car_prefix in name for car_prefix in ['car_', 'control_']) and p.requires_grad)
    trainable_car = sum(p.numel() for name, p in gpt_wo_ddp.named_parameters() 
                       if any(car_prefix in name for car_prefix in ['car_', 'control_']) and p.requires_grad)
    
    print(f'[PT][#para] Infinity: {infinity_params/1e6:.2f}M (trainable: {trainable_infinity/1e9:.2f}B)')
    print(f'[PT][#para] CAR: {car_params/1e6:.2f}M (trainable: {trainable_car/1e6:.2f}M)')
    del infinity_params, car_params, trainable_infinity, trainable_car
    print()
    