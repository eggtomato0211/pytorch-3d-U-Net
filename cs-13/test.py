def check_cuda_availability():
    """
    Check CUDA availability and return detailed information about the GPU setup.
    
    Returns:
        dict: Dictionary containing CUDA and GPU information
    """
    import torch
    
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': 0,
        'cuda_device_names': [],
        'cuda_version': None,
        'current_device': None,
        'device_properties': {}
    }
    
    if info['cuda_available']:
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_version'] = torch.version.cuda
        info['current_device'] = torch.cuda.current_device()
        
        # Get information about each CUDA device
        for i in range(info['cuda_device_count']):
            props = torch.cuda.get_device_properties(i)
            info['cuda_device_names'].append(props.name)
            info['device_properties'][i] = {
                'name': props.name,
                'total_memory': f"{props.total_memory / 1024**2:.2f} MB",
                'compute_capability': f"{props.major}.{props.minor}",
                'multi_processor_count': props.multi_processor_count
            }
    
    return info

def print_cuda_info():
    """
    Print formatted CUDA availability information.
    """
    info = check_cuda_availability()
    
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"Number of CUDA devices: {info['cuda_device_count']}")
        print(f"Current device: {info['current_device']}")
        
        print("\nGPU Information:")
        for device_id, props in info['device_properties'].items():
            print(f"\nDevice {device_id}:")
            print(f"  Name: {props['name']}")
            print(f"  Total Memory: {props['total_memory']}")
            print(f"  Compute Capability: {props['compute_capability']}")
            print(f"  Multi Processors: {props['multi_processor_count']}")

import torch
torch.cuda.empty_cache()
print(torch.cuda.is_available())  # True なら GPU が使用可能
print(torch.cuda.memory_allocated())  # 使用中のメモリ量
print(torch.cuda.memory_reserved())  # 予約済みメモリ量

checkpoint_path = r"C:\Users\Owner\mizusaki\pytorch-3dunet\checkpoint\32x32x128\patch=64_stride=16_fm=16_valpatch=64\best_checkpoint.pytorch"

checkpoint = torch.load(checkpoint_path, map_location='cpu')

print("Checkpoint のキー一覧:", checkpoint.keys())
