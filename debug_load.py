import torch
import yaml
import importlib
from pytorch3dunet.unet3d.model import UNet3D

def debug_load():
    config_path = 'project/configs/train_config.yaml'
    checkpoint_path = "D:/nosaka/checkpoint/clean/best_checkpoint.pytorch"
    
    # Load config
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_config = config['model']
    print(f"Model Config: {model_config}")
    
    # Instantiate Model
    print("Instantiating UNet3D...")
    try:
        model = UNet3D(**model_config)
    except Exception as e:
        # Some params might need filtering if not in __init__
        # But UNet3D usually takes kwargs
        # Or specifically: in_channels, out_channels, f_maps, layer_order, num_groups, is_segmentation
        # Let's try passing arguments explicitly if kwargs fail
        model = UNet3D(
            in_channels=model_config['in_channels'],
            out_channels=model_config['out_channels'],
            f_maps=model_config['f_maps'],
            layer_order=model_config['layer_order'],
            num_groups=model_config['num_groups'],
            is_segmentation=model_config['is_segmentation']
        )
    
    print("Model instantiated.")
    print(f"Model keys example: {list(model.state_dict().keys())[:5]}")
    
    # Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in state:
        state_dict = state['model_state_dict']
    else:
        state_dict = state
        
    print(f"Checkpoint keys example: {list(state_dict.keys())[:5]}")
    
    # Attempt Load
    print("Attempting load_state_dict...")
    try:
        model.load_state_dict(state_dict, strict=True)
        print("SUCCESS! Model loaded cleanly.")
    except RuntimeError as e:
        print("\n!!! LOAD FAILED !!!")
        msg = str(e)
        if "Missing key(s)" in msg:
            print("\n--- Missing Keys (Model expects these, Checkpoint doesn't have) ---")
            # Parse from message or checks
            # Just print the counts
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())
            missing = model_keys - ckpt_keys
            print(f"Count: {len(missing)}")
            print("Examples:", list(missing)[:10])
            
        if "Unexpected key(s)" in msg:
            print("\n--- Unexpected Keys (Checkpoint has these, Model doesn't expect) ---")
            ckpt_keys = set(state_dict.keys())
            model_keys = set(model.state_dict().keys())
            unexpected = ckpt_keys - model_keys
            print(f"Count: {len(unexpected)}")
            print("Examples:", list(unexpected)[:10])
            
        if "size mismatch" in msg:
            print("\n--- Size Mismatch ---")
            print(msg)

if __name__ == "__main__":
    debug_load()
