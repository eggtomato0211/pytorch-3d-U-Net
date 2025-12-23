import torch
import argparse

def inspect(path):
    print(f"Loading {path}...")
    try:
        state = torch.load(path, map_location='cpu')
        if 'model_state_dict' in state:
            keys = list(state['model_state_dict'].keys())
        else:
            keys = list(state.keys())
        
        print(f"Total keys: {len(keys)}")
        print("First 20 keys:")
        for k in keys[:20]:
            print(k)

        print("\nAll Decoder Keys:")
        for k in keys:
            if 'decoders' in k:
                print(k)
        
        # Check specific layers to deduce structure
        print("\nChecking for specific layers...")
        if any('encoders' in k for k in keys):
            print(" - Has 'encoders' (UNet3D)")
        
        # Count encoders
        encoders = set()
        for k in keys:
            if 'encoders' in k:
                parts = k.split('.')
                # encoders.0.basic_module...
                if parts[1].isdigit():
                    encoders.add(int(parts[1]))
        print(f" - Encoders count: {len(encoders)} (Indices: {sorted(list(encoders))})")
        
        # Check input channels from first conv weight
        if 'encoders.0.basic_module.SingleConv1.conv.weight' in keys:
            w = state['model_state_dict']['encoders.0.basic_module.SingleConv1.conv.weight']
            print(f" - First Conv Weight Shape: {w.shape} (In: {w.shape[1]}, Out: {w.shape[0]})")
            
        # Check decoders
        decoders = set()
        for k in keys:
            if 'decoders' in k:
                parts = k.split('.')
                if parts[1].isdigit():
                    decoders.add(int(parts[1]))
        print(f" - Decoders count: {len(decoders)} (Indices: {sorted(list(decoders))})")

        # Check final conv
        if 'final_conv.weight' in keys:
            print(" - Has final_conv")
        else:
            print(" - MISSING final_conv")
            
        # Check layer order hints
        if 'encoders.0.basic_module.SingleConv1.groupnorm.weight' in keys:
            print(" - Has GroupNorm (likely 'gcr' or similar)")
        else:
            print(" - NO GroupNorm found (could be 'cr' or similar)")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()
    inspect(args.path)
