import os
import shutil
import sys
sys.path.append("cs-13")
import generate_noisy_data

# Map source to dest
SRC_ROOT = "D:/nosaka/data/3d-holography_output"
DST_ROOT = "D:/nosaka/data/3d-holography_output_noisy"
TARGET_PSNR = 28.0

def main():
    phases = ["train", "val"]
    
    for phase in phases:
        src_dir = os.path.join(SRC_ROOT, phase)
        dst_dir = os.path.join(DST_ROOT, phase)
        
        print(f"Processing {phase}: {src_dir} -> {dst_dir}")
        
        if not os.path.exists(src_dir):
            print(f"Skipping {phase} (Source not found)")
            continue
            
        # Call the processing function from the existing script
        # We need to make sure generate_noisy_data has a callable interface
        # It has `process(data_dir, output_dir, target_psnr)`
        
        try:
            generate_noisy_data.process(src_dir, dst_dir, TARGET_PSNR)
        except Exception as e:
            print(f"Error processing {phase}: {e}")

if __name__ == "__main__":
    # Ensure cs-13 is in path if needed, but since we are in project root...
    # We might need to adjust python path.
    # Actually, simpler to just import file directly or copy function.
    # Let's import assuming correct path setup.
    import sys
    sys.path.append("cs-13")
    # Re-import after path fix
    import generate_noisy_data
    
    main()
