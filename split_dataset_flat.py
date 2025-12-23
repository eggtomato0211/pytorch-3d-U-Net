import os
import shutil
import random
import glob

def split_dataset():
    source_root = "D:/nosaka/data/3d-holography_output"
    train_dir = os.path.join(source_root, "train")
    val_dir = os.path.join(source_root, "val")
    test_dir = os.path.join(source_root, "test")
    
    # Clean/Create dirs
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
        
    print(f"Scanning {source_root} for h5 files...")
    # Find all h5 files recursively (excluding the target dirs if they already exist/have files)
    all_files = []
    for root, dirs, files in os.walk(source_root):
        # Skip target dirs to avoid double counting if re-running
        if root.startswith(train_dir) or root.startswith(val_dir) or root.startswith(test_dir):
            continue
            
        for f in files:
            if f.endswith(".h5"):
                all_files.append(os.path.join(root, f))
                
    total = len(all_files)
    print(f"Found {total} files.")
    
    if total < 1250:
        print("Warning: Not enough files for requested split (800/200/250). Using ratios instead.")
        # Fallback to ratios if needed, but intended is strict numbers
    
    random.shuffle(all_files)
    
    # Requested counts
    n_train = 1000
    n_val = 250
    n_test = 250
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train+n_val]
    test_files = all_files[n_train+n_val:n_train+n_val+n_test]
    
    print(f"Splitting: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    def move_files(file_list, dest_dir):
        count = 0
        for src in file_list:
            fname = os.path.basename(src)
            dst = os.path.join(dest_dir, fname)
            try:
                shutil.move(src, dst)
                count += 1
            except Exception as e:
                print(f"Error moving {src}: {e}")
        print(f"Moved {count} files to {dest_dir}")

    move_files(train_files, train_dir)
    move_files(val_files, val_dir)
    move_files(test_files, test_dir)
    
    print("Done splitting.")

if __name__ == "__main__":
    split_dataset()
