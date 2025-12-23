import argparse, h5py, numpy as np, napari

def load_volume(path, ds):
    with h5py.File(path, "r") as f:
        if ds not in f:
            raise SystemExit(f"Dataset '{ds}' not found. Available: {list(f.keys())}")
        vol = f[ds][:]
    if vol.ndim == 4 and vol.shape[0] == 1:  # (C,Z,Y,X) -> (Z,Y,X)
        vol = vol[0]
    return vol

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--dataset", default="predictions")
    ap.add_argument("--colormap", default="gray")
    args = ap.parse_args()

    vol = load_volume(args.file, args.dataset)
    v = napari.Viewer(ndisplay=3)  # 3D 表示
    v.add_image(vol, name=args.dataset, colormap=args.colormap, rendering="mip")
    napari.run()

if __name__ == "__main__":
    main()
