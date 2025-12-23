import argparse
import glob
import os
from typing import List

import h5py
import yaml


def expand_paths(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ('*.h5', '*.hdf5', '*.hdf', '*.hd5'):
                out.extend(glob.glob(os.path.join(p, ext)))
        else:
            out.append(p)
    return out


def load_patch_shape(cfg, phase: str):
    try:
        return cfg['loaders'][phase]['slice_builder']['patch_shape']
    except Exception:
        return None


def fits_patch(shape_zyx, patch):
    if shape_zyx is None or patch is None:
        return True
    return all(s >= p for s, p in zip(shape_zyx, patch))


def main():
    ap = argparse.ArgumentParser(description='Validate val dataset availability and shapes against config')
    ap.add_argument('--config', default='project/configs/train_config.yaml')
    ap.add_argument('--phase', choices=['val', 'test'], default='val')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    phase_cfg = cfg['loaders'][args.phase]
    file_paths = expand_paths(phase_cfg['file_paths'])
    patch = load_patch_shape(cfg, args.phase)

    print(f"Phase={args.phase} files found: {len(file_paths)}")
    if not file_paths:
        print('ERROR: no files found. Check loaders.{phase}.file_paths in config.')
        return 2

    ok = 0
    for fp in file_paths:
        try:
            with h5py.File(fp, 'r') as f:
                keys = list(f.keys())
                if 'raw' not in keys or (args.phase != 'test' and 'label' not in keys):
                    print(f"WARN: missing keys in {fp}. keys={keys}")
                    continue
                raw = f['raw']
                shape = raw.shape[-3:]  # D,H,W possibly with channel first
                if not fits_patch(shape, patch):
                    print(f"WARN: patch_shape {patch} does not fit raw shape {shape} for {fp}")
                else:
                    ok += 1
        except Exception as e:
            print(f"ERROR: failed to open {fp}: {e}")

    if ok == 0:
        print('ERROR: no usable files (with required keys and shapes)')
        return 3

    print(f"OK: usable files: {ok}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

