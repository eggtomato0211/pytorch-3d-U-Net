import argparse
import glob
import os
from typing import Iterable

import h5py


def iter_files(paths: Iterable[str]):
    for p in paths:
        if os.path.isdir(p):
            for ext in ('*.h5', '*.hdf5', '*.hdf', '*.hd5'):
                yield from glob.glob(os.path.join(p, ext))
        else:
            yield p


def main():
    ap = argparse.ArgumentParser(description='Validate HDF5 files for required keys and shape constraints')
    ap.add_argument('--paths', nargs='+', required=True, help='Files or directories to validate')
    ap.add_argument('--require-label', action='store_true', help='Require label dataset in addition to raw')
    ap.add_argument('--raw-key', default='raw')
    ap.add_argument('--label-key', default='label')
    ap.add_argument('--min-shape', type=int, nargs=3, metavar=('D','H','W'), help='Minimum shape to accept (ZYX)')
    args = ap.parse_args()

    files = list(iter_files(args.paths))
    print(f"Files to check: {len(files)}")
    ok = 0
    for fp in files:
        try:
            with h5py.File(fp, 'r') as f:
                keys = set(f.keys())
                if args.raw_key not in keys:
                    print(f"[MISS] {fp}: missing '{args.raw_key}' (keys={sorted(keys)})")
                    continue
                if args.require_label and args.label_key not in keys:
                    print(f"[MISS] {fp}: missing '{args.label_key}' (keys={sorted(keys)})")
                    continue
                raw = f[args.raw_key]
                shape = raw.shape[-3:]
                if args.min_shape and any(s < m for s, m in zip(shape, args.min_shape)):
                    print(f"[SMALL] {fp}: shape {shape} < min {tuple(args.min_shape)}")
                    continue
                print(f"[OK] {fp}: shape={shape} dtype={raw.dtype}")
                ok += 1
        except Exception as e:
            print(f"[ERR] {fp}: {e}")

    if ok == 0:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

