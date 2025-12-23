import os
import h5py

# ✅ noisy データフォルダを指定
DATA_DIR = r"C:\Users\Owner\Desktop\test250_noisy"

files = [f for f in os.listdir(DATA_DIR) if f.endswith('.h5')]

print(f"📂 対象ファイル数: {len(files)}\n")

missing_raw = []
missing_label = []
error_files = []

for fname in files:
    fpath = os.path.join(DATA_DIR, fname)
    try:
        with h5py.File(fpath, "r") as f:
            has_raw = "raw" in f
            has_label = "label" in f

            if not has_raw:
                missing_raw.append(fname)
            if not has_label:
                missing_label.append(fname)

        print(f"✅ OK: {fname}")

    except Exception as e:
        error_files.append((fname, str(e)))
        print(f"❌ ERROR: {fname}: {e}")

print("\n==== チェック結果まとめ ====")
if missing_raw:
    print(f"⚠ raw が無いファイル: {missing_raw}")
else:
    print("✅ 全ファイル raw あり")

if missing_label:
    print(f"⚠ label が無いファイル: {missing_label}")
else:
    print("✅ 全ファイル label あり")

if error_files:
    print("\n❌ 読み込みエラーがあったファイル:")
    for fname, err in error_files:
        print(f" - {fname}: {err}")
else:
    print("✅ 読み込みエラーなし")

print("\n🎉 チェック完了！")
