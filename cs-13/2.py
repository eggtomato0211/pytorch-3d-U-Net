import os
import h5py

# 👇 noisy ファイルがあるフォルダ（今指定しているフォルダと同じにしてください）
DATA_DIR = r"C:\Users\Owner\mizusaki\pytorch-3dunet\project\data\train"

# 念のため確認表示
files_to_delete = [
    f for f in os.listdir(DATA_DIR)
    if f.lower().endswith(".h5") and "noisy" in f.lower()
]

print(f"🗂 削除対象ファイル数: {len(files_to_delete)}")
for f in files_to_delete:
    print(" -", f)

if len(files_to_delete) == 0:
    print("✅ 削除対象はありません")
else:
    confirm = input("\n⚠ 本当に削除しますか？ (y/N): ").strip().lower()
    if confirm == "y":
        for f in files_to_delete:
            path = os.path.join(DATA_DIR, f)
            try:
                os.remove(path)
                print(f"✅ 削除: {f}")
            except Exception as e:
                print(f"❌ 削除失敗: {f}, error: {e}")
        print("\n🎯 完了！")
    else:
        print("キャンセルしました ✅")
