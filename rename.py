import os

# 対象ディレクトリ
base_path = r"C:\Users\Owner\mizusaki\pytorch-3dunet\checkpoint\32x32x128"

# フォルダ名変更スクリプト
def rename_folders(base_path):
    for folder_name in os.listdir(base_path):
        old_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(old_path):
            continue

        # # 名前を変更する処理
        # 1. 先頭の `32x32x128_0-1_` を削除
        new_name = folder_name.replace("32x32x128_0-1_", "")
        # # 2. `start=` を `fm=` に置換
        # new_name = new_name.replace("start=", "fm=")
        # # 3. `patch=128` を追加
        # new_name = f"patch=128_{new_name}"

        # 新しいパス
        new_path = os.path.join(base_path, new_name)

        # フォルダ名を変更
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
        except Exception as e:
            print(f"Error renaming {old_path}: {e}")

# 実行
rename_folders(base_path)
