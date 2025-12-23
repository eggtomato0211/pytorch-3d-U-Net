import torch
from collections import OrderedDict

# --- パスを自身の環境に合わせて設定してください ---
# 元のチェックポイントファイルのパス
original_path = r'C:\Users\Owner\mizusaki\pytorch-3dunet\CHECKPOINT_DIR\best_checkpoint.pytorch'
# 修正後に保存する新しいチェックポイントファイルのパス
fixed_path = r'C:\Users\Owner\mizusaki\pytorch-3dunet\CHECKPOINT_DIR\best_checkpoint_fixed.pytorch'
# ----------------------------------------------------

# 元のチェックポイントをCPUに読み込む
checkpoint = torch.load(original_path, map_location='cpu')

# 新しいstate_dictを準備
new_state_dict = OrderedDict()

# 元のstate_dictの各キーに 'module.' 接頭辞を追加する
for k, v in checkpoint['model_state_dict'].items():
    if not k.startswith('module.'):
        name = 'module.' + k  # 'module.' を先頭に追加
        new_state_dict[name] = v
    else:
        new_state_dict[k] = v # 既に接頭辞がある場合はそのまま

# チェックポイントの model_state_dict を新しいものに更新
checkpoint['model_state_dict'] = new_state_dict

# 修正したチェックポイントを新しいファイルに保存
torch.save(checkpoint, fixed_path)

print(f"修正済みのチェックポイントを {fixed_path} に保存しました。")