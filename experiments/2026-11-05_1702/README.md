# 第2回：回答中の活性と疎な編集

固定条件: `ROUND2_PROTOCOL.md`。入力: `data/round2/`。結果: `results/round2/`。生ログ: `runs/round2/`。モデル: `models/`。

`data/*.json` は第1回との重複検査用の参照入力。内部の `round2/` 名は凍結済みソースとハッシュを保持するため残している。

日時はJSTの実行記録ファイル作成時刻に基づく。[実行手順と共有環境](../../README.md)

```powershell
# リポジトリルートで実行
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 -m pytest -q
```
