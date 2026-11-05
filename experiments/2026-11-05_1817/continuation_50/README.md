# 語尾成功率50%を目指す継続実験

**既存100問で56/100（56%）、候補固定後の新規50問で28/50（56%）。今回の50%目標を達成した。** 未編集はどちらも0%。初回モデルの既存100問15%から41ポイント増加した。

[報告書](REPORT.md)に方法、全候補の比較、失敗例、検証を記載する。[集計と各回答](results/report.json)、[選択記録](results/selection.json)、[保存・再ロード検証](results/verification.json)を参照。

選択したモデルは `models/zundamon-edited/`。層19（0始まり）の `mlp.down_proj.weight` だけを編集し、3,597,433要素が変化した。旧15%モデルは親ディレクトリの `models/zundamon-edited/` に保持している。

目標は既存100問の自然な「のだ／なのだ」語尾成功率50%以上。内容の正誤は評価しない。語中切断、引用内だけ、独立した語尾だけは除外し、本文中に自然な文・名詞文が1つでもあれば成功とする。

初回のモデル・結果・報告書は変更しない。全候補は固定revisionの未編集Qwen3.5-0.8Bから構築する。推論時は一般的な日本語指示のみ、greedy、thinking無効、max_new_tokens=128を維持する。学習・勾配・損失フィッティング・最小二乗回帰・embedding/output/final norm編集は禁止。

既存100問と以前の検証・汎化質問は既に結果を見ているため、継続実験では開発用と扱う。新しいheldout50問の編集後生成は候補固定後に行う。

## 初期仮説

初回の末尾方向への強い書き込みは、語尾を単語の途中に挿入し、終了後も反復させた。今回の優先仮説は、回答開始時の口調方針を中間層から変更すること。診断用few-shot文脈で自然な語尾を出せることを確認し、その内部方向を観察する。通常体と目標語尾の例は意味と文法の接続を合わせる。

few-shotや口調指示は内部観察・診断だけに使い、編集モデルの実際の評価には挿入しない。平均・射影・外積による介入を使うが、回答例への回帰フィッティングは行わない。

## 使用

以下はリポジトリルートで実行する。通常のTransformersの `from_pretrained` でもこのモデルフォルダを読み込める。独自の推論フックは不要。

```powershell
$env:UV_CACHE_DIR = "$PWD/cache/uv"
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_50/infer.py "百円玉を三枚集めるといくらになる？"
```

## 固定したモデルの再構築

環境の準備はリポジトリのREADMEを参照。1817の `config.json` と `experiment.py` を共通入口として使い、1702実験の設定には依存しない。固定した入力とソースのLF正規化SHA-256は `results/selection.json` にある。

```powershell
$env:UV_CACHE_DIR = "$PWD/cache/uv"
# 元モデルで60組×3文脈の活性を再収集（既にruns/fewshot_stats.ptがあれば省略可）
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_50/run.py collect
# 選択済みの式から差分を再計算し、固定した差分SHAと照合
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_50/rebuild_selected.py
# 保存先と最終評価ファイルが存在しない新しいcheckoutで実行
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_50/finalize.py
```

最終評価JSONはGit管理されるため、通常のcheckoutには既に存在する。再検証する場合は、別checkoutで `results/base_heldout_128.json`、`results/edited_heldout_128.json`、`results/base_hashes.json`、`results/verification.json` を別名に退避してから実行する。`models/zundamon-edited/`、`models/actual_delta.safetensors`、`runs/finalization/` も既存の場合は退避が必要。元の実験記録を上書きしないため、`finalize.py` は保存先が存在すると停止する。

差分SHAが一致しなければ再構築を停止する。異なるGPU・ライブラリ版でのビット単位の再現は未確認。保存済みモデルの再ロード一致はRTX 2080 Ti／FP16で検証済み。

## 探索と評価の再実行

探索を再現する場合は `run.py diagnostic`、`run.py collect` の後、`run.py search --family start`、`--family sustain`、`--family branch` の順に実行する。`sustain` は位置重複を含む当時の探索条件をそのまま保持しており、採用モデルには使わない。

探索は別checkoutで行い、対象の既存成果物を事前に退避する。`diagnostic` と `search` は同名ログ・差分を、`collect` は活性ファイルを上書きする。以下の `evaluate` を再実行する場合も、Gitにある `results/validation_branch_04.json` と `results/questions_branch_04.json` をそのcheckout内で退避しておく。提供済みの元ログとレビューは比較用に保持する。

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_50/run.py evaluate --candidates branch_04 --split validation
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_50/run.py evaluate --candidates branch_04 --split questions
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_50/build_report.py
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 -m pytest -q continuation_50/test_continuation.py --basetemp=continuation_50/runs/pytest
```

`evaluate` も既存の同名出力を上書きしない。集計には全回答の理由付きレビューが必要で、語尾の自動検出だけで正式な成功率を出さない。既存100問は開発用、新規50問は候補固定後の確認用と区別している。

`build_report.py` は凍結記録との整合性を検証する。探索・評価を再生成すると、同じ回答でも計測時間 `seconds` によりログのSHAが変わるため、元の `selection.json` のままでは集計を停止する。固定モデルの再構築では提供済みの開発ログとレビューを保持する。探索そのものをやり直す場合は別の実験記録として、再レビュー後に `freeze.py` で新しい選択記録を作成し、保存検証・集計まで実施する。元の凍結記録は保持する。

モデル・差分・活性・詳細列ログ・候補の10問短縮生成ログはGit管理外。コード、入力、本評価の回答、判定理由、全候補の比較集計、ハッシュをGitに保存する。
