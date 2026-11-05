# Weight-space Zundamon experiment

Small local replication-inspired experiment: direct weight edits to Qwen3.5-0.8B for Japanese のだ/なのだ endings. See `experiments/2026-11-05_1536/PROTOCOL.md` before running. This is not a faithful reproduction of the source's math benchmark.

## 実験ディレクトリ

実験ごとに `experiments/YYYY-MM-DD_HHMM/` にコード・設定・入力・生成ログ・結果・編集モデルをまとめる。日時はJST。既存2回は最初の実行記録ファイルの作成時刻（分単位）を採用した。

| 実験 | ディレクトリ | 内容 |
| --- | --- | --- |
| 第1回 | [2026-11-05_1536](experiments/2026-11-05_1536/) | 初回の4候補による語尾編集 |
| 第2回 | [2026-11-05_1702](experiments/2026-11-05_1702/) | 回答中の活性、疎な編集、ランダム対照 |

`.venv/`、`cache/`、依存定義はルートで共有する。以下のコマンドはリポジトリルートで実行する。`run_experiment.py` が対象ディレクトリを作業場所にして、共有キャッシュを指定して起動する。

既存の実験コード・入力・結果・ハッシュ記録は内容を変更せず移動した。第2回内部の `data/round2/`・`runs/round2/`・`results/round2/` は凍結済みコードとハッシュの互換性のため維持している。第2回の `data/*.json` は重複検査とデータ生成に使う第1回入力の参照コピー。各マニフェストのパスは所属する実験ディレクトリからの相対パス。

新しい試行は新しい日時のディレクトリを作り、必要なコード・設定・入力のみをコピーする。既存の生成ログ・結果・モデル・凍結マニフェストは流用しない。大きなログとモデルは引き続きGit管理外。記事と掲載用画像はルートの `articles/`・`images/` に置く。第2回の図を再生成した場合は実験内の画像を掲載用ディレクトリへコピーする。

## Observed result

Final 48 prompts × 3 seeds: target endings base **0/171 sentences**, style prompt **39/547 (7.1%)**, weight edit **0/168**. No style improvement was observed for the selected edit. Four candidates all scored zero on validation; the smallest eligible change was selected by the fixed tie-break rule. `selection_passed` in the JSON means eligibility under the degeneration checks, **not experimental success**.

The Japanese unpublished Zenn draft is `articles/weight-space-zundamon.md`; figures are under `images/weight-space-zundamon/`. Numeric tables, blind agent ratings, tensor verification and provenance hashes are in `experiments/2026-11-05_1536/results/`. 768 generations were completed, including training/validation/side runs. Review judgments are descriptive and do not establish factual correctness. The chosen generation settings differ from the model card recommendations; results are limited to this configuration.

## Environment (Windows / NVIDIA)

```powershell
$env:UV_CACHE_DIR = "$PWD/cache/uv"
uv venv --python 3.12
uv pip sync --python .venv/Scripts/python.exe requirements.lock --extra-index-url https://download.pytorch.org/whl/cu126
$env:HF_HOME = "$PWD/cache/huggingface"
uv run --no-project --python .venv/Scripts/python.exe python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17')"
```

The checked-in `requirements.lock` records exact installed versions. Commands use `uv run --no-project` deliberately to use this frozen environment without resolving pyproject's broad library constraints. Model downloads require network; experiment execution is offline. About 3 GB model data plus CUDA PyTorch and outputs are needed. The checkpoint includes vision weights; the experiment loads only its causal language model.

## Run in order

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 -m pytest -q
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 experiment.py smoke
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 experiment.py prepare
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 experiment.py candidates
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 select_candidate.py
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 experiment.py final
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 analyze.py
```

Each generation is saved immediately. Rerunning stages resumes matching completed rows; do not change configuration or dataset in an existing `experiments/2026-11-05_1536/runs/` directory. Create a new timestamped directory under `experiments/` for a different experiment. Blind review packets are exported by `analyze.py`; reviews are independent agent judgments, not ground-truth annotations. Raw generations and checkpoints are intentionally outside Git. Small summaries and article figures are in `experiments/2026-11-05_1536/results/`.

## Interpretation

Activations from paired prompted/unprompted training questions inform the edit. No gradient descent is used, but data and validation feedback **are** used. The rank-one edit applies on every token and may change unrelated content. Lexical style success alone is insufficient. Compare instruction following, relevance, naturalness and examples too.

Source: https://rikiyat.github.io/blog/weight-space-programming/

## Detailed Qwen weight-editing guide

[Qwen3.5 の重みを直接編集する調査手順と復元手順](docs/qwen-weight-editing-guide.md) documents the upstream experiment logs: activation probes, causal controls, column selection, exact weight formulas, BF16 reconstruction, hash checks and differences from this style experiment. The upstream p50c edit was reconstructed locally and both edited tensor hashes matched the published manifest; this did not rerun the math benchmarks. [Verification record](docs/qwen-upstream-rebuild-verification.json).

## 第2実験：回答中の活性・層・強度・対照条件

元実験の手順を参考に、意味をそろえた24組の敬体／のだ体回答をteacher forcingで読み込み、層15・19の回答中の活性を測る。疎なrank-one書き込みの強度0.5/2/6と単層／両層を比較し、射影増幅候補と直交ランダム方向の対照も評価する。候補10種＋ランダム対照3種。最終比較は未編集・プロンプト・選択編集・ノルムをそろえたランダム編集の4条件。

[固定プロトコル](experiments/2026-11-05_1702/ROUND2_PROTOCOL.md)と`experiments/2026-11-05_1702/data/round2/`は旧実験と分離している。初期実験の最大6候補という上限を拡張した、別の探索である。

```powershell
$env:UV_CACHE_DIR='C:\Users\yo\work\playground\.uv-cache'
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 -m pytest -q
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 round2.py search
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 round2.py final
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 round2_report.py
```

固定データはGitに含むため`round2_data.py`の再実行は不要。データ生成スクリプトは設計の出所を残すために保存している。`search`は回答ごとに再開可能で、`final`は候補を再選択しても同じ選択記録になることを検証してから開始する。最終回答を見た再編集は行わない。再実験のコード・設定ハッシュは`experiments/2026-11-05_1702/results/round2/implementation.json`で固定される。新しい条件を試すなら新しい実験ディレクトリを作る。

生の回答と活性は`experiments/2026-11-05_1702/runs/round2/`、編集モデルは`experiments/2026-11-05_1702/models/round2-edited/`、実際のFP16丸め後の差分は`experiments/2026-11-05_1702/models/round2_delta.safetensors`に保存する。これらはGit管理外。小さな集計・検証記録は`experiments/2026-11-05_1702/results/round2/`、図は`images/weight-space-zundamon/`。品質評価は`round2_report.py`が作る条件名なしの固定144回答サンプルを、既存の`review_rubric.md`で評価する。採点を再現するには別途エージェント評価が必要で、集計スクリプトが意味品質を自動採点するわけではない。

品質評価サンプルは各カテゴリの先頭3問×3seed×4条件。`experiments/2026-11-05_1702/runs/round2/blind_packet0.json`〜`blind_packet2.json`を別々の担当に渡し、条件対応表は渡さない。採点基準のファイルパスだけを今回のパケットに読み替え、`{review_id, relevance, naturalness, note}`の配列を`blind_reviews0.json`〜`blind_reviews2.json`に保存する。集計は次で行う。

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 round2_report.py --quality
```

これは固定サンプルへの一回のエージェント判断で、全576回答の意味品質を網羅する評価ではない。語尾から条件を推測できる場合もあり、人間の評価や事実検証を代替しない。
