# 56%からの追加改善

**既存100問で56%から77%へ改善した。候補固定後の新規50問では、同じ質問で旧モデル64%から新モデル72%へ改善した。** 内容の正誤は採点していない。

| 集合 | 未編集 | 旧56%モデル | 今回のモデル |
| --- | ---: | ---: | ---: |
| 既存100問（開発評価） | 0/100 | 56/100 | **77/100** |
| 候補比較50問（開発評価） | 0/50 | 27/50 | **40/50** |
| 今回の新規50問（固定後） | 0/50 | 32/50 | **36/50** |

[報告書](REPORT.md)、[全回答・判定理由・カテゴリ集計](results/report.json)、[選択記録](results/selection.json)、[保存検証](results/verification.json)を参照。最新の既存100問の生回答は [questions_class_post_01.json](results/questions_class_post_01.json)、対応するレビューは [reviews/questions_class_post_01.json](reviews/questions_class_post_01.json)。

保存モデルは `models/zundamon-edited/`。0始まりの層19の `mlp.down_proj.weight` だけに、名詞・動詞・形容詞の3方向と語尾直後の1方向を合成して書き込んだ。FP16で3,654,154要素が変化し、他319テンソルは不変。単純な強度調整だけでは改善が小さかった。

既存100問の反復は6→9件、上限到達は6→10件で増えた。77成功のうち1件は自然な文を出した後に反復して上限へ達している。内容や反復を合否から外した語尾評価であり、会話品質全般の改善を意味しない。

前回の56%モデル・凍結結果は保持する。今回も固定revisionの未編集Qwen3.5-0.8Bから各候補を構築し、内部MLPの重みを直接編集する。勾配・optimizer・回帰フィッティング・出力層編集は行わない。

推論は一般的な日本語systemのみ、greedy、thinking無効、max_new_tokens=128。内容の正誤は採点せず、自然な「のだ／なのだ」で終わる回答の割合を改善する。引用内だけ、不自然な接続、独立した語尾だけは成功としない。

既存100問・validation50問・これまでのheldoutは既知の開発用と扱う。新規heldout50問は候補固定後まで生成しない。最初に層19近傍・強度を調整し、次に名詞／動詞／形容詞の活性条件の違いと語尾後の継続を調べる。

各候補はvalidationの1問おき25問で比較し、有望なものを全50問・既存100問で回答単位レビューする。生成ログの自動指標は正式な成功率とは区別する。

## 使用

リポジトリルートで次を実行する。通常のTransformers形式で保存しており、推論時の編集フックは不要。

```powershell
$env:UV_CACHE_DIR = "$PWD/cache/uv"
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_refine/infer.py "百円玉を三枚集めるといくらになる？"
```

## 固定モデルの再構築

リポジトリのPython環境・固定revisionの元checkpointを準備する。活性ファイルは前回の `continuation_50/runs/fewshot_stats.pt` を再利用する。未取得なら[前回手順](../continuation_50/README.md)に従って `continuation_50/run.py collect` で作成する。観察データは前回の60組で、今回は新たな活性収集や回答へのフィッティングをしていない。

```powershell
$env:UV_CACHE_DIR = "$PWD/cache/uv"
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_refine/delivery.py rebuild
```

再計算した差分SHAが選択記録と一致することを検証する。既存の異なる差分は上書きしない。観察ファイル・ソース・入力のハッシュが異なる場合も停止する。異なる環境でのビット単位の再現は未確認。

新しいcheckoutでモデル保存・評価・検証を再実行する場合は、旧56%モデルも前回の再構築手順で `continuation_50/models/zundamon-edited/` に用意する。これは比較対象を読み込むためであり、新モデルの編集元には使わない。

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_refine/delivery.py finalize
```

`finalize` は既存成果物を上書きしない。再実行するcheckoutでは、今回フォルダ内の `models/zundamon-edited/`、`models/actual_delta.safetensors`、`runs/finalization/`、`results/base_heldout_128.json`、`results/edited_heldout_128.json`、`results/previous_heldout_128.json`、`results/base_hashes.json`、`results/verification.json` を事前に退避する。凍結された開発ログとレビューは保持する。

## 探索・集計・検証

探索は別checkoutの新しい実験記録として扱い、提供済み成果物を保存してから行う。`research.py` は同名の生成ログがあると停止する。

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_refine/research.py search --family strength
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_refine/research.py search --family post
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_refine/research.py search --family class
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_refine/research.py search --family class_post
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_refine/research.py evaluate --candidates class_post_01 --split validation
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 continuation_refine/research.py evaluate --candidates class_post_01 --split questions
```

全回答を同じ基準でレビューしてから `delivery.py freeze --candidate class_post_01 --family class_post` で固定し、`finalize`、新規50問の3条件レビュー、`delivery.py report` の順に進める。既存の選択・レポートがあれば停止する。再生成では計測時間も変わるため、古いログのSHAを新しいログに流用しない。提供された凍結記録をそのまま再構築する手順と、新たな探索・再選択は区別する。

品詞キーの応答・不要位置への直交性・非有限値・退化キー・データ順序を検査する14テストが成功した。元重みへの復元、非累積性、有限値、重み共有、保存・再ロードの全ハッシュと8問greedy出力一致も検証済み。

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 -m pytest -q continuation_refine/test_class_edit.py --basetemp=continuation_refine/runs/pytest
```

コード・入力・本評価の回答・全回答レビュー・候補比較・ハッシュをGitへ保存する。モデル・差分・活性・候補の短縮生成ログはGit管理外。旧15%・56%モデルとその凍結結果は保持している。
