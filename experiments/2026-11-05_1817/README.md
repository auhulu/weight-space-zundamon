# 1817：指示なしの「のだ」語尾を目指す重み編集

未編集の `Qwen/Qwen3.5-0.8B` から開始し、内部MLPの重みだけを変更する実験。元実験の観察・仮説・直接介入という方法を、日本語の語尾制御へ応用する。数学ベンチマークや上流の数値結果の再現ではない。

[実験報告書](REPORT.md)に目的・方法・評価結果・出力例・限界を、[全候補の比較表](REPORT_CANDIDATES.md)に50候補の条件と結果をまとめている。

**実施済み。既存100問の語尾成功は0/100から15/100へ増えたが、目標80%には未達。** 内容の正誤は合否に含めていない。別の汎化50問では0/50から13/50（26%）だった。

| 評価集合 | 未編集 | 編集後 |
| --- | ---: | ---: |
| 既存100問 | 0/100（0%） | 15/100（15%） |
| 新規汎化50問 | 0/50（0%） | 13/50（26%） |

保存先は `models/zundamon-edited/`（Git管理外、モデル重み約1.50GB）。層23の `mlp.down_proj.weight` の3,638,144要素が変化し、他のパラメータは不変。元重みへの復元、差分の再計算、保存・再ロードの全パラメータハッシュ、再ロード前後8問のgreedy出力一致を確認した。48テストが成功している。

編集後100問には反復76件、128トークン上限到達83件がある。内容・反復を選択基準から外したため、通常の会話品質は改善を保証しない。自動語尾検出は58%だったが、語中切断などを回答単位で確認した正式な語尾成功率は15%。評価はCodexエージェントによるもので、人間の独立評価ではない。[100問の結果](results/report.json)・[汎化結果](results/generalization_report.json)・[重み検証](results/verification.json)を参照。

## 固定条件

- 元revision：`2fc06364715b967f1860aea9cf38778875588b17`。過去の実験の編集済みモデルは使用しない。
- 推論system：`日本語で簡潔に、原則1文で答えてください。`
- `max_new_tokens=128`、greedy、thinking無効、FP16、RTX 2080 Ti。
- backward・optimizer・教師あり回帰フィッティングを行わない。embedding・出力層・final normを変更せず、重み共有も維持する。
- 候補は常に元重みから作る。生成時に口調指示や回答例を挿入せず、出力も書き換えない。

成功基準は既存100問の語尾成功率80%以上。ユーザーの追加指示により、内容の正しさは合否条件に含めない。詳細は [PROTOCOL.md](PROTOCOL.md) を参照。

## データと評価

| 入力 | 件数 | 用途 |
| --- | ---: | --- |
| `data/train_pairs.json` | 100組 | 通常体／のだ体の意味をそろえた回答で内部活性を観察 |
| `data/validation.json` | 50問 | 編集候補の比較 |
| `data/questions.json` | 100問 | 既存の評価質問、10カテゴリ各10問 |
| `data/generalization.json` | 50問 | 別の質問で語尾の汎化を確認 |

既存100問の原文は編集データに使わない。各集合間の質問重複を検査する。候補選択を固定してから既存100問と汎化50問の編集後回答を生成する。

`scoring.py` の語尾指標は機械的な候補選別用。最終語尾率は `reviews/` の回答単位レビューで算出する。本文に自然な接続の文・名詞文が1つでもあれば成功。「のだ」だけの出力・独立行、引用中だけの一致、不自然な接続は不合格。反復・上限到達・内容の誤りは参考情報とし、自然な目標語尾を持つ回答をそれだけで不合格にはしない。

未編集モデルの100問を128トークン上限で再生成した結果、語尾成功は **0/100**。最初の回答末尾の活性差に基づくMLP編集は、10候補の短縮評価で語尾率0%だったため探索を中断した。

## 編集と検証

1. `experiment.py collect`：意味をそろえた通常体／のだ体の回答から全層の残差・MLP活性を収集。
2. 初期探索：中盤以降の層への書き込み10候補を比較。増幅候補は計画したが、このラウンドでは実行前に打ち切った。
3. `contrast.py collect`：同一の回答prefixを口調指示あり／なしで読ませ、開始・中間・語尾分岐点の活性差を測る。ここでの口調指示は診断専用。
4. 第2ラウンド：語尾分岐点の方向への書き込みと、逆向きに書き込む列の成分抑制を比較。
5. 第3ラウンド：語尾分岐点の平均MLP活性を、回答開始・中間の平均活性からGram–Schmidt法で直交化する。そのkeyでrank-one編集を作り、開始・中間には平均的に書き込まず、語尾分岐点で観測方向を書き込む仮説を試す。単独層と複数層、編集強度を比較する。回帰や損失フィッティングは行わない。
6. `finalize.py`：選択した編集を適用し、変更対象以外の不変性、非累積性、元重みへの復元、保存・再ロード後のハッシュとgreedy出力一致を検証。

モデルと差分は `models/`、大きな活性と候補生成ログは `runs/` に保存し、Git管理外とする。設定、入力、選択記録、集計、レビュー、検証記録をGitに残す。

## 再実行

依存環境と元checkpointはルートREADMEの手順で準備する。以下はリポジトリルートで実行する。

```powershell
$env:UV_CACHE_DIR = "$PWD/cache/uv"
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 -m pytest -q --basetemp=runs/pytest-verification
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 experiment.py baseline
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 experiment.py collect
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 contrast.py collect
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 contrast.py search --round 2
```

第3ラウンドは `contrast.py search --round 3` を使う。

第3ラウンド内で `--extension` による強度比較を行った後、語尾後の反復に対して `contrast.py collect_post` と `contrast.py search --round 3 --terminal` を実施した。後者は、開始・中間に加えて目標語尾後3位置の平均活性もkeyから除く。

探索途中で独立した「のだ」行の自動誤検出を修正した。探索JSONの自動指標は当時の記録であり、最新版の検出器で探索を再実行すると候補順位が変わる場合がある。最終選択は自然な接続のレビューで行い、固定モデルの再現には下記の差分再構築を使う。固定後に既存100問・汎化50問の編集後出力を見て再調整していない。

テキストのSHA-256は改行をLFに正規化して計算する。最終評価後、Gitの改行変換に対応するためテキスト出力とハッシュ記録の処理のみを修正した。モデル計算と候補選択は変更していない。バイナリの重み・差分は生のバイト列で照合する。

固定した編集は `results/selection.json` の `round3_terminal_03`。層23の `mlp.down_proj.weight` に、強度8のrank-one形式の差分を加える。検証50問のレビューは18/50（36%）。内容・反復を合否に使わないため、反復の少ない別候補（16/50）よりもこちらを選んだ。これは安定した会話品質を意味しない。

探索を繰り返さず固定モデルを再構築するには、上記の環境準備と `experiment.py collect` の後、次を実行する。`rebuild_selected.py` は再計算した差分のSHA-256が固定値と一致することを確認する。既存の評価結果から候補を再選択しない。

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 contrast.py collect
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 contrast.py collect_post
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 rebuild_selected.py
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 finalize.py --edit models/round3_terminal_03_delta.safetensors
```

保存モデルへの任意の質問は次で実行できる。systemと128トークン上限は評価と同じ。

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 infer.py "雨の日に家で楽しめることを教えて。"
```

最終評価の集計は次の形式で行う。

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1817 report.py --condition base=results/base_questions_128.json --condition edited=results/edited_questions_128.json --review base=reviews/base_questions.json --review edited=reviews/edited_questions.json --output results/report.json
```

レビューが欠けている場合、`report.py` は機械的な語尾指標だけを表示し、正式な成功判定を行わない。レビューはCodexエージェントによるもので、人間の独立評価ではない。
