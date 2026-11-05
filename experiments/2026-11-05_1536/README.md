# 第1回：プロンプト末尾の活性差による語尾編集

実験IDは`2026-11-05_1536`。日時は最初の実行記録ファイルの作成時刻に基づくJST。Qwen3.5-0.8Bへ口調指示を与えた場合と与えない場合の内部活性の差を使い、通常の質問だけでも「のだ／なのだ」を出すことを目指した。

## 条件と結果

固定した4候補を検証し、選択した編集を未使用の48問×3seedで評価した。最終的に層12の`mlp.down_proj.weight`を編集し、FP16の2,484,659要素が変化した。保存・再ロード後のテンソルとgreedy生成の一致も確認した。

| 条件 | 目標語尾を持つ文 / 全文 | 語尾率 |
| --- | ---: | ---: |
| 未編集 | 0 / 171 | 0% |
| プロンプトによる口調指定 | 39 / 547 | 7.1% |
| 重み編集 | 0 / 168 | 0% |

口調改善は確認できなかった。検証時も4候補すべてが目標語尾0件だったため、事前に決めた同点処理で適格候補のうち変更量の小さいものを選んだ。`selection_passed`は退化チェック上の適格性であり、実験成功を意味しない。

学習用・検証用・副作用評価を含めて768回答を生成した。thinking無効・最大256トークン等の設定はモデルカードの推奨設定とは異なり、この条件での結果として扱う。品質レビューは固定サンプルへの記述的な評価で、事実の正しさを保証するものではない。

## ファイル

| ファイル・ディレクトリ | 内容 |
| --- | --- |
| [PROTOCOL.md](PROTOCOL.md) | 固定した実験条件 |
| [config.json](config.json)・[requirements.lock](requirements.lock) | モデルrevision・生成条件・依存 |
| [data/](data/)・[input_manifest.json](input_manifest.json) | 入力と凍結ハッシュ |
| [editor_notes.md](editor_notes.md)・[edit_design.py](edit_design.py) | 候補設計と編集式 |
| [results/summary.json](results/summary.json) | 語尾率と副作用の集計 |
| [results/verification.json](results/verification.json) | 重みの変更・再ロード検証 |
| [results/run_manifest.json](results/run_manifest.json) | 生成数・実行時間・出所ハッシュ |
| `runs/` | 生の生成ログ・活性・品質レビュー（Git管理外） |
| `models/edited/`・`models/edit_delta.safetensors` | 編集モデルと実際の差分（Git管理外） |

マニフェスト内のパスはこのディレクトリからの相対パス。ディレクトリ整理では凍結済みコード・入力・結果の内容は変更していない。

## 再実行

[ルートREADME](../../README.md)に従って共有環境を準備し、リポジトリルートから順番に実行する。

```powershell
$env:UV_CACHE_DIR = "$PWD/cache/uv"
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 -m pytest -q
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 experiment.py smoke
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 experiment.py prepare
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 experiment.py candidates
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 select_candidate.py
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 experiment.py final
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 analyze.py
```

生成は回答ごとに保存され、条件が一致する既存回答から再開する。条件を変える場合は別の実験ディレクトリを作る。`analyze.py`が作る匿名化パケットを[review_rubric.md](review_rubric.md)に沿って別途評価し、`runs/blind/reviews_0.json`〜`reviews_2.json`へ保存してから、次で品質を含む報告を作る。

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1536 write_report.py
```

集計スクリプト自体が意味品質を自動採点するわけではない。記事の掲載画像はルートの`images/weight-space-zundamon/`にあるため、図の再生成後は必要に応じて掲載用コピーを更新する。
