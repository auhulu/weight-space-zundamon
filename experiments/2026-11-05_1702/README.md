# 第2回：回答中の活性・層・強度・対照条件

実験IDは`2026-11-05_1702`。日時は最初の実行記録ファイルの作成時刻に基づくJST。意味をそろえた24組の敬体／のだ体回答をteacher forcingで読み込み、層15・19の回答中の活性から編集を作る。

## 条件と結果

64列に絞ったrank-one書き込みについて強度0.5・2・6と単独層／両層を比較し、射影増幅候補と直交ランダム方向の対照も評価した。候補10種とランダム対照3種。最終比較は未編集・プロンプト・選択編集・ノルムをそろえたランダム編集の4条件。

選択されたのは`write_15_a0.5`。層15の`mlp.down_proj.weight`の57,577要素がFP16で変化した。他の319パラメータテンソルは不変で、元値への復元・保存後の再ロードのハッシュ・greedy生成の一致を確認した。

| 条件 | 目標語尾を持つ文 / 全文 | 語尾率 |
| --- | ---: | ---: |
| 未編集 | 0 / 159 | 0% |
| プロンプトによる口調指定 | 54 / 351 | 15.4% |
| 選択した重み編集 | 0 / 168 | 0% |
| ランダム編集 | 0 / 164 | 0% |

各条件48問×3seedの結果。重みは変更されたが、目標語尾の増加は確認できなかった。選択記録の`eligible`は退化チェック上の適格性で、口調制御の成功ではない。初回とは問題集合が異なるので、語尾率の差をそのまま実験間の改善と解釈しない。

## ファイル

| ファイル・ディレクトリ | 内容 |
| --- | --- |
| [ROUND2_PROTOCOL.md](ROUND2_PROTOCOL.md) | 固定条件 |
| [data/round2/](data/round2/) | 今回の入力とマニフェスト |
| [data/](data/)直下のJSON | 初回との問題重複検査・データ生成用の参照コピー |
| [round2.py](round2.py) | 活性収集・候補探索・編集・最終生成 |
| [results/round2/implementation.json](results/round2/implementation.json) | コード・設定の凍結ハッシュ |
| [results/round2/summary.json](results/round2/summary.json) | 条件別集計 |
| [results/round2/verification.json](results/round2/verification.json) | 変更対象・保存・再ロード検証 |
| [results/round2/provenance.json](results/round2/provenance.json) | 生ログや差分のハッシュ |
| `runs/round2/` | 生の生成ログ・活性・品質レビュー（Git管理外） |
| `models/round2-edited/`・`models/round2_delta.safetensors` | 編集モデルと実際の差分（Git管理外） |
| `images/weight-space-zundamon/` | 再生成される図 |

内部の`round2/`名は凍結済みコードとハッシュを保持するため維持している。マニフェスト内のパスはこのディレクトリからの相対パス。

## 再実行

[ルートREADME](../../README.md)に従って共有環境を準備し、リポジトリルートから実行する。

```powershell
$env:UV_CACHE_DIR = "$PWD/cache/uv"
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 -m pytest -q
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 round2.py search
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 round2.py final
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 round2_report.py
```

固定入力はGitに含むため`round2_data.py`の再実行は不要。生成スクリプトは設計の出所として保存している。`search`は回答ごとに再開可能。`final`は再選択した候補が凍結済みの選択記録と一致することを確認してから進む。最終回答を見て編集条件を変える場合は、新しい実験として分ける。

## 品質レビュー

`round2_report.py`が作る条件名を伏せた144回答を別途評価する。各カテゴリの先頭3問×3seed×4条件で、全576回答を網羅する意味品質評価ではない。

`runs/round2/blind_packet0.json`〜`blind_packet2.json`を別々の担当に渡し、条件対応表は渡さない。[review_rubric.md](review_rubric.md)のファイルパスを今回のパケットに読み替え、`{review_id, relevance, naturalness, note}`の配列を同ディレクトリの`blind_reviews0.json`〜`blind_reviews2.json`へ保存する。

```powershell
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py 2026-11-05_1702 round2_report.py --quality
```

集計スクリプトが意味品質を自動採点するわけではない。語尾から条件を推測できる場合もあり、人間による評価や事実検証の代替にはならない。図を再生成したら、掲載用にはルートの`images/weight-space-zundamon/`へコピーする。
