# 元リポジトリはどうQwenの重みを編集したか

以下は上流のExperiment 8、主に`p50c`の説明。参照コミットは`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`に固定する。上流の成果と、このプロジェクトの口調実験の成果は別である。

詳細なコード例、調査の経緯、復元コマンド、ハッシュ一覧は[Qwen重み編集の詳細ガイド](qwen-weight-editing-guide.md)にまとめている。

### 1. 編集用LLMが観察と介入を繰り返す

編集担当の`gpt-6-astra`がCPU上のPythonカーネルで重みを読み書きし、別の推論サービスへ生成や内部活性の取得を要求する。`load_original(name)`で元重みを読み、`save_edit(name, tensor)`で同じ名前・shapeのテンソルを保存する。これらは実験専用APIで、PyTorchの標準関数ではない。

逆伝播・optimizer・損失に対する反復的なフィッティングは禁止。一方、練習問題を使って仮説を立て、編集箇所や強度を比較することは許されている。「勾配学習なし」は「データも評価結果も使わない」という意味ではない。forward passの予算はバッチ単位の問い合わせ回数で、50 passesは50トークンや50回答を意味しない。[上流の実験条件][setup]

### 2. まず何が失敗しているかを調べる

`p50c`の初期診断では96回答中85回答が32,768トークンの上限に達し、正答は11件だった。そこで数学知識を直接書き込むのではなく、同じ確認を繰り返して終わらなくなる挙動に注目した。

同じ問題について、計算が前に進む正常な継続と、確認を繰り返す継続を用意する。teacher forcingで固定した文章を読ませ、各層の残差ストリーム・MLP中間活性・トークンの対数確率を比較する。これは活性の測定であり、学習処理ではない。[上流の最終報告][report]

### 3. 反復に関連する方向とニューロンを探す

残差ストリームは1024次元、MLP中間活性は3584次元。ペアごとの継続末尾の平均を使い、反復側と正常側の差から方向ベクトルを作る。

```text
d = mean_pairs(mean_tail(h_repetitive) - mean_tail(h_productive))
u = d / ||d||
```

`W`をMLPの`down_proj.weight`とすると、`u @ W`は各中間ニューロンが方向`u`へ出力を書き込む係数になる。反復側で増える活性と、その出力先の両方を使って候補を選ぶ。

```text
writer_score[j] = mean_pairs(g_repetitive[j] - g_productive[j]) * (u @ W)[j]
```

最終的には層15・19でそれぞれ64ニューロンを選んだ。番号は0始まりで、`layers.15`は16番目のブロック。ニューロンは重みの「列」に対応するため、編集するのは`W[:, selected_ids]`。採用したIDは上流の[固定時マニフェスト][frozen]で確認できる。方向や列選択を作る具体的な経緯は[詳細ガイド](qwen-weight-editing-guide.md)の第5〜7節を参照。

### 4. 対象行列へ差分を直接加える

最終的に編集したのは、次の2テンソルだけである。[公開差分マニフェスト][manifest]

```text
model.language_model.layers.15.mlp.down_proj.weight  [1024, 3584]
model.language_model.layers.19.mlp.down_proj.weight  [1024, 3584]
```

層ごとの元行列を`W0`、単位方向を`u`、選択した64列が1で他が0のマスクを`m`とすると、式は次のとおり。

```text
W_new = W0 - alpha * outer(u, m * (u @ W0))
alpha = 6
```

`u`は1024要素、`m * (u @ W0)`は3584要素。その外積は元行列と同じ1024×3584になり、選択していない列の差分はゼロになる。各候補は常に元の`W0`から作り、前候補へ編集を累積させない。[上流の最終報告][report]

この式を実数演算で解釈すると、選択列の`u`方向成分は`(1-alpha)`倍になる。`alpha=1`なら除去、`alpha=6`なら元成分の−5倍である。したがって最終編集を「方向成分をゼロにしただけ」と解釈するのは誤り。層15と19は別々の方向と列集合を使う。

### 5. 対照実験を行ってから固定する

上流では単一ニューロンの無効化、単独層と両層、編集強度、全列と64列、方向に直交する同ノルムのランダム編集を比較している。また、正常な継続や早すぎるthinking終了への影響も調べている。相関があるニューロンを見つけるだけで終わらず、実際に介入して挙動が変わるか確かめる手順である。

最終候補を固定した後、選択に使っていない練習問題1152問で正答が122件から696件になったと報告されている。これは上流の練習用集合内での確認結果であり、独立ベンチマークとは区別する。[上流の最終報告][report]

### 6. 編集済み重みを保存し、その値で推論する

編集担当は変更したテンソル全体をFP32の`.npy`で提出し、評価側がBF16へ変換してcheckpointの元テンソルを置き換える。そのcheckpointをvLLMで読み込んで評価する。プロンプトを変えるだけの操作ではなく、推論が使用する重みそのものが変わる。[上流の保存・評価手順][setup]

公開差分の`p50c/edit/edit.npz`は、変更位置と**編集後のBF16のbit値**を保存している。保存値は加算量ではない。公式の`apply_edit.py`は固定revisionの元重みにこれらを代入し、編集前後のテンソルのSHA-256を検証する。[復元実装][apply]

計算上の対象は`2 × 64 × 1024 = 131,072`要素。BF16の丸め後に元値と異なる要素は130,188個で、両者は異なる数え方である。式からの再計算は丸めの順序などでbitが変わる可能性があるため、**公開された最終モデルと一致させるには公式差分を使う**。[公開差分マニフェスト][manifest]

### 7. 復元と探索の再現を区別する

| 目的 | 必要なもの |
| --- | --- |
| 最終重みの厳密な復元 | 固定revisionの元BF16 checkpoint・`edit.npz`・`manifest.json`・公式`apply_edit.py` |
| 編集方法の理解 | プロンプト・コード履歴・forward要求と結果・最終報告 |
| 探索の再実行 | 活性収集と生成のサービス、編集カーネル、同じ入力条件の再構築 |
| ベンチマークの再測定 | 評価データ・採点器・テンプレート・生成条件をそろえた推論環境 |

公開リポジトリには探索に使った大きな中間活性や実験サービス本体が含まれないため、最終差分を復元することと探索全体を再実行することは同じではない。[公開物の範囲][setup]

具体的なPowerShellコマンド、shardの確認、BF16とFP32の区別、保存後のハッシュ照合は[詳細ガイド](qwen-weight-editing-guide.md)の第11〜12節を参照する。

## 参考資料

- [Qwen重み編集の詳細ガイド](qwen-weight-editing-guide.md)：観察、式、対照条件、復元手順、評価条件を詳述。
- [元リポジトリ](https://github.com/RikiyaT/lm_understand-logs)：公開ログ・コード・差分。
- [元記事](https://rikiyat.github.io/blog/weight-space-programming/)：研究全体の説明。

上流資料はRikiyaTらによるもの。ログ等はCC BY 4.0、PythonコードはMITで公開されている。詳しい利用条件は[上流のライセンス][license]を参照。本READMEは日本語での要約と数式の説明を加えたもの。

[setup]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md
[report]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md
[frozen]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/frozen_weights_manifest.json
[manifest]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json
[apply]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/apply_edit.py
[license]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE
