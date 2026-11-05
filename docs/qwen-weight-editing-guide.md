# Qwen3.5 の重みを直接編集する調査手順と復元手順

対象は **Qwen3.5-0.8Bの言語モデル部分**。RikiyaTらが公開した[実験ログ][repo]のExperiment 8をもとに、失敗の診断、内部活性の調査、重み編集、対照実験、保存と検証までを整理する。

主に扱うのは50回のforward passを使った`p50c`である。最終的には、層15・19のMLPの各64列について反復に関連する方向の成分を減らした。数学の問題と正常な推論を保ちながら、長い反復から抜け出せるかを調べている。口調変更用の万能な編集式として読むのではなく、**特定の失敗を調べ、因果的な介入を検証した事例**として使う。[実験条件][setup]

参照コミットは`4d09ba6d6a7e891dd8063974dfe81b3d78e3494f`。以下の参照リンクはこのコミットに固定している。確認日は2026年11月5日。

## 1 再現する対象を選ぶ

|目的|必要な作業|公開物だけでできる範囲|
|---|---|---|
|最終的な編集済みの値を復元する|固定revisionの元重み＋公式差分を適用し、ハッシュ照合|公開スクリプトで厳密に復元できる|
|どのように編集を発見したか理解する|プロンプト、実行コード、forward要求、結果、最終監査を追う|ログから手順と判断を確認できる|
|探索そのものをもう一度実行する|編集用カーネルとforwardサービスを実装し、活性を再収集する|中間活性や実験専用の全サービス実装は未公開なので、追加実装が必要|
|数学ベンチマークの点数まで再測定する|同じモデル、テンプレート、生成条件、データ、採点器で評価する|公開コードには内部パスや未同梱の採点器があるため、環境構築が必要|

この作業では「最終的な編集済みの値を復元する」をローカルで検証した。元実験の数学ベンチマークは再実行していない。以前のずんだもん風口調実験のモデルや評価結果は変更していない。

## 2 元実験の実行構成

```text
編集担当LLM
  ├─ Pythonコード → CPUカーネル → 重みの読み取り・変更・保存
  └─ forward要求 → 推論サービス → 生成結果・log probability・活性
                  ↓
             結果を次の入力に戻す
                  ↓
            最終編集を固定する
                  ↓
          非公開の最終評価で比較する
```

編集担当は`gpt-6-astra`、reasoning effortは`max`。組み込みツールを停止し、実験側が提供したPythonカーネルとforward passの経路を使っている。`p50c`は158ターン、Python実行80回、成功したforward pass 50回で、報告上の壁時計時間は20.3時間だった。50回は50トークンや50回答ではなく、バッチ単位の問い合わせ回数である。[実験条件][setup]

編集用カーネルではGPU・ネットワークを使わず、以下の関数を提供する。これらは実験側のAPIであり、TransformersやPyTorchの標準関数ではない。[編集担当への指示][prompt]

変数はコード実行をまたいで維持されるが、カーネル再起動で失われる。指示上の1実行の制限は900秒・96GB、表示出力は10,000文字まで、Python実行予算は最大300回だった。解析用のメモと候補のバックアップは、変数とは別にファイルへ保存している。

|API|意味|
|---|---|
|`NAMES`|編集対象のテンソル名一覧|
|`load_original(name)`|元の重みをFP32のtorch tensorとして取得|
|`load(name)`|編集があれば編集値、なければ元の値を取得|
|`save_edit(name, tensor)`|同名・同shapeの編集値を保存|
|`drop_edit(name)`|指定した編集を取り消す|
|`edited()`|現在の編集対象を列挙|

禁止されていたのは、逆伝播・autograd・optimizer・損失への反復的なフィッティング、カーネル内での独自のモデル実行、ネットワークアクセス。さらに、最終norm、embedding／出力層、テンソル全体への一様なスカラー乗算も編集対象外だった。一方、重みやforward結果の解析、練習問題の利用、仮説に基づく編集強度の比較は許されていた。

**勾配学習をしないことと、実験結果を見ながら探索しないことは別**である。前回の口調実験の「固定4候補・強度探索なし」は、この元実験より制限が強い。

## 3 テンソルの構造と番号を確認する

言語モデルは24層、残差ストリーム幅1024、MLP幅3584。最終編集の対象は以下の2つである。[公開差分のmanifest][manifest]

```text
model.language_model.layers.15.mlp.down_proj.weight  [1024, 3584]
model.language_model.layers.19.mlp.down_proj.weight  [1024, 3584]
```

MLPを列ベクトルの表記で書くと、概念的には次の計算になる。

```text
g = SiLU(W_gate x) * (W_up x)   # g: [3584]
y = W_down g                  # y: [1024]
```

`down_proj.weight`の**列**がMLPの中間ニューロンに対応する。ニューロン`j`の出力先を変える操作は`W[:, j]`の編集であり、`W[j, :]`ではない。`W[:, selected_ids]`は1024×64のブロックになる。

|番号・名前|意味|混同すると起こる問題|
|---|---|---|
|`layers.15`|0始まりの16番目のブロック|1始まりで扱うと別の層を編集する|
|`hidden_16`|ブロック15の後の残差ストリーム|hiddenの番号と重みの層番号は1ずれる|
|`hidden_0`|embedding出力|ブロック0の出力ではない|
|`mlp_act_15`|ブロック15のdown projectionに入る中間活性|残差ストリーム1024次元と混ぜない|
|公開ファイルの`model.language_model.*`|上流のcheckpoint内のキー|Pythonモデルの属性名と必ずしも同じではない|
|今回のローカル`Qwen3_5ForCausalLM`の`model.layers.*`|現在の実装でのキー|文字列の見た目だけで公開差分を適用しない|

公開差分は公開checkpointのキーに適用する。ローカルのFP16モデルから同名らしいテンソルを探して代用すると、dtypeや元値が異なり、ハッシュ検証を通らない。

## 4 最初に失敗の内訳を調べる

`p50c`は最初にGSM8K trainから32問、MATH trainから64問を取り、元モデルで96回答を生成した。結果は正答11件、32,768トークン上限到達85件だった。この診断では「最後まで答えたが計算を間違えた」より、反復して終わらない失敗が支配的だった。[要求と結果の一覧][passes]

診断で保存する情報は、正誤だけでは足りない。

- 正答数、終了した誤答、終了したが回答を抽出できないもの。
- トークン上限に達した割合、生成長。
- 生成文末の反復、空白のみの繰り返し。
- thinkingが閉じたか、最終回答が出たか。
- 同一問題の正常・異常な継続を比較するためのテキスト。

「正答時に強くなるニューロン」をそのまま選ぶと、問題の難易度や回答長の差を拾う可能性がある。そのため元実験は、生成の観察からさらに、同じ問題について正常に進む継続と同じ確認を反復する継続を用意して調べている。[コード履歴の行8〜10][calls]

## 5 Teacher forcingで同じ文脈を比較する

Teacher forcingは、与えた文章をモデルに読ませ、各トークンの確率や内部活性を得る操作である。ここでは重みの学習はしない。

元実験は4問題について2種類の継続を作り、計8テキストを入力した。正常側は計算や検算を前に進め、反復側は同じ疑問や式の確認を繰り返す。修正後の入力は、評価と同じテンプレートで始まるraw textを`texts`へ渡している。

初期の`chats`形式では、assistantの推論部分が再整形され、空のthinkingブロックが追加される問題があった。公開ログではこれを見つけて修正している。既存回答を再採点する際は、**モデルに実際に渡したtoken IDsと、評価対象のトークン区間が一致していること**を確認する。[ツール仕様の訂正][corrections]

収集する配列の例は次のとおり。

|情報|概念上のshape|用途|
|---|---|---|
|残差ストリーム|`[text, token, 1024]`|反復に関連する方向の候補を作る|
|MLP中間活性|`[text, token, 3584]`|その方向を書き込むニューロンを探す|
|token log probability|モデルAPIの位置対応による|正常な継続を保ち、反復側だけが弱まるか測る|
|input IDsと実長|paddingを除いた有効区間|比較対象の位置ずれを検出する|

右paddingは平均に含めない。ログでは初期の方向推定に各継続の末尾36トークン、後のwriter選別に末尾24トークンを使っている。どちらも「プロンプト最後の1トークン」だけを比較する前回の口調実験とは異なる。[コード履歴の行10・62][calls]

## 6 反復方向と書き込み元を求める

層`l`の後の残差について、各ペアの末尾平均を取り、反復側から正常側を引く。ペア間の平均を正規化したベクトルが、反復方向の候補`u_l`となる。

```text
d_l = mean_pairs(mean_tail(h_repetitive) - mean_tail(h_productive))
u_l = d_l / ||d_l||
```

方向に沿ってMLPがどれだけ出力を書くかは、`u_l @ W_l`で調べられる。ニューロンの活性差だけでなく、**そのニューロンの出力先が反復方向に向いているか**を掛け合わせる。

```text
writer_j = (u_l @ W_l)[j]
score_j  = mean_pairs(g_repetitive[j] - g_productive[j]) * writer_j
```

層15・19は、反復方向への残差の増加とMLPの寄与が目立つ候補になった。まず層19のニューロン11の出力列をゼロにし、その後、層19全体の方向成分を減らす編集、層15と19を同時に編集する方法へ進んだ。最初の32問では改善幅に対して標準誤差も大きく、その時点で確定的な成功とは扱っていない。[コード履歴の行10・13・26][calls]

なお、保存された`repetition_directions.pt`では、いったん作ったrank-one差分の最大ノルム列から方向を取り出し直している。符号は反転し得るが、`u(u^T W)`は`u`と`-u`で同じ。ニューロン順位付けでは反復側に向くよう符号をそろえている。[コード履歴の行38・62][calls]

## 7 重みを編集する式

### 全列に対する方向成分の補正

基本となる候補は、元の重み`W0`から次の補正を引く。

```text
W_new = W0 - alpha * outer(u, u @ W0)
```

`u`は単位ベクトル。`alpha=1`なら、各列が持つ`u`方向の成分を除く。任意のMLP入力`g`に対して、元出力`W0 g`の`u`方向成分へ作用する。

これは前回の`outer(d, mean_input) / ||mean_input||²`という入力方向に基づく追加とは別の操作である。元実験の式は、**元の重みが特定の出力方向へ書き込む量**を使って補正する。

### 64列に絞った最終編集

選択ニューロンのmaskを`m`とすると、最終式は以下になる。

```text
W_new = W0 - 6 * outer(u, m * (u @ W0))
```

層15と19で、それぞれ別の`u`と64個のニューロン集合を使う。64列は元モデルの4ペアの活性から算出したwriter score上位で選び、6ペアの別文脈では順位の移り方や寄与を確認している。全64個のIDは[固定した重みのmanifest][frozen]に記録されている。[コード履歴の行62・91・98][calls]

`alpha=6`は直交射影そのものではない。選択した列については、実数演算なら`u^T W_new[:,j] = (1-6) u^T W0[:,j]`となるため、元の方向成分を反転して強く補正する。「成分をゼロにするだけ」と説明すると不正確になる。

以下は式を理解するための独立した実装例。`direction`と`neuron_ids`は測定から用意する必要があり、この関数だけで元実験の編集値が得られるわけではない。厳密な最終値の復元には後述の公式差分を使う。

```python
import torch

@torch.no_grad()
def edit_writer_columns(base_weight, direction, neuron_ids, alpha=6.0):
    w0 = base_weight.detach().to(device="cpu", dtype=torch.float32).clone()
    u = direction.detach().to(device="cpu", dtype=torch.float32).flatten()
    ids = torch.as_tensor(neuron_ids, dtype=torch.long, device="cpu")
    assert w0.shape == (1024, 3584) and u.shape == (1024,)
    assert ids.numel() == 64 and ids.unique().numel() == 64
    assert bool(((0 <= ids) & (ids < 3584)).all())
    assert torch.isfinite(w0).all() and torch.isfinite(u).all()
    assert u.norm() > 0
    u = u / u.norm()
    projected_columns = u @ w0
    masked = torch.zeros_like(projected_columns)
    masked[ids] = projected_columns[ids]
    candidate = w0 - alpha * torch.outer(u, masked)
    untouched = torch.ones(w0.shape[1], dtype=torch.bool)
    untouched[ids] = False
    assert torch.equal(candidate[:, untouched], w0[:, untouched])
    assert torch.isfinite(candidate).all()
    return candidate
```

候補は常に`W0`から作る。前候補の編集値にさらに同じ式を加えると、強度比較の意味が変わる。

### 10回と25回の実験で採用された操作

`p10c`・`p25c`・`p50c`は、それぞれ別の探索会話である。50回の会話の途中状態を10回・25回で切り出したものではない。

- **p10c**：層8〜22のattentionとMLPの出力行列、合計30個へ`W_new = W0 - 1.5 * outer(u, u @ W0)`型の編集を適用した。各行列の具体的な名前や方向はこのrunのログで確認する。p50cの2方向を流用する手順ではない。[10回の最終回答][p10]
- **p25c**：`model.language_model.layers.15.self_attn.o_proj.weight`の`[:, 1792:2048]`をゼロにした。shapeは1024×2048で、0始まりのquery head 7の256列に対応する。全headやMLPを消す操作ではない。[25回の最終回答][p25]

これらも`runs/<run>/edit`の差分で復元できるが、本書で実行してハッシュ確認した対象はp50cの2テンソルである。

## 8 相関から因果的な証拠へ進む

元実験では次を比較している。後から選んだ一番よい数字だけでなく、途中の弱い結果や副作用も読むことが重要になる。[実験の経緯][history]

|比較|確かめたいこと|ログの例|
|---|---|---|
|1ニューロンの列をゼロにする|候補ニューロンに因果的な寄与があるか|層19・ニューロン11から開始|
|1層と2層|効果が複数箇所に分散しているか|層15のみ、19のみ、両方|
|強度を変える|補正の強さに応じて振る舞いが変わるか|全方向のalpha 1・2・4など|
|直交するランダム方向|同じ大きさの摂動なら何でも効くのか|各層のFrobeniusノルムを合わせた対照|
|全列と上位64列|効果を限られたニューロンへ集中できるか|top64 alpha4からalpha6へ|
|正常な継続と反復継続|正常な推論まで一緒に壊していないか|同じtoken区間でlog probability差を比較|
|早すぎるthinking終了|単に回答を短くするだけになっていないか|早期終了の継続を対照として測定|

例えば、全方向のalpha 1・2・4では、反復側の平均token log probability差が約−0.0241、−0.0487、−0.1037 nats/tokenだった。最終top64 alpha6では反復側約−0.1031、正常側約−0.0090。これは特定のprobe上での選択性を示す測定であり、任意の文脈で同じ効果を保証する値ではない。[最終報告][report]

ランダム方向の対照は、反復方向に直交させ、層ごとの変更ノルムを合わせている。その生成比較の改善は約1.56ポイント、標準誤差約3.52ポイントで、狙った編集の効果とは異なる結果だった。[コード履歴の行72・81・93][calls]

## 9 Forward要求と評価の対応を保つ

### 生成比較

実験専用APIの要求は、概念的には次の形である。これはローカルの`experiment.py`が受け付けるJSONではない。

```json
{
  "type": "generate",
  "model": "both",
  "problems": ["実在するpractice ID"],
  "n": 1,
  "max_tokens": 32768
}
```

1要求の上限は合計512生成で、`both`なら各モデル256生成まで。正誤の差は同じ問題・同じ要求内のoriginalとeditedで対応付ける。別の要求ではサーバーのバッチ構成により乱数が変化するため、同じseed指定でも完全に再現しないという訂正が出ている。別バッチの正答率をそのまま引くと、編集効果と標本差が混ざる。[ツール仕様の訂正][corrections]

### Teacher forcing

1要求で最大1024テキスト、1テキスト最大32,768トークン、合計524,288トークンまで。層指定や返却量にも制約がある。比較時には次を確認する。

attention確率を返す要求はfull-attentionの層3・7・11・15・19・23を指定し、最大2テキスト・各2048トークンという別の制限がある。一般のhidden state要求と同じ上限で呼ばない。[編集担当への指示][prompt]

1. 元モデルと編集モデルに同一のtoken IDsを入力している。
2. 対象継続の開始位置と終了位置が一致している。
3. 次トークンlog probabilityの配列に合わせ、位置を1つずらしている。
4. paddingや異なる長さの部分を平均へ混ぜていない。

実際に1つの反復文が短くなってしまったprobeでは、共通prefixのtoken IDs一致をassertし、その範囲だけで比較し直している。[コード履歴の行101][calls]

### 要求番号と成功回数

`p50c`には`fwd_69`までの要求があるが、成功として消費された予算は50回である。失敗要求が数に入らないため、`fwd_50`を「50回目の成功」と読み替えてはいけない。実行失敗はモデルが数学を間違えたという意味でもない。[要求一覧][passes]

探索途中には16,384トークンでの比較もある。これと最終32,768トークンの評価を同じ条件の数字として混ぜない。

## 10 候補を固定してから未使用問題で確認する

最終top64 alpha6を固定した後、編集選択に使わなかったpracticeの1152問で確認し、元モデル122正答、編集モデル696正答だった。これはGSM8KとMATHの**練習用集合の中の未使用問題**であり、MATH-500のスコアではない。固定後の重みハッシュも監査されている。[最終報告][report]

その後の独立したベンチマーク結果は公開READMEでは次のように対応付けられている。

|条件|MATH-500|GSM8K|IFEval|MATH-500の上限到達率|
|---|---:|---:|---:|---:|
|未編集|7.9%|13.8%|21.0%|93.0%|
|10 passes|12.3%|25.7%|49.0%|87.2%|
|25 passes|14.1%|25.1%|39.9%|86.4%|
|50 passes|48.1%|67.1%|62.3%|33.1%|

いずれも上流が報告した値。今回の手元の検証で再測定した数字ではない。[結果とファイルの対応][setup]

元実験はthinking有効、temperature 0.6、top-p 0.95、top-k 20、presence penalty 1.5、最大32,768トークン。問題に最終回答をboxed形式にする指示を追加し、最後のboxed回答を採点する。MATH-500は1問題4sample、GSM8Kは2sampleなど、ベンチマークごとに反復数も異なる。

評価コードを移植するときは、`profiles.py`の既定値が別モデルの`qwen3`である点に注意する。対象は`qwen35think`で、`<data>`などの内部パスを自環境へ直す必要がある。`build_ckpt.py`の先頭コメントにも古いQwen3の説明が残るため、コメントだけで対象モデルを判断せず、選択されたprofileと実際のキーを確認する。[profiles.py][profiles]、[build_ckpt.py][build]

## 11 公開差分から最終値を厳密に復元する

### 必要なファイル

|ファイル|用途|
|---|---|
|固定revisionのQwenの元safetensors|編集前のBF16の値|
|`exp8_math/runs/p50c/edit/manifest.json`|キー、shape、元・編集後のハッシュ|
|`exp8_math/runs/p50c/edit/edit.npz`|変更位置と新しいBF16のbit値|
|`exp8_math/code/apply_edit.py`|差分適用とハッシュ照合|
|`exp8_math/code/st_numpy.py`|BF16をuint16として扱う読み書き|

元モデルのrevisionは`2fc06364715b967f1860aea9cf38778875588b17`に固定する。すでにFP16化したローカルcheckpointを入力に使わない。[復元スクリプト][apply]

### 新しい作業フォルダでの手順

以下はPowerShell用。新しいフォルダで実行し、既存の口調編集モデルを上書きしない。上流をsparse checkoutし、他実験の大きなファイルを展開せずに必要なコードと差分を取得する。

```powershell
New-Item -ItemType Directory qwen-edit-rebuild | Out-Null
Set-Location qwen-edit-rebuild
$env:UV_CACHE_DIR = "$PWD/cache/uv"
$env:HF_HOME = "$PWD/cache/huggingface"

uv venv --python 3.12
uv pip install --python .venv/Scripts/python.exe numpy==2.5.4 huggingface_hub==2.4.0

git clone --filter=blob:none --no-checkout https://github.com/RikiyaT/lm_understand-logs.git upstream
git -C upstream sparse-checkout init --cone
git -C upstream sparse-checkout set exp8_math/code exp8_math/runs/p50c/edit
git -C upstream checkout 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f

uv run --no-project --python .venv/Scripts/python.exe python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17', local_dir='base', allow_patterns=['*.safetensors','*.json','*.jinja','*.txt'])"
```

続いて、実際のcheckpointのファイル名を確認する。今回取得済みのsnapshotでは、単独ファイルの名前が`model.safetensors-00001-of-00001.safetensors`で、`model.safetensors.index.json`も存在した。`model.safetensors`という名前に決め打ちしない。

```powershell
Get-ChildItem base -Filter '*.safetensors'
$weightIndex = Get-Content base/model.safetensors.index.json -Raw | ConvertFrom-Json
$shardName = $weightIndex.weight_map.'model.language_model.layers.15.mlp.down_proj.weight'
$otherShard = $weightIndex.weight_map.'model.language_model.layers.19.mlp.down_proj.weight'
if ($shardName -ne $otherShard) { throw '2つのテンソルが別shardにあるため、入力の構成を確認してください' }
$baseFile = Join-Path "$PWD/base" $shardName

uv run --no-project --python .venv/Scripts/python.exe upstream/exp8_math/code/apply_edit.py $baseFile upstream/exp8_math/runs/p50c/edit rebuilt
```

indexがない単独checkpointなら、そのファイルを`$baseFile`に指定する。複数shardのcheckpointを任意の1ファイルで代用してはいけない。公式`apply_edit.py`の入力は1つのsafetensorsファイルであり、indexを自動でたどるインターフェースではない。

期待する出力は次の形になる。

```text
model.language_model.layers.15.mlp.down_proj.weight: rebuilt (sparse, 65,260 entries changed), hash OK
model.language_model.layers.19.mlp.down_proj.weight: rebuilt (sparse, 64,928 entries changed), hash OK
2 tensors -> rebuilt
```

出力先をフォルダにすると、編集した2テンソルがFP32の`.npy`として保存される。ただしその値は、評価時のBF16の値をFP32へ展開したもの。BF16へ再変換すると同じbit値になる。

出力先を`.safetensors`で終わるファイル名にすると、入力ファイルの全テンソルを保持し、対象2テンソルだけ差し替えたファイルが書かれる。

```powershell
New-Item -ItemType Directory -Force rebuilt-checkpoint | Out-Null
uv run --no-project --python .venv/Scripts/python.exe upstream/exp8_math/code/apply_edit.py $baseFile upstream/exp8_math/runs/p50c/edit rebuilt-checkpoint/model.safetensors
```

これだけではconfigやtokenizerを含むモデルディレクトリは完成しない。元のconfig・tokenizer・テンプレートも配置し、sharded indexを引き継ぐなら参照ファイル名と出力名を一致させる。1つにまとめたファイルなのに古いindexが残っている、という状態を避ける。

### 厳密復元と式の再計算の違い

`p50c`の差分は`sparse`形式で、配列名`t0_idx`・`t0_val`、`t1_idx`・`t1_val`を持つ。`idx`は平坦化したテンソル内の位置、`val`は**加算する差分ではなく、編集後のBF16のbit値**である。これを元のbit列へ代入する。

`p10c`にはlow-rank因子から再構築し、丸めの例外を上書きする形式もある。式が同じでも計算順序と丸めでbitが変わるため、最終モデルの厳密復元には公式スクリプトと公開差分を使う。[復元実装][apply]

## 12 ハッシュと変更数を検証する

ハッシュはファイル全体ではなく、対象テンソルの連続したBF16 raw bytesに対するSHA-256である。FP32の`.npy`ファイル全体のハッシュと比較してはいけない。

|対象|元BF16のSHA-256|編集後BF16のSHA-256|
|---|---|---|
|層15|`ef60d2c687bab19464dca9c8bfd7777e46ad226df0c9e25643b0bab04b5a1cca`|`82e2d733ea64feaeb4af93a1e065641d3df98b494a8b7576ef2f52e90fe2ee53`|
|層19|`638eedeb701b94aa861f4438e615c07ec6c60ee703db1bfee2de577bb6498c3e`|`8ef3c61bc36af113ad58e89dbb0a4a14dcba5a45cb750c15ab515c82ddd12eb2`|

|数える対象|値|
|---|---:|
|計算上触れる値|2 × 64 × 1024 = 131,072|
|BF16でbitが変わる値|65,260 + 64,928 = 130,188|
|編集テンソル数|2|
|各テンソルの変更列数|64|
|BF16評価値の相対Frobenius変化|層15 約10.636%、層19 約8.106%|

丸めで元の値へ戻る要素があるため、131,072と130,188は矛盾しない。また、`work_notes/frozen_weights_manifest.json`のFP32提出値のハッシュと、`edit/manifest.json`のBF16評価値のハッシュも別物である。[公開差分manifest][manifest]、[提出時manifest][frozen]

この環境では、固定revisionの元ファイルからフォルダ形式への復元を実行し、2テンソルとも公開された編集後ハッシュに一致した。結果の記録は[復元検証JSON](qwen-upstream-rebuild-verification.json)に保存している。数学スコアや元の20時間規模の探索まで再現した、という検証ではない。

## 13 前回の口調実験との違い

|項目|元実験のp50c|ローカルのずんだもん風実験|
|---|---|---|
|狙い|思考中の反復による未完了を減らす|口調指定なしで語尾を変える|
|活性の取り方|正常・反復の継続の末尾区間|通常指示・口調指示のプロンプト末尾|
|主な編集|元重みが反復方向へ書く成分を補正|平均入力に対して指示の活性差を追加|
|強度と位置の検証|強度比較、単独層、ランダム対照、疎化|固定4候補、強度探索なし|
|最終対象|層15・19の各64列|層12のdown projection|
|編集行列の相対変化|約10.636%・8.106%|約0.263%|
|推論dtype|BF16|FP16|
|生成条件|thinking有効、最大32,768トークン|thinking無効、最大256トークン|

前回の失敗から、元実験の方法でも口調変更ができないとは判断できない。逆に、反復を抑制した編集をそのまま口調へ転用すれば成功するとも言えない。次の実験で取り入れるべきなのは、層15・19やalpha6という固定値よりも、**失敗の観察、文脈をそろえたprobe、介入の選択性、未使用問題での確認**という順序である。

## 14 作業時に残す記録

候補ごとに、元モデルrevision、対象キーとshape、活性を取った位置、入力テキストとtoken IDs、方向の作り方、選択ニューロン、強度、変更ノルム、保存dtypeを残す。さらに、要求番号と成功したpass数、同一バッチでの元・編集モデルの成績、失敗要求、候補を固定した時点、固定後のハッシュを対応付ける。

最低限の確認項目は次のとおり。

- 元値を別に保存し、各候補をそこから作り直せる。
- shape、有限値、変更対象キー、変更列、dtype変換後のbit差を確認する。
- 同じ問題と同じtoken区間で比較し、別バッチの乱数変化を改善に数えない。
- cutoffの減少だけで採用せず、正答や目的の達成率も評価する。
- 正常な出力、早期終了、空白反復などの副作用を測る。
- 候補選択後は編集を固定し、選択に使わなかった問題で確認する。
- 保存後のパラメータと、再読み込み後のパラメータを照合する。

## 15 出典と利用条件

本書の上流実験の説明は、RikiyaTほかの公開ログを日本語で要約し、数式の意味とローカル復元手順を補足したもの。元ログ・データ・ネットワークはCC BY 4.0、上流のPythonスクリプトはMIT。第三者の問題データは各データセットのライセンスを引き継ぐ。上流のスクリプトを再配布する場合はMITの著作権表示と許諾文も保持する。[ライセンス][license]

[repo]: https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f
[setup]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/README.md
[prompt]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/message.txt
[calls]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/tool_calls.jsonl
[passes]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/readable/passes.md
[history]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/experiment_history.md
[report]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/final_report.md
[frozen]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/work_notes/frozen_weights_manifest.json
[manifest]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p50c/edit/manifest.json
[apply]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/apply_edit.py
[corrections]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/notices/002_tool_facts.txt
[profiles]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/profiles.py
[build]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/code/build_ckpt.py
[license]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/LICENSE
[p10]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p10c/final_answer.md
[p25]: https://github.com/RikiyaT/lm_understand-logs/blob/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp8_math/runs/p25c/final_answer.md
