# Qwenの重み編集による口調制御

Qwen3.5-0.8Bの重みを直接編集し、推論時に口調を指定しなくても「のだ／なのだ」のような語尾を使うようにできるかを調べるプロジェクト。

参考にしているのは[RikiyaTらの実験ログ](https://github.com/RikiyaT/lm_understand-logs)のExperiment 8。元実験は数学回答中の反復を抑える研究で、本プロジェクトではその調査方法を口調制御へ応用する。数学ベンチマークの忠実な再現ではない。

## 実験結果

最新の編集では、自然な「のだ／なのだ」語尾を既存100問中77問、新規50問中36問で確認しました。語尾の評価に回答内容の正誤は含めていません。[最新の報告書](experiments/2026-11-05_1817/continuation_refine/REPORT.md)

| 実験 | 調べたこと |
| --- | --- |
| [第1回](experiments/2026-11-05_1536/README.md) | プロンプト末尾の活性差を使った語尾編集 |
| [第2回](experiments/2026-11-05_1702/README.md) | 回答中の活性を使った編集と対照実験 |
| [第3回](experiments/2026-11-05_1817/README.md) | 口調指示なしで自然な語尾を生成するための重み編集 |
| [追加改善](experiments/2026-11-05_1817/continuation_refine/README.md) | 品詞別の編集と語尾後の反復の抑制 |


## このリポジトリの使い方

```text
experiments/
  YYYY-MM-DD_HHMM/     実験ごとのコード・設定・入力・結果・README
    data/             固定した入力
    runs/             生成ログ・内部活性・品質レビュー（Git管理外）
    results/          集計・選択記録・ハッシュ検証
    models/           編集モデル・差分（Git管理外）
docs/                 元実験の調査資料・復元検証記録
articles/             記事原稿
images/               記事掲載用の画像
run_experiment.py     共通環境から実験を起動する入口
requirements.lock     固定した依存ライブラリ
.venv/                共通のPython環境（Git管理外）
cache/                モデル等の共有キャッシュ（Git管理外）
```

実験ディレクトリの日時はJST。実験の狙い・条件・結果・再実行手順は、各ディレクトリのREADMEに記載する。内部構成は実験により異なるため、実際のパスはそのREADMEを参照する。

新しい試行では新しい日時のディレクトリを作り、必要なコード・設定・入力のみを用意する。既存の生成ログ・集計結果・編集モデル・凍結マニフェストは流用しない。同じ実験の途中再開と、条件を変えた新規実験を区別する。共通環境の依存を変更する場合は、各実験の固定依存との整合も確認する。

### 環境の準備（Windows・NVIDIA GPU）

2026-11-08のローカル整理で、共有キャッシュ・仮想環境・各実験のモデル重みと編集差分を削除した。コード・入力・集計結果・レビュー・記事は保持している。再実行する際は、以下の手順で環境とベースモデルを取得し、各実験のREADMEに従って編集モデルを再生成する。

リポジトリルートで実行する。

```powershell
$env:UV_CACHE_DIR = "$PWD/cache/uv"
$env:HF_HOME = "$PWD/cache/huggingface"
uv venv --python 3.12
uv pip sync --python .venv/Scripts/python.exe requirements.lock --extra-index-url https://download.pytorch.org/whl/cu126
uv run --no-project --python .venv/Scripts/python.exe python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', revision='2fc06364715b967f1860aea9cf38778875588b17')"
```

モデル取得にはネットワークを使う。実験コードは取得済みモデルをオフラインで読み込む。モデル本体約3GBに加え、CUDA版PyTorch・編集モデル・生成ログの保存容量が必要。公開checkpointには視覚部分も含まれるが、口調実験は言語モデル部分を読み込む。

`uv run --no-project`は、`pyproject.toml`から依存を再解決せず、固定した環境を使うために指定している。

### 実験の起動

`$experimentId`には対象ディレクトリ名を指定する。以下は書式の例であり、`YYYY-MM-DD_HHMM`をそのまま実行するものではない。

```powershell
$env:UV_CACHE_DIR = "$PWD/cache/uv"
$experimentId = 'YYYY-MM-DD_HHMM'
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py $experimentId -m pytest -q
uv run --no-project --python .venv/Scripts/python.exe run_experiment.py $experimentId experiment.py --help
```

`run_experiment.py`は対象実験を作業ディレクトリにして、同じPython環境でスクリプトを実行する。`HF_HOME`・`UV_CACHE_DIR`・`MPLCONFIGDIR`が未指定なら、ルートの共有キャッシュを指定する。スクリプト名や段階ごとの引数は各実験のREADMEに従う。モデルと生ログはGitに含まれないため、新しいcheckoutでは生成段階から実行する必要がある。

## 関連資料

- [上流の Experiment 8 の概要](docs/upstream-experiment-8.md)
- [Qwen重み編集の詳細ガイド](docs/qwen-weight-editing-guide.md)
- [口調制御実験の記事原稿](articles/weight-space-zundamon.md)
