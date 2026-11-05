# Weight-space Zundamon experiment

Small local replication-inspired experiment: direct weight edits to Qwen3.5-0.8B for Japanese のだ/なのだ endings. See `PROTOCOL.md` before running. This is not a faithful reproduction of the source's math benchmark.

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
uv run --no-project --python .venv/Scripts/python.exe -m pytest -q
uv run --no-project --python .venv/Scripts/python.exe experiment.py smoke
uv run --no-project --python .venv/Scripts/python.exe experiment.py prepare
uv run --no-project --python .venv/Scripts/python.exe experiment.py candidates
uv run --no-project --python .venv/Scripts/python.exe select_candidate.py
uv run --no-project --python .venv/Scripts/python.exe experiment.py final
uv run --no-project --python .venv/Scripts/python.exe analyze.py
```

Each generation is saved immediately. Rerunning stages resumes matching completed rows; do not change configuration or dataset in an existing `runs/` directory. Use a fresh working copy for a different experiment. Blind review packets are exported by `analyze.py`; reviews are independent agent judgments, not ground-truth annotations. Raw generations and checkpoints are intentionally outside Git. Small summaries and article figures are in `results/`.

## Interpretation

Activations from paired prompted/unprompted training questions inform the edit. No gradient descent is used, but data and validation feedback **are** used. The rank-one edit applies on every token and may change unrelated content. Lexical style success alone is insufficient. Compare instruction following, relevance, naturalness and examples too.

Source: https://rikiyat.github.io/blog/weight-space-programming/
