# Weight-space Zundamon experiment

Small local replication-inspired experiment: direct weight edits to Qwen3.5-0.8B for Japanese のだ/なのだ endings. See `PROTOCOL.md` before running. This is not a faithful reproduction of the source's math benchmark.

## Observed result

Final 48 prompts × 3 seeds: target endings base **0/171 sentences**, style prompt **39/547 (7.1%)**, weight edit **0/168**. No style improvement was observed for the selected edit. Four candidates all scored zero on validation; the smallest eligible change was selected by the fixed tie-break rule. `selection_passed` in the JSON means eligibility under the degeneration checks, **not experimental success**.

The Japanese unpublished Zenn draft is `articles/weight-space-zundamon.md`; figures are under `images/weight-space-zundamon/`. Numeric tables, blind agent ratings, tensor verification and provenance hashes are in `results/`. 768 generations were completed, including training/validation/side runs. Review judgments are descriptive and do not establish factual correctness. The chosen generation settings differ from the model card recommendations; results are limited to this configuration.

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

## Detailed Qwen weight-editing guide

[Qwen3.5 の重みを直接編集する調査手順と復元手順](docs/qwen-weight-editing-guide.md) documents the upstream experiment logs: activation probes, causal controls, column selection, exact weight formulas, BF16 reconstruction, hash checks and differences from this style experiment. The upstream p50c edit was reconstructed locally and both edited tensor hashes matched the published manifest; this did not rerun the math benchmarks. [Verification record](docs/qwen-upstream-rebuild-verification.json).
