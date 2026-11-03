# qwen3-8b-heretic — ARC benchmark

Self-hosted, reproducible ARC benchmark for
[`kalpakjian/qwen3-8b-heretic`](https://huggingface.co/kalpakjian/qwen3-8b-heretic)
and its base [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B).

No third-party benchmarking service. You get the same numbers a vendor would
hand you, minus the attribution and publication terms — and the run record is
signed by *you*, which is stronger evidence anyway.

## Quick start (Omarchy / Arch, NVIDIA)

```bash
git clone <this-repo> qwen3-heretic-bench && cd qwen3-heretic-bench
./scripts/setup-cuda-arch.sh          # installs deps, builds venv, verifies CUDA
source .venv/bin/activate

python bench_arc.py --limit 20 --tag smoke     # ~2 min plumbing check
python bench_arc.py --no-thinking --tag heretic
python bench_arc.py --no-thinking --tag base --model Qwen/Qwen3-8B --root-weights
```

`setup-cuda-arch.sh` is idempotent. Flags:

| Flag | Effect |
|---|---|
| `--check` | Verify only, install nothing |
| `--vllm` | Also install vLLM (much faster runs) |

## Windows

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install -r requirements.txt
python bench_arc.py --no-thinking --tag heretic
```

**The `--index-url` is not optional.** A plain `pip install torch` on Windows
frequently resolves to a `+cpu` build. Verify before running:

```powershell
python -c "import torch; print(torch.cuda.is_available())"   # must be True
```

`bench_arc.py` also checks this itself and aborts with a clear message before
downloading any weights.

## Dual-boot pre-flight (Windows side)

Before touching BIOS or installing Omarchy alongside Windows, run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-dualboot.ps1
```

Run it **elevated** — otherwise the BitLocker check reports `UNKNOWN` rather
than a real answer. It is strictly read-only. It reports:

- BitLocker / device-encryption state — **back up the 48-digit recovery key
  before changing Secure Boot or TPM, or you can lose the Windows volume**
- Secure Boot and TPM state (Omarchy's installer requires these disabled)
- Fast Startup (must be off, or Linux cannot mount NTFS cleanly)
- Per-disk partition layout, including which disk holds each EFI System Partition
- Whether a non-Microsoft bootloader is already present

If it finds an existing Linux bootloader, **you may not need to install
anything** — just select it from the firmware boot menu.

## Commands

| Command | Purpose |
|---|---|
| `python bench_arc.py --limit 20` | Smoke test. Score is meaningless; proves plumbing |
| `python bench_arc.py` | Full ARC-Challenge, 25-shot, Heretic model |
| `... --model Qwen/Qwen3-8B --root-weights` | Base model, identical config |
| `... --backend vllm` | Same task/config, much faster |
| `... --tasks arc_challenge,arc_easy` | Both ARC subsets |
| `... --dry-run` | Print the lm-eval command, run nothing |
| `... --skip-vram-check` | Bypass the GPU pre-check |

Run `python bench_arc.py --help` for the full flag list.

## Why the base model matters

An abliterated variant is only interesting through its **delta** against base.
Run both with identical config, otherwise the number means nothing.

## The two settings that actually change the score

**`--no-thinking`** (recommended). Qwen3-8B is a hybrid-reasoning model and by
default emits a long ` thinking` block before answering. ARC is a direct-answer
task, so thinking mode inflates runtime, can blow the context on 25-shot
prompts, and makes `max_gen_toks` a dominant free parameter.

Whatever you choose, **state it on the model card**. A score without that label
is not reproducible — which is precisely the criticism you would be levelling
at a benchmarking vendor.

**`--num-fewshot 25`** (default). Conventional ARC-Challenge setup, and what
published Qwen3-8B numbers use. Do not drop to 0-shot and then compare against
published tables.

## Outputs

`results/` gets four files per run (git-ignored — regenerate, never hand-edit):

| File | Purpose |
|---|---|
| `*_<stamp>.json` | Raw lm-eval output (accuracy, stderr, configs, versions) |
| `*_META.json` | GPU, lm-eval version + **git commit**, transformers version, full command, wall time |
| `*_TABLE.md` | Paste-ready markdown table for the model card |
| `*_RERUN.txt` | Single command anyone can re-execute to verify |

The commit hash in `_META.json` is what makes the run checkable.

## Caveats

- Weights live under `model_files/` in that HF repo, not at the root. The
  default `--subfolder model_files` handles it; `--root-weights` is for the
  base model.
- **Do not mix HF and vLLM results in one table.** Logits differ slightly
  between backends, so that is not a controlled comparison.
- ARC only measures grade-school science multiple choice. It is a weak proxy
  for chat quality and says nothing about refusal behaviour. The repo's
  existing `refusal_audit_*` reports cover that and should stay on the card.
- Report `acc` and `acc_norm` separately. Conflating them is the most common
  way ARC tables get inflated.
