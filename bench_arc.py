#!/usr/bin/env python
"""
Reproducible ARC benchmark for kalpakjian/qwen3-8b-heretic (and Qwen3-8B base).

Writes a results JSON plus a paste-ready markdown table for the model card.

Deps:
    pip install -U lm-eval transformers accelerate torch
    (optional, faster)  pip install vllm

Examples
--------
# Smoke test first (cheap sanity check, ~2 min on one GPU)
python bench_arc.py --backend hf --limit 20

# The real run against the Heretic model
python bench_arc.py --backend hf

# Base model, identical config -- needed for the delta
python bench_arc.py --backend hf --model Qwen/Qwen3-8B --subfolder "" --tag base

# If you have vLLM (much faster, same task/config)
python bench_arc.py --backend vllm
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Defaults: deliberately match the conventional lm-eval ARC-Challenge setup so
# numbers are comparable with published Qwen3-8B results.
# --------------------------------------------------------------------------- #
DEFAULT_MODEL = "kalpakjian/qwen3-8b-heretic"
DEFAULT_SUBFOLDER = "model_files"  # weights live here, NOT at repo root
DEFAULT_TASKS = "arc_challenge"
DEFAULT_FEWSHOT = 25
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Reproducible ARC benchmark runner (lm-eval wrapper).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path.")
    p.add_argument("--subfolder", default=DEFAULT_SUBFOLDER,
                   help="HF subfolder holding weights. Use '' for repo root / base.")
    p.add_argument("--root-weights", action="store_true",
                   help="Shortcut for --subfolder '' (base model / weights at repo root).")
    p.add_argument("--backend", choices=["hf", "vllm"], default="hf",
                   help="lm-eval backend: 'hf' (transformers) or 'vllm'.")
    p.add_argument("--tasks", default=DEFAULT_TASKS,
                   help="Comma-separated lm-eval tasks, e.g. arc_challenge,arc_easy")
    p.add_argument("--num-fewshot", type=int, default=DEFAULT_FEWSHOT)
    p.add_argument("--batch-size", default="auto",
                   help="lm-eval batch size; 'auto' lets lm-eval size it.")
    p.add_argument("--limit", type=int, default=None,
                   help="Evaluate only N samples (smoke test / cost cap).")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default=None, help="e.g. cuda:0. Default: auto.")
    p.add_argument("--tp-size", type=int, default=1, help="vLLM tensor-parallel size.")
    p.add_argument("--gpu-mem-util", type=float, default=0.90,
                   help="vLLM GPU memory utilisation.")
    p.add_argument("--max-model-len", type=int, default=8192,
                   help="vLLM max context. 8192 is plenty for ARC.")
    # Qwen3 specifics
    p.add_argument("--no-thinking", action="store_true",
                   help="Answer directly instead of reasoning first. Recommended: "
                        "ARC is a direct-answer task and Qwen3 thinking mode will "
                        "burn the token budget and can exceed the 8k context.")
    p.add_argument("--apply-chat-template", action="store_true",
                   help="Wrap prompts in the chat template (non-standard for ARC; "
                        "only use if you intend a chat-style eval and say so).")
    p.add_argument("--seed", default="0,1234,1234,1234",
                   help="lm-eval seed arg: seeds,nsamples,numpy,torch")
    p.add_argument("--tag", default=None,
                   help="Label for output files, e.g. 'heretic' or 'base'.")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--log-samples", action="store_true",
                   help="Write per-sample generations (big; useful for auditing).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command and exit without running.")
    p.add_argument("--skip-vram-check", action="store_true",
                   help="Don't warn/abort when free VRAM looks insufficient.")
    return p.parse_args()



def git_sha(repo: Path) -> str | None:
    """Pinned commit hash of the harness -- the thing that makes a run checkable."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def harness_info() -> dict:
    info: dict = {}
    try:
        import lm_eval  # noqa: PLC0415

        info["lm_eval_version"] = getattr(lm_eval, "__version__", "unknown")
        info["lm_eval_commit"] = git_sha(Path(lm_eval.__file__).resolve().parent)
    except ImportError:
        info["lm_eval_version"] = "NOT INSTALLED"
    try:
        import torch  # noqa: PLC0415

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_total_mem_gib"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except ImportError:
        info["torch_version"] = "NOT INSTALLED"
    try:
        import transformers  # noqa: PLC0415

        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "NOT INSTALLED"
    info["python"] = platform.python_version()
    info["platform"] = f"{platform.system()} {platform.release()}"
    return info


def resolve_model_id(model: str, subfolder: str, dry_run: bool = False) -> str:
    """lm-eval wants a single path. HF subfolders get fetched locally first."""
    if not subfolder or Path(model).exists():
        return model
    if dry_run:
        # Don't pull 15 GB just to print a command line.
        return f"<{model}:{subfolder} (fetched on real run)>"
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError:
        print("!! huggingface_hub missing; assuming weights at repo root.",
              file=sys.stderr)
        return model

    print(f"-> Resolving {model}/{subfolder} ...", flush=True)
    return snapshot_download(repo_id=model, allow_patterns=[f"{subfolder}/*"])


def build_cmd(args: argparse.Namespace, model_path: str, out_path: Path) -> list[str]:
    if args.backend == "vllm":
        model_args = ",".join([
            f"pretrained={model_path}",
            "dtype=auto",
            f"tensor_parallel_size={args.tp_size}",
            f"gpu_memory_utilization={args.gpu_mem_util}",
            f"max_model_len={args.max_model_len}",
        ])
    else:
        model_args = ",".join([f"pretrained={model_path}", f"dtype={args.dtype}"])

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", args.backend,
        "--model_args", model_args,
        "--tasks", args.tasks,
        "--num_fewshot", str(args.num_fewshot),
        "--batch_size", str(args.batch_size),
        "--seed", args.seed,
        "--output_path", str(out_path),
    ]
    if args.device and args.backend == "hf":
        cmd += ["--device", args.device]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.log_samples:
        cmd += ["--log_samples"]

    # --- Qwen3 thinking-mode handling -------------------------------------- #
    extras: list[str] = []
    if args.apply_chat_template:
        extras += ["apply_chat_template=true", "fewshot_as_multiturn=true"]
    if args.no_thinking:
        # Deterministic, short, stop at end-of-turn: right for a direct-answer
        # task. Keeps runs reproducible and inside the context window.
        extras.append("gen_kwargs=do_sample=false,max_gen_toks=16,until=<|im_end|>")
    if extras:
        cmd[cmd.index("--model_args") + 1] += "," + ",".join(extras)
    return cmd


# Approximate resident weights for the configs we actually run. Deliberately
# conservative -- better to warn early than to eat a CUDA OOM 40 minutes in.
WEIGHT_GIB = {
    "bfloat16": 16.0,   # 8.2B params ~= 15.3 GiB + overhead
    "float16": 16.0,
    "float32": 33.0,
}


def vram_report(dtype: str, backend: str) -> tuple[bool, str]:
    """Return (ok, human-readable message) for the current GPU free memory."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return True, "torch not importable -- skipping VRAM check."

    if not torch.cuda.is_available():
        return False, (
            "torch.cuda.is_available() is False.\n"
            "  This is usually a CPU-only torch build. Reinstall:\n"
            "    pip install --index-url https://download.pytorch.org/whl/cu128 torch"
        )

    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gib = free_b / 1024**3
    total_gib = total_b / 1024**3

    need = WEIGHT_GIB.get(dtype, 16.0)
    # vLLM preallocates gpu_memory_utilization up front, so it needs the whole
    # slice free; transformers grows into it and needs headroom for activations.
    reserve = 2.0 if backend == "hf" else 0.5
    required = need + reserve

    msg = (f"GPU      : {torch.cuda.get_device_name(0)}\n"
           f"VRAM     : {free_gib:.1f} GiB free / {total_gib:.1f} GiB total\n"
           f"Needed   : ~{required:.0f} GiB ({dtype} weights {need:.0f} + headroom)")

    if free_gib < required:
        msg += (f"\n\nSHORT BY: {required - free_gib:.1f} GiB\n"
                "  Something else is holding the GPU. Check with:\n"
                "    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv\n"
                "    nvidia-smi   # look for C+G entries; on Windows that is browsers/Electron apps")
        return False, msg
    return True, msg


def table_from_results(results: dict, label: str) -> str:
    """Markdown table from lm-eval's results dict, paste-ready for a model card."""
    lines = ["| Task | acc | acc_norm | stderr |", "|---|---|---|---|"]
    for task, metrics in results.get("results", {}).items():
        if "acc,none" not in metrics and "acc_norm,none" not in metrics:
            continue

        def fmt(key: str) -> str:
            val = metrics.get(key)
            return f"{val:.4f}" if isinstance(val, (int, float)) else "-"

        lines.append(
            f"| {task} | {fmt('acc,none')} | "
            f"{fmt('acc_norm,none')} | {fmt('acc_stderr,none')} |"
        )
    return f"### {label}\n\n" + "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    if args.root_weights:
        args.subfolder = ""
    args.outdir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or args.model.split("/")[-1]
    stem = f"{tag}_{args.tasks.replace(',', '-')}_{args.num_fewshot}shot_{stamp}"
    out_path = args.outdir / f"{stem}.json"

    # Check the GPU BEFORE pulling 15 GB of weights -- fail fast and cheap.
    if not args.dry_run and not args.skip_vram_check:
        ok, detail = vram_report(args.dtype, args.backend)
        print("\n=== GPU check ===")
        print(detail)
        if not ok:
            print("\nRefusing to start. Fix the above, or pass --skip-vram-check.",
                  file=sys.stderr)
            return 2
        print()

    model_path = resolve_model_id(args.model, args.subfolder, dry_run=args.dry_run)
    cmd = build_cmd(args, model_path, out_path)

    print("\n=== resolved model path ===")
    print(model_path)
    print("\n=== command ===")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd), "\n")

    if args.dry_run:
        return 0

    t0 = time.time()
    rc = subprocess.call(cmd)
    elapsed_s = round(time.time() - t0, 1)

    if rc != 0:
        print(f"\n!! lm-eval exited {rc}", file=sys.stderr)
        return rc
    if not out_path.exists():
        print(f"\n!! expected results at {out_path} but file missing", file=sys.stderr)
        return 1

    results = json.loads(out_path.read_text(encoding="utf-8"))
    hw = harness_info()

    meta = {
        "tag": tag,
        "model": args.model,
        "subfolder": args.subfolder,
        "resolved_model_path": model_path,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "backend": args.backend,
        "thinking_enabled": not args.no_thinking,
        "chat_template": args.apply_chat_template,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "limit": args.limit,
        "elapsed_seconds": elapsed_s,
        "harness": hw,
        "command": cmd,
        "raw_results_file": out_path.name,
        "timestamp_utc": stamp,
    }
    (args.outdir / f"{stem}_META.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # One-line command anyone can re-execute to verify the run.
    (args.outdir / f"{stem}_RERUN.txt").write_text(
        " ".join(f'"{c}"' if " " in c else c for c in cmd)
        + f" --output_path {stem}_rerun.json\n",
        encoding="utf-8",
    )

    md = table_from_results(results, f"{args.tasks} ({args.num_fewshot}-shot)")
    md += (
        f"\n*lm-eval {hw.get('lm_eval_version')}"
        f" @ `{hw.get('lm_eval_commit') or 'n/a'}`, "
        f"{hw.get('gpu_name', 'unknown GPU')}, "
        f"seed `{args.seed}`, thinking `{not args.no_thinking}`, "
        f"batch `{args.batch_size}`, {elapsed_s}s.*\n"
    )
    md_path = args.outdir / f"{stem}_TABLE.md"
    md_path.write_text(md, encoding="utf-8")

    print("=" * 70)
    print(md)
    print("=" * 70)
    print(f"results : {out_path}")
    print(f"table   : {md_path}")
    print(f"elapsed : {elapsed_s}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
