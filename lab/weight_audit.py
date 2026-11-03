#!/usr/bin/env python3
"""
權重差異稽核：證明「heretic 真的被改過」，並指出改在哪裡。

為什麼需要這個檔案
------------------
量測 kl_eval 時發現「兩個模型的 log-prob 幾乎一樣」、MMLU 也一模一樣，
第一反應會是「是不是根本載到同一個模型？」。本腳本直接從 safetensors 讀張量比對，
在不載入 GPU 的情況下回答這個問題（純 CPU，可與其他實驗並行）。

實測結論（2026-11-04，Qwen3-8B → qwen3-8b-heretic）
---------------------------------------------------
* 399 個張量的**名稱完全相同** → 沒有增刪結構，是純權重改造。
* 絕大多數張量**位元完全相同**（`q_proj`、`gate_proj`、多數層的 `o_proj`/`down_proj`）。
* 只有**極少數**張量被改，例如 `model.layers.35.mlp.down_proj.weight`：
  約 78% 元素改變、最大絕對差 0.046。

這解釋了為什麼 KL 只有 0.005、MMLU 完全沒動 —— 這是一個**外科手術式**的低秩改造，
不是重新訓練。也因為如此，「用單一層的 q_proj 做指紋」會誤判成「兩個模型一樣」。

用法：
  python weight_audit.py                # 全量比對（約 1–3 分鐘，純 CPU）
  python weight_audit.py --top 10       # 只列出改動最大的 10 個張量
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

import common as C


def tensor_map(dirpath: str, pattern: str) -> dict[str, str]:
    """key -> 檔案路徑（safetensors 索引，不載入資料）。"""
    from safetensors import safe_open

    out: dict[str, str] = {}
    for f in sorted(glob.glob(os.path.join(dirpath, pattern))):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():
                out[k] = f
    return out


def snapshot_dir(repo: str, subfolder: str | None) -> str:
    from huggingface_hub import snapshot_download

    root = snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])
    return os.path.join(root, subfolder) if subfolder else root


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25, help="列出改動最大的前 N 個張量")
    ap.add_argument("--sample", type=int, default=0, help="每個張量只比對前 N 個元素（0 = 全比對）")
    args = ap.parse_args()

    from safetensors import safe_open

    bdir = snapshot_dir(C.BASE_REPO, None)
    hdir = snapshot_dir(C.HERETIC_REPO, C.HERETIC_SUBFOLDER)
    mb = tensor_map(bdir, "model-*-of-*.safetensors")
    mh = tensor_map(hdir, "model-*-of-*.safetensors")
    if set(mb) != set(mh):
        print(f"  [warn] 張量名稱不一致：只在 base {len(set(mb) - set(mh))}、"
              f"只在 heretic {len(set(mh) - set(mb))}")

    rows = []
    for i, key in enumerate(sorted(set(mb) & set(mh))):
        with safe_open(mb[key], "pt") as a, safe_open(mh[key], "pt") as c:
            tb = a.get_tensor(key)
            th = c.get_tensor(key)
        if tb.shape != th.shape:
            rows.append({"tensor": key, "changed_frac": 1.0, "max_abs_diff": float("inf"), "note": "shape mismatch"})
            continue
        if args.sample:
            tb = tb.flatten()[: args.sample]
            th = th.flatten()[: args.sample]
        changed = int((tb != th).sum())
        rows.append({
            "tensor": key,
            "numel": int(tb.numel()),
            "changed": changed,
            "changed_frac": changed / tb.numel() if tb.numel() else 0.0,
            "max_abs_diff": float((tb.float() - th.float()).abs().max()),
        })
        print(f"\r  [{i + 1}/{len(mb)}] {key[:60]:<60}", end="", flush=True)
    print()

    rows.sort(key=lambda r: r["changed_frac"], reverse=True)
    touched = [r for r in rows if r.get("changed", 0) > 0]
    summary = {
        "models": {"base": C.BASE_REPO, "heretic": C.HERETIC_REPO},
        "n_tensors": len(rows),
        "n_tensors_changed": len(touched),
        "frac_tensors_changed": len(touched) / len(rows) if rows else 0.0,
        "sample_limited": args.sample or None,
        "changed": touched,
        "unchanged_examples": [r["tensor"] for r in rows if r.get("changed", 0) == 0][:10],
    }
    out = C.results_dir() / "results_weight_audit.json"
    C.save_json(out, summary)

    print(f"\n  張量總數 {summary['n_tensors']}，有改動 {summary['n_tensors_changed']} "
          f"（{summary['frac_tensors_changed']:.1%}）")
    print(f"  {'tensor':<52} {'changed':>9} {'max|Δ|':>9}")
    for r in touched[: args.top]:
        print(f"  {r['tensor']:<52} {r['changed_frac']:>8.1%} {r['max_abs_diff']:>9.5f}")
    print(f"\n  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
