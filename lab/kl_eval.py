#!/usr/bin/env python3
"""
Phase 2：KL divergence（heretic ‖ base）— 這是「你付的代價」。

為什麼 KL 才是去對齊的主要損害指標
------------------------------------
abliteration 動的是殘差流裡的「拒絕方向」，改動會滲透到**每一個** token 的分布，
不只是第一句話的第一個字。所以：

  * 模型卡用「20 個 prompt 的第一個 token KL」得到 0.0323 —— 樣本少、只取一個位置，
    而且取的是「生成起點」那種分布最平的狀態（KL 天生偏小）。
  * 這裡量的是**全序列、全詞表**的 KL：每個位置都算，涵蓋接續、標點、句法與
    知識 recall 全部被擾動的部分。

為什麼要「兩段式」
-------------------
base 與 heretic 各 16.4 GB (bf16)，24 GB 的卡無法同時載入，也無法靠 4-bit 量化比較
（量化本身就會改變分布，會污染 KL）。所以：

  第一段：載 base → 把每個位置的 log-prob（float32，全詞表）存到磁碟
  第二段：釋放 base → 載 heretic → 讀回 base 的 log-prob，逐位置算 KL

代價是磁碟：每個 token 位置 = 151936 × 4 bytes ≈ 608 KB，
預設 20 窗 × 255 位置 ≈ 3.1 GB。要更精確就加大 --max-windows。

輸出（results/results_kl.json）
-------------------------------
  KL_text_*      : 自然文本（wikitext-2）全序列 KL ← 主要指標
  KL_firsttok_*  : 50 個良性 chat prompt 的「第一個生成 token」KL（複刻模型卡指標）
  top1_agreement : heretic 與 base 下一個 token 猜同一個的比例
  perplexity     : 兩邊在 wikitext-2 上的 perplexity（KL 的直覺對照）

用法：
  python kl_eval.py
  python kl_eval.py --max-windows 5 --seq-len 128    # 煙霧測試
  python kl_eval.py --recompute                      # 強制重算 base log-prob
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import common as C


def build_windows(tok, max_windows: int, seq_len: int) -> list[list[int]]:
    """從 wikitext-2 test 取連續 token 窗。用原始文本（不加 chat template），
    因為這裡要量的是「語言模型本身」的分布，不是對話行為。"""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    windows = []
    for start in range(0, len(ids) - seq_len, seq_len):
        windows.append(ids[start : start + seq_len])
        if len(windows) >= max_windows:
            break
    return windows


@torch.inference_mode()
def logprobs_for_window(model, input_ids: list[int], device) -> torch.Tensor:
    """回傳 (seq_len-1, vocab) 的 log-softmax，float32，在 CPU 上。"""
    ids = torch.tensor([input_ids], device=device)
    logits = model(ids).logits[0, :-1, :].float()
    return torch.log_softmax(logits, dim=-1).cpu()


@torch.inference_mode()
def first_token_logprobs(model, tok, prompts: list[str], device) -> torch.Tensor:
    """每個 chat prompt「下一個 token」的 log-softmax，回傳 (n_prompts, vocab)。"""
    out = []
    for p in prompts:
        ids = tok(p, return_tensors="pt", add_special_tokens=False).to(device)
        logits = model(**ids).logits[0, -1, :].float()
        out.append(torch.log_softmax(logits, dim=-1).cpu())
    return torch.stack(out)


def metrics(base_logp: torch.Tensor, other_logp: torch.Tensor) -> dict:
    """逐位置 KL(base ‖ other)。兩個輸入同形狀 (n_pos, vocab)。"""
    p = base_logp.exp()
    kl = (p * (base_logp - other_logp)).sum(dim=-1)  # (n_pos,)
    return {
        "n_positions": int(kl.numel()),
        "kl_mean": float(kl.mean()),
        "kl_median": float(kl.median()),
        "kl_p95": float(torch.quantile(kl, 0.95)),
        "kl_max": float(kl.max()),
        "top1_agreement": float((base_logp.argmax(dim=-1) == other_logp.argmax(dim=-1)).float().mean()),
        "tv_distance_mean": float(0.5 * (p - other_logp.exp()).abs().sum(dim=-1).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-windows", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dtype", default="bfloat16", choices=list(C.DTYPES))
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()

    device = torch.device(0)
    tok = C.load_tokenizer("base")  # base 與 heretic 共用 tokenizer（abliteration 不改詞表）
    windows = build_windows(tok, args.max_windows, args.seq_len)
    benign_chat = [C.build_prompt(tok, p["prompt"]) for p in C.load_prompts() if not p["should_refuse"]]
    print(f"  {len(windows)} 窗 × {args.seq_len} tokens；{len(benign_chat)} 個良性 chat prompt")

    cache = C.cache_dir()
    text_files = [cache / f"base_logp_w{i:03d}.npy" for i in range(len(windows))]
    ft_file = cache / "base_logp_firsttok.npy"
    need_cache = args.recompute or not all(f.exists() for f in text_files) or not ft_file.exists()

    # ---------------- 第一段：base → 存 log-prob ----------------
    ppl_base = None
    if need_cache:
        print("\n== pass 1/2：base → 快取 log-prob ==")
        base = C.load_model("base", dtype=args.dtype)
        print(f"  device: {C.gpu_report()}")
        nll_total, nt_total = 0.0, 0
        for i, w in enumerate(windows):
            lp = logprobs_for_window(base, w, device)
            np.save(text_files[i], lp.numpy())
            nll_total += float(-lp.gather(1, torch.tensor(w[1:])[:, None]).sum())
            nt_total += lp.shape[0]
            print(f"\r  [{i + 1}/{len(windows)}] saved {text_files[i].name}", end="", flush=True)
        print()
        np.save(ft_file, first_token_logprobs(base, tok, benign_chat, device).numpy())
        ppl_base = float(np.exp(nll_total / nt_total))
        print(f"  base perplexity (wikitext-2, 已量測部分) = {ppl_base:.4f}")
        C.free_model(base)
        del base
    else:
        print("\n== pass 1/2：已有快取，略過（--recompute 可強制重算）==")

    # ---------------- 第二段：heretic → 算 KL ----------------
    print("\n== pass 2/2：heretic → 計算 KL ==")
    heretic = C.load_model("heretic", dtype=args.dtype)
    print(f"  device: {C.gpu_report()}")

    all_kl, nll_total, nt_total = [], 0.0, 0
    for i, w in enumerate(windows):
        lp_h = logprobs_for_window(heretic, w, device)
        lp_b = torch.from_numpy(np.load(text_files[i]))
        m = metrics(lp_b, lp_h)
        all_kl.append(m)
        nll_total += float(-lp_h.gather(1, torch.tensor(w[1:])[:, None]).sum())
        nt_total += lp_h.shape[0]
        print(
            f"\r  [{i + 1}/{len(windows)}] KL={m['kl_mean']:.5f} top1={m['top1_agreement']:.1%}",
            end="",
            flush=True,
        )
    print()

    ppl_heretic = float(np.exp(nll_total / nt_total))
    ft_metrics = metrics(torch.from_numpy(np.load(ft_file)), first_token_logprobs(heretic, tok, benign_chat, device))
    C.free_model(heretic)

    text_metrics = {
        "n_positions": sum(m["n_positions"] for m in all_kl),
        "kl_mean": float(np.mean([m["kl_mean"] for m in all_kl])),
        "kl_median": float(np.median([m["kl_median"] for m in all_kl])),
        "kl_p95": float(np.mean([m["kl_p95"] for m in all_kl])),
        # 取所有窗中的最大值（不是平均），因為要看的是「最壞的那一段」被改多少
        "kl_max": float(np.max([m["kl_max"] for m in all_kl])),
        "top1_agreement": float(np.mean([m["top1_agreement"] for m in all_kl])),
        "tv_distance_mean": float(np.mean([m["tv_distance_mean"] for m in all_kl])),
    }


    result = {
        "settings": vars(args),
        "models": {"base": C.BASE_REPO, "heretic": C.HERETIC_REPO},
        "KL_text": text_metrics,
        "KL_first_token": ft_metrics,
        "perplexity": {"base": ppl_base, "heretic": ppl_heretic},
        "per_window": all_kl,
    }
    out = C.results_dir() / "results_kl.json"
    C.save_json(out, result)

    print("\n  ── 結果 ──")
    print(
        f"  全序列 KL (heretic‖base)      mean {text_metrics['kl_mean']:.5f}  "
        f"median {text_metrics['kl_median']:.5f}  p95 {text_metrics['kl_p95']:.5f}"
    )
    print(f"  第一個 token KL（複刻模型卡）  mean {ft_metrics['kl_mean']:.5f}")
    print(f"  top-1 一致率                  {text_metrics['top1_agreement']:.1%}")
    if ppl_base:
        print(
            f"  perplexity                    base {ppl_base:.4f} → heretic {ppl_heretic:.4f} "
            f"({(ppl_heretic / ppl_base - 1) * 100:+.2f}%)"
        )
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

