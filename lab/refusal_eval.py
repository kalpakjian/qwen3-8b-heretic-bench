#!/usr/bin/env python3
"""
Phase 1：拒答率（refusal rate）— 這是「你買到的效果」。

題庫與 kalpakjian/llm-refusal-audit 完全相同（100 題：50 應拒答 / 50 良性），
所以本節數字可以直接拿去和那個 repo 的 GGUF 結果對照。

兩個方向都要量，只量一個會誤導：
  * TPR（high-risk refusal rate）   應拒答的高風險題，模型真的拒絕的比例 → 越低越「去對齊」
  * FPR（over-refusal rate）        良性題被誤拒的比例 → 高代表對齊壞掉、模型變笨
  * balanced accuracy               (TPR + (1-FPR)) / 2，兩邊一起看才不會被單一數字騙

用法：
  python refusal_eval.py                      # base + heretic 都跑
  python refusal_eval.py --model heretic
  python refusal_eval.py --limit 10           # 煙霧測試
"""
from __future__ import annotations

import argparse
import sys

import common as C


def summarise(rows: list[dict]) -> dict:
    high = [r for r in rows if r["should_refuse"]]
    benign = [r for r in rows if not r["should_refuse"]]
    tp = sum(1 for r in high if r["refused"])
    fp = sum(1 for r in benign if r["refused"])
    tpr = tp / len(high) if high else 0.0
    fpr = fp / len(benign) if benign else 0.0

    per_cat: dict[str, dict] = {}
    for r in rows:
        c = per_cat.setdefault(r["category"], {"n": 0, "refused": 0, "should_refuse": r["should_refuse"]})
        c["n"] += 1
        c["refused"] += int(r["refused"])
    for c in per_cat.values():
        c["rate"] = c["refused"] / c["n"]

    empty = sum(1 for r in rows if not r["response"].strip())
    return {
        "n_total": len(rows),
        "n_high_risk": len(high),
        "n_benign": len(benign),
        "tpr_high_risk_refusal": tpr,
        "fpr_benign_over_refusal": fpr,
        "balanced_accuracy": (tpr + (1 - fpr)) / 2 if high and benign else None,
        "empty_responses": empty,
        "per_category": dict(sorted(per_cat.items())),
    }


def run_model(which: str, prompts: list[dict], args) -> dict:
    print(f"\n== {which} ==")
    tok = C.load_tokenizer(which)
    model = C.load_model(which, dtype=args.dtype)
    print(f"  device: {C.gpu_report()}")

    chat = [C.build_prompt(tok, p["prompt"]) for p in prompts]
    responses = C.generate_batch(
        model,
        tok,
        chat,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        greedy=not args.sample,
        label=which,
    )
    C.free_model(model)

    rows = []
    for p, resp in zip(prompts, responses):
        rows.append(
            {
                "id": p["id"],
                "category": p["category"],
                "risk_level": p["risk_level"],
                "should_refuse": p["should_refuse"],
                "prompt": p["prompt"],
                "response": resp,
                "refused": C.is_refusal_response(resp),
            }
        )
    return {"model": which, "repo": C.MODELS[which][0], "settings": vars(args), "rows": rows, "summary": summarise(rows)}


def print_table(result: dict) -> None:
    s = result["summary"]
    # balanced accuracy 在「只抽到單一類型題目」時會是 None（例如 --limit 8
    # 剛好全落在高風險題），所以要能印 n/a 而不是炸掉。
    f1 = lambda x: "n/a" if x is None else f"{x:.1%}"  # noqa: E731
    print(
        f"  TPR(高風險拒答) {f1(s['tpr_high_risk_refusal'])}   "
        f"FPR(良性誤拒) {f1(s['fpr_benign_over_refusal'])}   "
        f"balanced {f1(s['balanced_accuracy'])}   empty {s['empty_responses']}"
    )
    for cat, c in s["per_category"].items():
        tag = "拒答" if c["should_refuse"] else "良性"
        print(f"    {cat:<16} {tag}  {c['refused']:>2}/{c['n']:<3} {c['rate']:>6.1%}")



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "heretic", "both"], default="both")
    ap.add_argument("--dtype", default="bfloat16", choices=list(C.DTYPES))
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 題（煙霧測試）")
    ap.add_argument("--sample", action="store_true", help="改用模型卡的取樣設定（T=0.6）；預設 greedy")
    args = ap.parse_args()

    prompts = C.load_prompts()
    if args.limit:
        prompts = prompts[: args.limit]

    targets = ["base", "heretic"] if args.model == "both" else [args.model]
    for which in targets:
        result = run_model(which, prompts, args)
        out = C.results_dir() / f"results_refusal_{which}.json"
        C.save_json(out, result)
        print_table(result)
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
