#!/usr/bin/env python3
"""
人工稽核用：列出每一題「為什麼」被判成拒答（哪個關鍵字命中）。

關鍵字法最大的問題是誤判 —— 例如正常回答裡出現「危險」「違法」就會被算成拒答。
本腳本把命中的關鍵字與原始回覆攤開，讓你判断 38% 這種數字到底是
「真的還在拒絕」還是「字串剛好撞到」。

用法：
  python inspect_refusals.py                     # heretic 的所有拒答
  python inspect_refusals.py --model base --all
  python inspect_refusals.py --category self_harm
"""
from __future__ import annotations

import argparse
import sys

import common as C


def matched_keywords(text: str) -> list[str]:
    low = text.lower()
    for neg in C.SOFT_NEGATIVE_CONTEXT:
        low = low.replace(neg, "")
    return [kw for kw in C.REFUSAL_KEYWORDS if kw.lower() in low]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "heretic"], default="heretic")
    ap.add_argument("--category")
    ap.add_argument("--all", action="store_true", help="連未判為拒答的題目也列")
    ap.add_argument("--width", type=int, default=160)
    args = ap.parse_args()

    path = C.results_dir() / f"results_refusal_{args.model}.json"
    if not path.exists():
        sys.exit(f"[fail] 找不到 {path}，請先跑 refusal_eval.py")
    data = C.load_json(path)

    rows = data["rows"]
    if args.category:
        rows = [r for r in rows if r["category"] == args.category]
    if not args.all:
        rows = [r for r in rows if r["refused"]]

    print(f"{args.model}: {len(rows)} 題（{'全部' if args.all else '判為拒答'}）\n")
    for r in rows:
        kws = matched_keywords(r["response"])
        tag = "REJECT" if r["refused"] else "      "
        print(f"[{tag}] {r['id']} ({r['category']}, should_refuse={r['should_refuse']})")
        print(f"  Q: {r['prompt'][:120]}")
        print(f"  A: {r['response'][: args.width].replace(chr(10), ' ')}")
        print(f"  命中關鍵字: {kws if kws else '（無）'}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
