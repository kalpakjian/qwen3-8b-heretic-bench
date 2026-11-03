#!/usr/bin/env python3
"""
從 `kalpakjian/llm-refusal-audit` 的 `llm_safety_test.py` 抽出 100 題測試集，
輸出成 `prompts_refusal.json`。

為什麼要抽而不是重寫？
  兩邊用同一份題庫，拒答率才能直接跨 repo 對照（那個 repo 已公布
  heretic q4_K_S 0% / q4_K_M 4% / q8_0 2% 的 keyword TPR）。
  本 repo 只是把同一批題目改跑在 **bf16 全精度 safetensors** 上。

做法：用 ast 靜態解析 `TEST_PROMPTS` 字面值，不 import 模組
（該模組會 import ollama / sklearn / matplotlib，這裡不需要那些相依）。

用法：
  python extract_prompts.py                       # 從 GitHub 抓上游
  python extract_prompts.py --source local.py     # 從本機檔案抽
"""
from __future__ import annotations

import argparse
import ast
import json
import urllib.request
from pathlib import Path

UPSTREAM = "https://raw.githubusercontent.com/kalpakjian/llm-refusal-audit/main/llm_safety_test.py"
OUT = Path(__file__).resolve().parent / "prompts_refusal.json"


def extract(src: str) -> list[dict]:
    tree = ast.parse(src)
    for node in tree.body:
        # 目標形式： TEST_PROMPTS: List[Dict] = [...]
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) else node.target
        if getattr(target, "id", None) != "TEST_PROMPTS":
            continue
        value = node.value
        if not isinstance(value, ast.List):
            raise SystemExit("[fail] TEST_PROMPTS 不是 list 字面值，無法靜態解析")
        # literal_eval 需要可雜湊的節點；逐項轉比較安全
        return [ast.literal_eval(el) for el in value.elts]
    raise SystemExit("[fail] 找不到 TEST_PROMPTS")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", help="本機的 llm_safety_test.py；省略則從 GitHub 抓")
    args = ap.parse_args()

    if args.source:
        src = Path(args.source).read_text(encoding="utf-8")
    else:
        print(f"  -> {UPSTREAM}")
        with urllib.request.urlopen(UPSTREAM, timeout=60) as r:
            src = r.read().decode("utf-8")

    prompts = extract(src)

    # 基本健檢：欄位齊全、拒答題 / 良性題各有幾題
    required = {"id", "category", "risk_level", "prompt", "should_refuse"}
    for p in prompts:
        missing = required - set(p)
        if missing:
            raise SystemExit(f"[fail] {p.get('id')} 缺欄位 {missing}")

    refuse = sum(1 for p in prompts if p["should_refuse"])
    benign = len(prompts) - refuse
    OUT.write_text(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"  [ok] {len(prompts)} 題 -> {OUT}")
    print(f"       應拒答 {refuse} 題 / 良性 {benign} 題")

    cats: dict[str, int] = {}
    for p in prompts:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print("       類別：" + ", ".join(f"{k}={v}" for k, v in sorted(cats.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
