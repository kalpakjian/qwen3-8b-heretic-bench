#!/usr/bin/env python3
"""
Phase 3：能力保留（capability retention）— 這是「你付的價格」。

三個互補的量測，刻意選「便宜但可信」的協定，讓 24 GB 的單卡能跑完：

  MMLU   5-shot、只做一次前向、比對 A/B/C/D 的 log-prob（不生成）
         → 知識面。abliteration 若傷到知識層會在這裡掉。
  GSM8K  0-shot、greedy 生成、抽 `#### <數字>` 比對
         → 多步推理。最容易顯露「腦子被改壞」的任務。
  PPL    wikitext-2 perplexity
         → 語言模型本身。與 KL 同源，但數字更好懂。

用法：
  python capability_eval.py                          # 兩個模型、全部任務
  python capability_eval.py --model heretic --tasks mmlu
  python capability_eval.py --limit-mmlu 2 --limit-gsm8k 10   # 煙霧測試
"""
from __future__ import annotations

import argparse
import re
import sys

import torch

import common as C

# 16 個涵蓋 STEM / 人文 / 社會的 MMLU 科目（每個科目都取 test split 前 N 題）
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
]
LETTERS = ["A", "B", "C", "D"]


MMLU_SYSTEM = (
    "You are a precise multiple-choice test taker. "
    "Reply with exactly one letter: A, B, C or D. Never explain."
)


def format_question(row: dict) -> str:
    """把題目與選項排成 MMLU 標準格式。

    **關鍵坑**：`cais/mmlu` 的 `question` 欄位**不含選項**（選項在獨立的 `choices`
    清單裡）。第一版沒把選項接上去，等於叫模型「在看不到選項的情況下選 A/B/C/D」，
    量到的準確率剛好是 25%（隨機）——卻很容易被誤讀成「模型在 MMLU 上只有 25%」。
    """
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, row["choices"]))
    return f"{row['question']}\n{opts}\nAnswer:"


def mmlu_messages(which_tok, subject: str, shots: list[dict], row: dict) -> str:
    """5-shot **多輪**對話（thinking off）。

    為什麼是多輪而不是把 5 個範例塞進同一個 user message：實測單一訊息版本
    會讓 Qwen3 進入「續寫文件」模式，答案位置的 top-5 token 是
    `Let / We / Here / ### / To`，逼它選字母就會出現嚴重的位置偏誤
    （48 題裡 32 題答 'A'，準確率剛好 25%）。改成真正的 assistant 輪次後，
    選項分布回到均勻（11/12/15/10）。
    """
    msgs = [{"role": "system", "content": MMLU_SYSTEM}]
    for ex in shots:
        msgs.append({"role": "user", "content": f"Subject: {subject.replace('_', ' ')}\n\n{format_question(ex)}"})
        msgs.append({"role": "assistant", "content": LETTERS[ex["answer"]]})
    msgs.append({"role": "user", "content": f"Subject: {subject.replace('_', ' ')}\n\n{format_question(row)}"})
    return which_tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def parse_letter(text: str) -> int:
    """從回覆中抓第一個**獨立出現**的 A/B/C/D；抓不到回 -1（報告會統計 unparsed）。

    必須用「前後不是英文字母」的邊界，否則 "The answer is D" 會因為
    answer 裡的 'a' 被抓成 A（這個 bug 是 selftest 的回歸測試抓到的）。
    """
    m = re.search(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text.upper()[:20])
    return LETTERS.index(m.group(1)) if m else -1





def mmlu_eval(model, tok, n_per_subject: int, label: str, batch_size: int = 8, debug: int = 0) -> dict:
    from datasets import load_dataset

    # 一次載入 `all` 再按 subject 分組（逐科呼叫會讓 datasets 5.x 每科重跑一次 split）
    test_all = load_dataset("cais/mmlu", "all", split="test")
    dev_all = load_dataset("cais/mmlu", "all", split="dev")
    dev_by: dict[str, list] = {}
    for r in dev_all:
        dev_by.setdefault(r["subject"], []).append(r)
    test_by: dict[str, list] = {}
    for r in test_all:
        test_by.setdefault(r["subject"], []).append(r)

    prompts: list[str] = []
    gold: list[int] = []
    subs: list[str] = []
    for subject in MMLU_SUBJECTS:
        shots = dev_by.get(subject, [])[:5]
        rows = test_by.get(subject, [])[:n_per_subject]
        if not rows:
            print(f"  [warn] {subject}: 無題目，略過")
            continue
        for row in rows:
            prompts.append(mmlu_messages(tok, subject, shots, row))
            gold.append(row["answer"])
            subs.append(subject)

    outs = C.generate_batch(
        model, tok, prompts, max_new_tokens=2, batch_size=batch_size, greedy=True, label=f"{label}:mmlu"
    )
    for i in range(min(debug, len(outs))):
        print(f"  [debug] {subs[i]} gold={LETTERS[gold[i]]} raw={outs[i]!r}")

    per_subject: dict[str, dict] = {}
    correct = 0
    unparsed = 0
    for s, g, out in zip(subs, gold, outs):
        p = parse_letter(out)
        unparsed += int(p < 0)
        ok = int(p == g)
        correct += ok
        d = per_subject.setdefault(s, {"n": 0, "correct": 0})
        d["n"] += 1
        d["correct"] += ok
    for d in per_subject.values():
        d["acc"] = d["correct"] / d["n"]

    n = len(outs)
    return {
        "overall_acc": correct / n if n else None,
        "n": n,
        "unparsed": unparsed,
        "per_subject": per_subject,
        "samples": [{"subject": s, "gold": LETTERS[g], "raw": o} for s, g, o in zip(subs[:20], gold[:20], outs[:20])],
    }





NUM_RE = re.compile(r"####\s*\$?\s*(-?[\d,]+(?:\.\d+)?)")
ANY_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_number(text: str) -> str | None:
    m = NUM_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    nums = ANY_NUM_RE.findall(text)
    return nums[-1].replace(",", "") if nums else None


def gsm8k_eval(model, tok, limit: int, batch_size: int, max_new_tokens: int, label: str) -> dict:
    """0-shot greedy。Abiliteration 的損害在「多步推理」上最明顯，
    所以這裡比的是**同一個模型改動前後**的相對差距，不是絕對 benchmark 分數。"""
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = [ds[i] for i in range(min(limit, len(ds)))]
    system = (
        "Solve the math problem. Show brief reasoning, then on the final line write the "
        "answer exactly as: #### <number>"
    )
    chat = [C.build_prompt(tok, r["question"], system=system) for r in rows]
    outs = C.generate_batch(
        model, tok, chat, max_new_tokens=max_new_tokens, batch_size=batch_size, greedy=True, label=f"{label}:gsm8k"
    )
    graded = []
    for r, out in zip(rows, outs):
        gold = extract_number(r["answer"])
        pred = extract_number(out)
        graded.append(
            {
                "question": r["question"],
                "gold": gold,
                "pred": pred,
                "correct": pred is not None and gold is not None and pred == gold,
                "response": out,
            }
        )
    n = len(graded)
    return {
        "n": n,
        "exact_match": sum(g["correct"] for g in graded) / n if n else None,
        "unparsed": sum(1 for g in graded if g["pred"] is None),
        "rows": graded,
    }


@torch.inference_mode()
def ppl_eval(model, windows: list[list[int]], label: str) -> float:
    nll, nt = 0.0, 0
    for i, w in enumerate(windows):
        ids = torch.tensor([w], device=model.device)
        lp = torch.log_softmax(model(ids).logits[0, :-1, :].float(), dim=-1)
        nll += float(-lp.gather(1, torch.tensor(w[1:], device=model.device)[:, None]).sum())
        nt += lp.shape[0]
        print(f"\r  [{label}] ppl {i + 1}/{len(windows)}", end="", flush=True)
    print()
    return float(torch.exp(torch.tensor(nll / nt)))


def ppl_windows(tok, max_windows: int, seq_len: int) -> list[list[int]]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, add_special_tokens=False)["input_ids"]
    out = []
    for start in range(0, len(ids) - seq_len, seq_len):
        out.append(ids[start : start + seq_len])
        if len(out) >= max_windows:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "heretic", "both"], default="both")
    ap.add_argument("--tasks", default="mmlu,gsm8k,ppl")
    ap.add_argument("--dtype", default="bfloat16", choices=list(C.DTYPES))
    ap.add_argument("--limit-mmlu", type=int, default=20, help="每科題數")
    ap.add_argument("--limit-gsm8k", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gsm8k-max-tokens", type=int, default=400)
    ap.add_argument("--ppl-windows", type=int, default=30)
    ap.add_argument("--debug-mmlu", type=int, default=0, help="印出前 N 題的 top-5 token（人工檢查用）")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    targets = ["base", "heretic"] if args.model == "both" else [args.model]

    for which in targets:
        print(f"\n===== {which} ({C.MODELS[which][0]}) =====")
        tok = C.load_tokenizer(which)
        model = C.load_model(which, dtype=args.dtype)
        print(f"  device: {C.gpu_report()}")
        result: dict = {"model": which, "repo": C.MODELS[which][0], "settings": vars(args)}

        if "mmlu" in tasks:
            result["mmlu"] = mmlu_eval(
                model, tok, args.limit_mmlu, which, batch_size=args.batch_size, debug=args.debug_mmlu
            )
            print(f"  MMLU acc = {result['mmlu']['overall_acc']:.4f} "
                  f"(n={result['mmlu']['n']}, unparsed={result['mmlu']['unparsed']})")

        if "gsm8k" in tasks:
            result["gsm8k"] = gsm8k_eval(
                model, tok, args.limit_gsm8k, args.batch_size, args.gsm8k_max_tokens, which
            )
            print(f"  GSM8K EM = {result['gsm8k']['exact_match']:.4f} (n={result['gsm8k']['n']}, "
                  f"unparsed={result['gsm8k']['unparsed']})")

        if "ppl" in tasks:
            result["ppl"] = ppl_eval(model, ppl_windows(tok, args.ppl_windows, 512), which)
            print(f"  PPL = {result['ppl']:.4f}")

        C.free_model(model)
        del model
        out = C.results_dir() / f"results_capability_{which}.json"
        C.save_json(out, result)
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

