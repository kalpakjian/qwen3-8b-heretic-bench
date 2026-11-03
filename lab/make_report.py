#!/usr/bin/env python3
"""
Phase 4：把三個階段的 JSON 彙整成報告。

刻意只讀既有 JSON、不重算任何數字 —— 報告與原始結果永遠對得上，
而且可以單獨重跑（跑完 Phase 1 就想先看拒答表也行）。

輸出：
  --lang zh（預設）→ results/report.md      中文
  --lang en         → results/report_EN.md  英文

所有文字都在 `report_i18n.py`，這個檔案只負責「取數字 → 套模板」，
所以兩種語言的**結構保證一致**（不會出現中文有某節、英文漏掉）。

用法：
  python make_report.py              # 中文
  python make_report.py --lang en    # 英文
  python make_report.py --lang en --out /tmp/r.md
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import common as C
from report_i18n import STRINGS

# 由 main() 依 --lang 覆寫；各 section 函式共用，讓簽章維持簡單。
L: dict[str, str] = STRINGS["zh"]


def pct(x, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def delta_pt(new, old, digits: int = 1) -> str:
    if new is None or old is None:
        return "—"
    return f"{(new - old) * 100:+.{digits}f} pt"


def num(x, fmt: str = "{:.3f}") -> str:
    return "—" if x is None else fmt.format(x)


def load_optional(name: str):
    p = C.results_dir() / name
    return C.load_json(p) if p.exists() else None


def overview_section(ref: dict, cap: dict, kl: dict, judged: dict | None = None) -> tuple[str, dict]:
    sb = (ref.get("base") or {}).get("summary", {})
    sh = (ref.get("heretic") or {}).get("summary", {})
    cb = cap.get("base") or {}
    ch = cap.get("heretic") or {}
    judged = judged or {}

    m = {
        "tpr_b": sb.get("tpr_high_risk_refusal"),
        "tpr_h": sh.get("tpr_high_risk_refusal"),
        "fpr_b": sb.get("fpr_benign_over_refusal"),
        "fpr_h": sh.get("fpr_benign_over_refusal"),
        "ba_b": sb.get("balanced_accuracy"),
        "ba_h": sh.get("balanced_accuracy"),
        "judge_b": ((judged.get("base") or {}).get("summary") or {}).get("judge_tpr_refusal_rate"),
        "judge_h": ((judged.get("heretic") or {}).get("summary") or {}).get("judge_tpr_refusal_rate"),
        "mmlu_b": (cb.get("mmlu") or {}).get("overall_acc"),
        "mmlu_h": (ch.get("mmlu") or {}).get("overall_acc"),
        "gsm_b": (cb.get("gsm8k") or {}).get("exact_match"),
        "gsm_h": (ch.get("gsm8k") or {}).get("exact_match"),
        "ppl_b": cb.get("ppl"),
        "ppl_h": ch.get("ppl"),
    }
    klt = (kl or {}).get("KL_text") or {}
    klf = (kl or {}).get("KL_first_token") or {}
    m["kl"] = klt.get("kl_mean")

    e, c = L["axis_effect"], L["axis_cost"]
    rows = [
        f"| {e} | {L['m_keyword']} | {pct(m['tpr_b'])} | {pct(m['tpr_h'])} | **{delta_pt(m['tpr_h'], m['tpr_b'])}** |",
        f"| {e} | {L['m_judge']} | {pct(m['judge_b'])} | {pct(m['judge_h'])} | **{delta_pt(m['judge_h'], m['judge_b'])}** |",
        f"| {e} | {L['m_fpr']} | {pct(m['fpr_b'])} | {pct(m['fpr_h'])} | {delta_pt(m['fpr_h'], m['fpr_b'])} |",
        f"| {c} | {L['m_kl']} | 0 | **{num(m['kl'], '{:.5f}')}** | — |",
        f"| {c} | {L['m_top1']} | 100.0% | {pct(klt.get('top1_agreement'))} | — |",
        f"| {c} | {L['m_mmlu']} | {pct(m['mmlu_b'])} | {pct(m['mmlu_h'])} | {delta_pt(m['mmlu_h'], m['mmlu_b'])} |",
        f"| {c} | {L['m_gsm']} | {pct(m['gsm_b'])} | {pct(m['gsm_h'])} | {delta_pt(m['gsm_h'], m['gsm_b'])} |",
        f"| {c} | {L['m_ppl']} | {num(m['ppl_b'])} | {num(m['ppl_h'])} | "
        f"{('{:+.2f}%'.format((m['ppl_h'] / m['ppl_b'] - 1) * 100)) if (m['ppl_b'] and m['ppl_h']) else '—'} |",
    ]
    lines = [L["h1"], "", L["hdr_overview"], "|---|---|---|---|---|", *rows, ""]
    if m["tpr_b"] is not None and m["judge_b"] is not None:
        lines += [L["insight"].format(
            tpr_b=pct(m["tpr_b"]),
            tpr_h=pct(m["tpr_h"]),
            d_keyword=delta_pt(m["tpr_h"], m["tpr_b"]),
            judge_b=pct(m["judge_b"]),
            judge_h=pct(m["judge_h"]),
        ), ""]
    if klf:
        lines += [L["klf_note"].format(v=num(klf.get("kl_mean"), "{:.5f}")), ""]
    return "\n".join(lines), m


def category_table(ref: dict) -> str:
    cats: dict[str, dict] = {}
    for which in ("base", "heretic"):
        if which not in ref:
            continue
        for cat, c in ref[which]["summary"]["per_category"].items():
            cats.setdefault(cat, {"should_refuse": c["should_refuse"]})
            cats[cat][which] = c["rate"]
            cats[cat][f"{which}_frac"] = f"{c['refused']}/{c['n']}"
    lines = [L["h2"], "", L["hdr_category"], "|---|---|---|---|---|"]
    for cat, c in sorted(cats.items()):
        kind = L["type_refuse"] if c["should_refuse"] else L["type_benign"]
        rb, rh = c.get("base"), c.get("heretic")
        note = ""
        if c["should_refuse"] and rb is not None and rh is not None and rb > 0.5 and rh < 0.2:
            note = L["note_failed"]
        elif not c["should_refuse"] and rh is not None and rh > 0.1:
            note = L["note_over"]
        lines.append(
            f"| `{cat}` | {kind} | {pct(rb)} ({c.get('base_frac', '')}) | "
            f"{pct(rh)} ({c.get('heretic_frac', '')}) | {note} |"
        )
    lines.append("")
    return "\n".join(lines)


def judge_section(judged: dict) -> str:
    if not judged:
        return L["h2b_notrun"]
    lines = [L["h2b"], "", L["hdr_judge"], "|---|---|---|---|---|---|"]
    for which in ("base", "heretic"):
        d = judged.get(which)
        if not d:
            continue
        s = d["summary"]
        lines.append(
            f"| {which} | {pct(s.get('keyword_tpr_same_subset'))} | {pct(s.get('judge_tpr_refusal_rate'))} | "
            f"{s.get('comply')} | {s.get('partial')} | "
            f"{delta_pt(s.get('judge_tpr_refusal_rate'), s.get('keyword_tpr_same_subset'))} |"
        )
    lines += ["", L["judge_explain"], ""]
    return "\n".join(lines)


def weight_section(wa: dict | None) -> str:
    if not wa:
        return ""
    rows = [r for r in wa["changed"] if r.get("changed")]
    top = sorted(rows, key=lambda r: r["changed_frac"], reverse=True)[:10]
    lines = [
        L["h3b"],
        "",
        L["wa_total"].format(n=wa["n_tensors"]),
        L["wa_changed"].format(k=wa["n_tensors_changed"], frac=wa["frac_tensors_changed"]),
        "",
        L["hdr_weight"],
        "|---|---|---|",
    ]
    for r in top:
        lines.append(f"| `{r['tensor']}` | {r['changed_frac']:.1%} | {r['max_abs_diff']:.5f} |")
    lines += ["", L["wa_note"], ""]
    return "\n".join(lines)


def kl_section(kl: dict | None) -> str:
    if not kl:
        return L["h3_notrun"]

    t, f = kl["KL_text"], kl["KL_first_token"]
    lines = [
        L["h3"],
        "",
        L["kl_samples"].format(mw=kl["settings"]["max_windows"], sl=kl["settings"]["seq_len"], np_=t["n_positions"]),
        L["kl_mean"].format(
            mean=t["kl_mean"], med=t["kl_median"], p95=t["kl_p95"], mx=t.get("kl_max", float("nan"))
        ),
        L["kl_top1"].format(agr=pct(t["top1_agreement"]), tv=t["tv_distance_mean"]),
        L["kl_firsttok"].format(v=f["kl_mean"], n=f["n_positions"]),
        "",
    ]
    ppl = kl.get("perplexity") or {}
    if ppl.get("base") and ppl.get("heretic"):
        d = (ppl["heretic"] / ppl["base"] - 1) * 100
        lines += [
            L["kl_ppl"].format(b=ppl["base"], h=ppl["heretic"], d=d),
            L["kl_ppl_warn"],
            "",
        ]
    return "\n".join(lines)


def verdict_section(m: dict) -> str:
    lines = [L["h4"], ""]
    if m.get("tpr_b") is not None and m.get("tpr_h") is not None:
        lines.append(L["v_refusal"].format(
            a=pct(m["tpr_b"]), b=pct(m["tpr_h"]), d=delta_pt(m["tpr_h"], m["tpr_b"])
        ))
    if m.get("kl") is not None:
        lines.append(L["v_kl"].format(v=m["kl"]))
    for key, name in (("mmlu", "MMLU"), ("gsm", "GSM8K")):
        b, h = m.get(f"{key}_b"), m.get(f"{key}_h")
        if b is not None and h is not None:
            lines.append(L["v_task"].format(name=name, a=pct(b), b=pct(h), d=delta_pt(h, b)))
    b, h = m.get("ppl_b"), m.get("ppl_h")
    if b and h:
        lines.append(L["v_ppl"].format(a=b, b=h, d=(h / b - 1) * 100))
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    global L
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=sorted(STRINGS), default="zh", help="報告語言（預設 zh）")
    ap.add_argument("--out", help="輸出路徑（預設 results/report.md 或 results/report_EN.md）")
    args = ap.parse_args()
    L = STRINGS[args.lang]

    ref = {}
    for which in ("base", "heretic"):
        d = load_optional(f"results_refusal_{which}.json")
        if d:
            ref[which] = d
    cap = {
        "base": load_optional("results_capability_base.json"),
        "heretic": load_optional("results_capability_heretic.json"),
    }
    judged = {}
    for which in ("base", "heretic"):
        d = load_optional(f"results_refusal_{which}_judged.json")
        if d:
            judged[which] = d
    kl = load_optional("results_kl.json")
    wa = load_optional("results_weight_audit.json")

    if not ref:
        sys.exit(L["fail"])

    overview, m = overview_section(ref, cap, kl, judged)
    s = ref.get("heretic") or ref.get("base")
    header = [
        L["title"],
        "",
        f"- {L['generated']}{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        L["base_line"].format(repo=C.BASE_REPO),
        L["heretic_line"].format(repo=C.HERETIC_REPO),
        L["settings"].format(n=s["settings"].get("max_new_tokens")),
        "",
    ]
    parts = [
        *header,
        overview,
        category_table(ref),
        judge_section(judged),
        kl_section(kl),
        weight_section(wa),
        verdict_section(m),
        L["limits"],
    ]
    body = "\n".join(p for p in parts if p)

    if args.out:
        out = Path(args.out)
    else:
        name = "report.md" if args.lang == "zh" else f"report_{args.lang.upper()}.md"
        out = C.results_dir() / name
    out.write_text(body, encoding="utf-8")
    print(body)
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

