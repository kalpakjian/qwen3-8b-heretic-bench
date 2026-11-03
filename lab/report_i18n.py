#!/usr/bin/env python3
"""
`make_report.py` 的雙語字串表。

為什麼要獨立一個檔案
--------------------
報告的**數字**全部來自 JSON，只有**文字**需要在語言間切換。把文字集中在這裡，
`make_report.py` 就只剩「取數字 → 套模板」的邏輯，兩種語言的結構保證一致
（不會出現中文有某一節、英文漏掉的情況）。

新增語言：在 `STRINGS` 加一個 key，鍵名與現有語言完全相同即可，
`selftest.py` 會檢查所有語言的鍵集合是否一致。
"""

STRINGS: dict[str, dict[str, str]] = {
    # ------------------------------------------------------------------ 中文 --
    "zh": {
        # 頁首
        "title": "# Qwen3-8B-Heretic 基準測試報告",
        "generated": "產生時間：",
        "base_line": "- base：`{repo}`",
        "heretic_line": "- heretic：`{repo}`（Heretic trial 131/200，bf16 全精度 safetensors）",
        "settings": "- 設定：greedy、thinking off、max_new_tokens={n}",
        # 第 1 節
        "h1": "## 1. 總覽：效果 vs 代價",
        "hdr_overview": "| 軸 | 指標 | base `Qwen/Qwen3-8B` | heretic | 變化 |",
        "axis_effect": "效果",
        "axis_cost": "代價",
        "m_keyword": "顯式拒答語句率（keyword）",
        "m_judge": "**實質拒絕率（LLM-judge）**",
        "m_fpr": "良性誤拒率 FPR（越低越好）",
        "m_kl": "全序列 KL（heretic‖base）",
        "m_top1": "top-1 一致率",
        "m_mmlu": "MMLU（16 科 × 20 題）",
        "m_gsm": "GSM8K exact match（100 題）",
        "m_ppl": "wikitext-2 PPL",
        "insight": (
            "> **兩種拒答判定差很多**：keyword 由 {tpr_b} 降到 {tpr_h}（{d_keyword}），"
            "但 judge 幾乎沒動（{judge_b} → {judge_h}）。意思是這次去對齊主要改掉了"
            "**拒絕的說法**，而沒有改掉拒絕的**行為**。"
        ),
        "klf_note": (
            "> 複刻模型卡協定（第一個 token KL，50 個良性 prompt）：**{v}**"
            "（模型卡自稱 0.0323，樣本為 20 個 prompt）"
        ),
        # 第 2 節
        "h2": "## 2. 拒答率（逐類別，100 題題庫）",
        "hdr_category": "| 類別 | 類型 | base | heretic | 說明 |",
        "type_refuse": "應拒答",
        "type_benign": "良性",
        "note_failed": "對齊完全失效",
        "note_over": "過度拒絕（不該發生）",
        # 第 2b 節
        "h2b": "## 2b. 兩種判定法的落差（為什麼不能只看 keyword）",
        "h2b_notrun": "## 2b. Judge 複判\n\n（尚未執行 `judge_rescore.py`）\n",
        "hdr_judge": "| 模型 | keyword 拒答率 | judge 拒答率 | COMPLY | PARTIAL | 落差 |",
        "judge_explain": (
            "keyword 只認「拒絕語句」；judge 問的是「有沒有真的提供有害內容」。\n"
            "兩者差距就是**『換了說法但行為沒變』**的比例 —— 人工抽查已確認 judge 的判定與原文一致"
            "（例如「怎麼製作炸彈？」的回覆整段在講風險與替代方案，沒有給出製作方法）。"
        ),
        # 第 3 節
        "h3": "## 3. KL 明細（全序列、全詞表）",
        "h3_notrun": "## 3. KL 明細\n\n（尚未執行 `kl_eval.py`）\n",
        "kl_samples": "- 樣本：{mw} 窗 × {sl} tokens（{np_} 個位置，每個位置都比對 151936 維分布）",
        "kl_mean": "- 平均 KL：**{mean:.5f}**　中位數 {med:.5f}　p95 {p95:.5f}　最大 {mx:.5f}",
        "kl_top1": "- top-1 一致率 {agr}；平均 TV 距離 {tv:.5f}",
        "kl_firsttok": "- 第一個 token KL（複刻模型卡協定）：**{v:.5f}**（n={n}）",
        "kl_ppl": (
            "- perplexity（**本節 20×256 窗，僅供 KL 對照**）：base {b:.3f} → "
            "heretic {h:.3f}（{d:+.2f}%）"
        ),
        "kl_ppl_warn": (
            "  ⚠️ 這個數字與第 1 節的 PPL 不同，是因為窗長與窗數不同（30×512）。"
            "要看「能力有沒有掉」請用第 1 節。"
        ),
        # 第 3b 節
        "h3b": "## 3b. 權重稽核（證明模型真的被改過）",
        "wa_total": "- 張量總數 **{n}**，名稱集合完全相同（純權重改造，未增刪結構）",
        "wa_changed": "- 有改動的張量 **{k}**（{frac:.1%}） → 這是外科手術式的低秩改造，不是重新訓練",
        "hdr_weight": "| 張量 | 改動比例 | max‖Δ‖ |",
        "wa_note": (
            "> 為什麼要驗這個：KL 只有 0.005、MMLU 完全沒動，很容易被誤解成「根本載到同一個模型」。\n"
            "> 直接比對 safetensors 後可確認：絕大多數張量位元相同，改動集中在少數層的 "
            "`down_proj`/`o_proj` —— 與 abliteration 的原理一致。"
        ),
        # 第 4 節
        "h4": "## 4. 判讀",
        "v_refusal": "- 拒答：{a} → {b}（{d}）。",
        "v_kl": "- 全序列 KL {v:.5f}；Heretic 自己的門檻是 <0.05 = minimal damage、<0.5 = low damage。",
        "v_task": "- {name}：{a} → {b}（{d}）。",
        "v_ppl": "- PPL：{a:.3f} → {b:.3f}（{d:+.2f}%）。",
        "limits": """
## 5. 方法與限制（讀數字前先看）

1. **單次 greedy 快照**：拒答判定用 greedy + 關鍵字（與 `llm-refusal-audit` 同一套），
   不做多次取樣。真實拒答率會隨取樣溫度上升，這裡的數字偏**下界**。
2. **關鍵字判定會誤判**：`抱歉`、`policy` 之類字串可能出現在非拒答回覆中。
   已扣掉 `sorry to hear` 這類同情語境，但無法完全消除；要更準請接 LLM-judge
   （`llm-refusal-audit` 實測 judge 會把 TPR 上調 1–8 pt）。
3. **KL 是「分布距離」不是「能力損失」**：KL 小不代表能力沒掉，反之亦然，
   所以第 1 節刻意把兩軸並列。abliteration 造成的 KL 集中在「拒絕方向」相關 token，
   平均值會被大量無關 token 稀釋 —— 這正是只看平均值會低估損害的原因。
4. **MMLU / GSM8K 是抽樣**：每科 20 題、GSM8K 100 題，標準誤約 ±2–5 pt，
   小於此幅度的差距不要當真；要更嚴謹請加大 `--limit-mmlu` / `--limit-gsm8k`。
5. **GSM8K 用 0-shot**：不是標準 8-shot，絕對分數會低於公開排行；
   但 base 與 heretic 協定完全相同，**相對差距**才是這裡要看的。
6. **本 repo 不重跑 Heretic**：量測的是已產出的權重，不是重新搜尋 ablation 參數。
""",
        "fail": "[fail] 找不到 results_refusal_*.json，請先跑 refusal_eval.py",
    },
    # ---------------------------------------------------------------- English --
    "en": {
        "title": "# Qwen3-8B-Heretic Benchmark Report",
        "generated": "Generated: ",
        "base_line": "- base: `{repo}`",
        "heretic_line": "- heretic: `{repo}` (Heretic trial 131/200, bf16 full-precision safetensors)",
        "settings": "- Settings: greedy, thinking off, max_new_tokens={n}",
        "h1": "## 1. Overview: effect vs cost",
        "hdr_overview": "| Axis | Metric | base `Qwen/Qwen3-8B` | heretic | Delta |",
        "axis_effect": "effect",
        "axis_cost": "cost",
        "m_keyword": "Explicit refusal phrasing (keyword)",
        "m_judge": "**Substantive refusal (LLM-judge)**",
        "m_fpr": "Over-refusal on benign prompts (FPR, lower is better)",
        "m_kl": "Full-sequence KL (heretic‖base)",
        "m_top1": "top-1 token agreement",
        "m_mmlu": "MMLU (16 subjects × 20 questions)",
        "m_gsm": "GSM8K exact match (100 questions)",
        "m_ppl": "wikitext-2 perplexity",
        "insight": (
            "> **The two refusal metrics disagree sharply**: keyword drops from {tpr_b} to "
            "{tpr_h} ({d_keyword}), while the judge barely moves ({judge_b} → {judge_h}). "
            "This abliteration removed **the wording of refusal**, not the **behaviour**."
        ),
        "klf_note": (
            "> Model-card protocol replicated (first-token KL, 50 benign prompts): **{v}**"
            " (the card claims 0.0323 on 20 prompts)"
        ),
        "h2": "## 2. Refusal rate by category (100-prompt suite)",
        "hdr_category": "| Category | Kind | base | heretic | Note |",
        "type_refuse": "should refuse",
        "type_benign": "benign",
        "note_failed": "alignment fully lost",
        "note_over": "over-refusal (should not happen)",
        "h2b": "## 2b. Keyword vs judge (why one metric is not enough)",
        "h2b_notrun": "## 2b. Judge re-scoring\n\n(`judge_rescore.py` has not been run)\n",
        "hdr_judge": "| Model | keyword refusal | judge refusal | COMPLY | PARTIAL | Gap |",
        "judge_explain": (
            "Keyword matching only sees refusal *phrasing*; the judge is asked whether the "
            "harmful content was actually provided.\nThe gap between them is the share of "
            "**\"same behaviour, different wording\"** — hand-checked against the raw "
            "responses (e.g. the bomb-making reply discusses risk and alternatives at length "
            "without ever giving instructions)."
        ),
        "h3": "## 3. KL detail (full sequence, full vocabulary)",
        "h3_notrun": "## 3. KL detail\n\n(`kl_eval.py` has not been run)\n",
        "kl_samples": "- Sample: {mw} windows × {sl} tokens ({np_} positions, each compared over a 151,936-dim vocabulary)",
        "kl_mean": "- KL mean: **{mean:.5f}**  median {med:.5f}  p95 {p95:.5f}  max {mx:.5f}",
        "kl_top1": "- top-1 agreement {agr}; mean total-variation distance {tv:.5f}",
        "kl_firsttok": "- First-token KL (model-card protocol): **{v:.5f}** (n={n})",
        "kl_ppl": "- perplexity (**this section's 20×256 windows, for KL reference only**): base {b:.3f} → heretic {h:.3f} ({d:+.2f}%)",
        "kl_ppl_warn": (
            "  ⚠️ This differs from the PPL in section 1 because the windows differ (30×512). "
            "Use section 1 to judge capability retention."
        ),
        "h3b": "## 3b. Weight audit (proof the checkpoint was really modified)",
        "wa_total": "- Tensors: **{n}**, identical name sets (pure weight modification, no structural change)",
        "wa_changed": "- Tensors changed: **{k}** ({frac:.1%}) → a surgical low-rank edit, not retraining",
        "hdr_weight": "| Tensor | Changed | max‖Δ‖ |",
        "wa_note": (
            "> Why this check exists: with KL at 0.005 and MMLU unmoved, it is easy to suspect "
            "the same weights were loaded twice.\n> Comparing the safetensors directly "
            "confirms the opposite: almost every tensor is bit-identical, and the edits "
            "concentrate in a few layers' `down_proj`/`o_proj` — which is what abliteration "
            "is supposed to do."
        ),
        "h4": "## 4. Interpretation",
        "v_refusal": "- Refusal: {a} → {b} ({d}).",
        "v_kl": "- Full-sequence KL {v:.5f}; Heretic's own thresholds are <0.05 = minimal damage, <0.5 = low damage.",
        "v_task": "- {name}: {a} → {b} ({d}).",
        "v_ppl": "- PPL: {a:.3f} → {b:.3f} ({d:+.2f}%).",
        "limits": """
## 5. Method and limitations (read before quoting a number)

1. **Single greedy snapshot**: refusal verdicts use greedy decoding plus keyword matching
   (the same list as `llm-refusal-audit`), with no repeated sampling. Real refusal rates
   rise with temperature, so these numbers sit closer to a **lower bound**.
2. **Keyword matching misfires**: strings like `sorry` or `policy` can appear in replies
   that are not refusals. Sympathetic contexts such as `sorry to hear` are subtracted, but
   the problem cannot be fully removed; use the LLM judge for anything load-bearing
   (`llm-refusal-audit` measured the judge raising TPR by 1-8 pt).
3. **KL is distribution distance, not capability loss**: a small KL does not prove
   capability survived, and vice versa — which is why section 1 puts both axes side by
   side. Abliteration's KL concentrates on refusal-direction tokens and is diluted by the
   many unrelated ones, so a mean alone understates the damage.
4. **MMLU / GSM8K are sampled**: 20 per subject and 100 questions respectively, standard
   error roughly ±2-5 pt. Treat smaller gaps as noise; raise `--limit-mmlu` /
   `--limit-gsm8k` to tighten.
5. **GSM8K is 0-shot**, not the standard 8-shot, so absolute scores sit below published
   leaderboards. Both models use the identical protocol, so the **relative** gap is the
   meaningful part.
6. **This repo does not re-run Heretic**: it measures the exported weights; it does not
   search for new ablation parameters.
""",
        "fail": "[fail] no results_refusal_*.json found; run refusal_eval.py first",
    },
}

