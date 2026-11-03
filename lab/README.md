# Qwen3-8B-Heretic Bench — 實驗腳本（Omarchy / RTX 4090）

量測 `kalpakjian/qwen3-8b-heretic`（abliterated）相對 `Qwen/Qwen3-8B`（base）的
**效果**與**代價**：

1. **效果**：拒答率下降多少（100 題題庫，TPR / FPR）
2. **代價**：分布被改動多少（全序列 KL、top-1 一致率）
3. **代價**：能力掉了多少（MMLU / GSM8K / perplexity）

> 對應筆記：`../notes/heretic-method-notes.md`（為什麼要量這四軸、循環性風險、記憶體數學）

---

## 0. 前置條件

| 項目 | 需求 | 檢查方式 |
|---|---|---|
| OS | Linux（本 repo 以 Omarchy / Arch 為主） | `grep PRETTY_NAME /etc/os-release` |
| GPU | 24 GB VRAM（8B bf16 = 16.4 GB） | `nvidia-smi` |
| 磁碟 | ≥ 45 GB（2×16.4 GB 模型 + 6 GB venv + 3 GB KL 快取） | `df -h ~` |
| 記憶體 | 32 GB 以上（KL 階段 CPU 端要放 608 KB × 位置數的陣列） | `free -g` |
| Python | **3.12**（用 `uv venv --python 3.12` 隔離） | `uv --version` |

**不需要** Ollama、llama.cpp、vLLM。全部用 transformers 跑 bf16。

---

## 1. 安裝與執行

```bash
git clone https://github.com/kalpakjian/qwen3-8b-heretic-bench.git
cd qwen3-8b-heretic-bench/lab

bash setup.sh          # 建 venv + 下載兩個模型 + 預抓資料集（首次很久）
bash run_all.sh        # 一鍵跑完 Phase 0–4
```

只想先確認流程會動（約 10 分鐘，數字沒有統計意義）：

```bash
SMOKE=1 bash run_all.sh
```

---

## 2. 階段

| Phase | 腳本 | 量什麼 | 主要參數 |
|---|---|---|---|
| 0 | `selftest.py` | 判定邏輯與題目格式（不需 GPU，27 項檢查） | — |
| 0b | `weight_audit.py` | 權重差集（純 CPU）：證明模型真的被改過、改在哪 | `--top 25` |
| 1 | `refusal_eval.py` | 拒答 TPR／良性 FPR／逐類別 | `--max-new-tokens 256`、`--batch-size 8` |
| 1b | `judge_rescore.py` | LLM-judge 複判（修正關鍵字法的雙向失準） | `--judge-model base` |
| 2 | `kl_eval.py` | 全序列 KL、top-1 一致率、第一個 token KL、PPL | `--max-windows 20`、`--seq-len 256` |
| 3 | `capability_eval.py` | MMLU（16 科抽樣）、GSM8K、PPL | `--limit-mmlu 20`、`--limit-gsm8k 100` |
| 4 | `make_report.py` | 彙整成 `../results/report.md` | 只讀 JSON、不重算 |

報告有中英兩版（同一份 JSON、同一套結構，文字在 `report_i18n.py`）：

```bash
python make_report.py              # results/report.md      中文
python make_report.py --lang en    # results/report_EN.md   英文
```

單獨重跑某個階段：

```bash
python refusal_eval.py --model heretic --limit 10      # 只跑 heretic、前 10 題
python judge_rescore.py --model heretic                # 只重判 heretic
python inspect_refusals.py --model heretic             # 人工稽核：哪個關鍵字命中
python kl_eval.py --max-windows 5 --seq-len 128
python capability_eval.py --tasks mmlu --limit-mmlu 5 --debug-mmlu 3
python weight_audit.py --top 10
python make_report.py
```

**重要**：`kl_eval.py` 第一次跑會把 base 的 log-prob 存到 `../results/_cache/`
（約 3 GB）。之後重跑會自動沿用；要強制重算加 `--recompute`。

---

## 3. 檔案清單

| 檔案 | 用途 |
|---|---|
| `setup.sh` | 冪等環境建置：uv → py3.12 venv → torch/transformers → 下載模型 → 預抓資料集 |
| `run_all.sh` | 一鍵執行 Phase 0–4；`SMOKE=1` 縮小樣本 |
| `env_check.py` | 版本、CUDA、bf16 matmul、可用 VRAM 檢查 |
| `download_models.py` | 下載 base（根目錄）與 heretic（`model_files/` 子目錄），排除用不到的 .pth/onnx/gguf |
| `common.py` | 模型載入、chat template、批次生成、拒答關鍵字、結果 IO |
| `extract_prompts.py` | 從 `llm-refusal-audit` 上游抽出 100 題 → `prompts_refusal.json` |
| `prompts_refusal.json` | 100 題（50 應拒答 / 50 良性），11 個類別 |
| `refusal_eval.py` | Phase 1（keyword） |
| `judge_rescore.py` | Phase 1b（LLM-judge 複判） |
| `inspect_refusals.py` | 人工稽核：列出每題命中的關鍵字與原始回覆 |
| `kl_eval.py` | Phase 2（兩段式） |
| `capability_eval.py` | Phase 3 |
| `weight_audit.py` | Phase 0b（safetensors 張量比對） |
| `make_report.py` | Phase 4 |
| `selftest.py` | Phase 0 |

---

## 4. 常見錯誤處理（都是實際踩過的）

| 症狀 | 原因 | 處理 |
|---|---|---|
| `TypeError: expected str ... not NoneType`（`os.path.join`） | 對 base 傳了 `subfolder=None`，transformers 5.x 直接炸 | 只在有子目錄時才傳 `subfolder`（`common.py` 已處理） |
| **MMLU 準確率剛好 25%** | `cais/mmlu` 的 `question` **不含選項**（在 `choices` 清單），沒接上去＝盲猜 | 用 `format_question()` 產生 `A. … B. …`；`selftest` 有回歸測試 |
| MMLU 全部答 `A` | 5-shot 全塞在單一 user message → 模型進入「續寫文件」模式（top-5 是 `Let/We/Here/###`） | 改用**多輪** assistant 輪次（`mmlu_messages()`） |
| 命中關鍵字但其實是照做 | 「當然可以…**不建議**你…」——警告語被當成拒答 | 跑 `judge_rescore.py`；並用 `inspect_refusals.py` 抽查 |
| `parse_letter('The answer is D')` 回 A | `answer` 裡的 'a' 被當成答案字母 | 已改字詞邊界比對；`selftest` 有回歸測試 |
| 懷疑「是不是載到同一個模型」 | abliteration 只改 6% 張量 → KL 很小、MMLU 一模一樣 | 跑 `weight_audit.py` 直接比張量（勿用單一層 q_proj 當指紋：它本來就不會被改） |
| `CUDA out of memory` 換模型時 | 前一個模型沒真的釋放：`free_model()` 只刪函式內的區域變數 | 呼叫端也要 `del model`，或用 `judge_rescore.py` 那種單模型流程 |
| `torch.cuda.is_available() == False` | 裝到 CPU-only wheel | `uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu130` |
| GSM8K `unparsed` 很多 | 模型沒照 `#### <數字>` 格式 | 已退回「最後一個數字」；仍偏高請看 `results_capability_*.json` 的 `rows` |
| 拒答率看起來「太好」（0–4%） | 關鍵字法與 Heretic 目標函數同構（循環性） | 讀 `../notes/heretic-method-notes.md` §2，並以 judge 結果為準 |
| `_cache/` 佔 3 GB | KL 階段暫存 | 可安全刪除，下次會重算 |

---

## 5. 設計說明（為什麼這樣量）

**為什麼 greedy？** 拒答與否在取樣下是隨機事件；單次快照要可重現，就得把溫度關掉。
模型卡的 T=0.6 設定適合「看回答品質」，不適合「數拒答比例」。

**為什麼關 thinking？** Qwen3 的 `enable_thinking=True` 會先產生一段 ` thinking`，
這段內容既吃 token 預算，也會被關鍵字法誤判為拒答（思考裡常有「我不應該…」）。
關掉後評估點落在真正的回覆上，與 Heretic 自己跳過 CoT 區塊的做法一致。

**為什麼 MMLU 要生成而不是比 logit？** 兩種都試過：比 A/B/C/D 的 log-prob 在
「模型根本不想回答字母」時會得到無意義的結果（實測 top-5 是 `Let/We/Here/###`）。
改成 5-shot 多輪 + 生成 2 個 token + 解析字母後，選項分布才恢復均勻
（11/12/15/10），準確率也回到 Qwen3-8B 應有的 **72.8%**。

**為什麼 KL 不跟 PPL 合併？** PPL 只看「base 自己那條路徑上的機率」，
KL 看「整個分布怎麼搬家」。abliteration 可能把機率搬到別處（PPL 幾乎不變）
但讓模型在別的路徑上做不同選擇——那正是 KL 抓得到、PPL 抓不到的東西。

**為什麼拒答題庫沿用 `llm-refusal-audit`？** 那個 repo 已經公布 heretic
各 GGUF quant 的 keyword TPR（q4_K_S 0% / q4_K_M 4% / q8_0 2%）。
同一份題庫才能問出「**量化 + Ollama 的結果，在 bf16 全精度下還成立嗎**」。
