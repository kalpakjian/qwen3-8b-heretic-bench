#!/usr/bin/env bash
# =============================================================================
# 一鍵執行：自測 -> 權重稽核 -> 拒答率 -> judge 複判 -> KL -> 能力 -> 報告
#
#   bash run_all.sh                # 完整流程（約 60–90 分鐘，含 33 GB 模型下載）
#   SMOKE=1 bash run_all.sh        # 煙霧測試（每階段縮到極小樣本，約 10 分鐘）
#
# 可用環境變數覆寫：
#   WORKDIR (預設 ~/heretic-bench)   MAX_TOKENS (預設 256)   BATCH (預設 8)
#   MMLU_N (每科題數，預設 20)       GSM8K_N (預設 100)      KL_WINDOWS (預設 20)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WORKDIR="${WORKDIR:-$HOME/heretic-bench}"
VENV="$WORKDIR/.venv"
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"

MAX_TOKENS="${MAX_TOKENS:-256}"
BATCH="${BATCH:-8}"
MMLU_N="${MMLU_N:-20}"
GSM8K_N="${GSM8K_N:-100}"
KL_WINDOWS="${KL_WINDOWS:-20}"

if [ "${SMOKE:-0}" = "1" ]; then
  echo "[smoke] 縮小樣本：只驗證流程會動，數字沒有統計意義"
  MMLU_N=2; GSM8K_N=8; KL_WINDOWS=2
fi

if [ ! -x "$VENV/bin/python" ]; then
  echo "[fail] 找不到 venv：$VENV，請先執行 bash setup.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

step() {
  echo
  echo "############################################################"
  echo "# $*"
  echo "############################################################"
}

step "Phase 0 — 不需 GPU 的邏輯自測"
python selftest.py

step "Phase 0b — 權重稽核（純 CPU，證明模型真的被改過）"
python weight_audit.py || echo "[warn] 權重稽核失敗，不影響其他階段"

step "Phase 1 — 拒答率（100 題，base vs heretic）"
python refusal_eval.py --max-new-tokens "$MAX_TOKENS" --batch-size "$BATCH"

step "Phase 1b — LLM-judge 複判（修正關鍵字法的雙向失準）"
python judge_rescore.py --batch-size "$BATCH" \
  || echo "[warn] judge 階段失敗；報告會退回只看 keyword（判讀時請記得這個限制）"

step "Phase 2 — KL divergence（全序列，heretic‖base）"
python kl_eval.py --max-windows "$KL_WINDOWS"

step "Phase 3 — 能力保留（MMLU / GSM8K / PPL）"
python capability_eval.py --limit-mmlu "$MMLU_N" --limit-gsm8k "$GSM8K_N" --batch-size "$BATCH"

step "Phase 4 — 彙整報告"
python make_report.py

echo
echo "[done] 產出："
ls -la ../results/*.json ../results/report.md 2>/dev/null || true
echo
echo "注意：results/_cache/ 是 KL 階段暫存的 base log-prob（約 3 GB），可安全刪除。"

