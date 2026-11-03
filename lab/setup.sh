#!/usr/bin/env bash
# =============================================================================
# Qwen3-8B-Heretic 基準測試環境建置腳本 (Arch Linux / Omarchy)
#   目標：Qwen/Qwen3-8B (base) vs kalpakjian/qwen3-8b-heretic (abliterated, bf16)
#   用法：bash setup.sh             # Python 環境 + 下載兩個模型（約 33 GB）
#         bash setup.sh --no-models # 只建 Python 環境，不下載模型
#   特性：冪等，重複執行安全
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$HOME/heretic-bench}"
VENV="$WORKDIR/.venv"
export HF_HOME="${HF_HOME:-$WORKDIR/hf-cache}"
WITH_MODELS=1
[ "${1:-}" = "--no-models" ] && WITH_MODELS=0

log()  { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

mkdir -p "$WORKDIR" "$HF_HOME"

# ---------------------------------------------------------------- 0. 環境檢查
log "0/6 環境檢查"
command -v pacman >/dev/null 2>&1 || warn "找不到 pacman（非 Arch 系？）—— Linux 上仍可繼續"
[ -f /etc/os-release ] && echo "  OS      : $(grep -m1 PRETTY_NAME /etc/os-release | cut -d'"' -f2)"
echo "  Kernel  : $(uname -r)"
echo "  工作目錄: $WORKDIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "找不到 nvidia-smi。Omarchy 需要先裝 NVIDIA 驅動：
       omarchy install nvidia && sudo mkinitcpio -P && reboot"
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  || warn "nvidia-smi 執行失敗，可能需重開機載入模組"
echo "  磁碟    : $(df -h "$WORKDIR" | awk 'NR==2{print $4" 可用 / "$2}')"

# 兩個 8B bf16 模型各 16.4 GB（HF 快取會存一份 blob + 一份 snapshot 連結），
# 加上 venv 的 torch（約 6 GB）與 MMLU/GSM8K/wikitext 資料集。
AVAIL_GB=$(df -BG "$WORKDIR" | awk 'NR==2{print $4}' | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 45 ] || warn "可用空間約 ${AVAIL_GB}GB，低於建議的 45GB（2×16.4GB 模型 + 6GB venv + 資料集）"

# 記憶體：兩個模型「不會」同時載入 GPU（見 notes/heretic-method-notes.md §4），
# 但 KL 階段需要把 base 的 logits 暫存到磁碟，故需要一些 RAM 與磁碟。
echo "  記憶體  : $(free -g | awk '/^Mem:/{print $2" GB 總計 / "$7" GB 可用"}')"

# ---------------------------------------------------------------- 1. 安裝 uv
log "1/6 準備 uv (Python 環境管理)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

# ------------------------------------------------------- 2. 建立 venv + 套件
log "2/6 建立 Python 3.12 虛擬環境並安裝套件"
# --allow-existing：uv >= 0.8 對已存在的 venv 會直接報錯，
# 沒有這個旗標的話「冪等」就不成立（第二次執行會死在這裡）。
uv venv --python 3.12 --allow-existing "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install --upgrade pip >/dev/null

# 版本以 2026-11-04 的 PyPI 為準（torch 2.15.0 / transformers 5.22.1）。
# 注意：Linux 上 PyPI 的 torch 就是 CUDA build（與 Windows 不同，那邊是 CPU-only）。
uv pip install \
  "torch==2.15.0" \
  "transformers==5.22.1" \
  "datasets==5.1.0" \
  "accelerate==1.15.0" \
  "huggingface_hub[hf_xet]"

# ------------------------------------------------------------------ 3. 自檢
log "3/6 CUDA / 套件自檢"
if ! python "$HERE/env_check.py"; then
  warn "torch 看不到 GPU。若裝到 CPU-only wheel，改用官方 index 重裝："
  warn "  uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu130"
  exit 1
fi

# ---------------------------------------------------------------- 4. 下載模型
if [ "$WITH_MODELS" = "1" ]; then
  log "4/6 下載模型（base 16.4 GB + heretic 16.4 GB，首次會很久）"
  python "$HERE/download_models.py"
else
  log "4/6 略過模型下載（--no-models）；之後可執行：python download_models.py"
fi

# ---------------------------------------------------------------- 5. 資料集
log "5/6 預抓資料集（MMLU / GSM8K / wikitext-2），避免正式跑時被下載卡住"
python - <<'PY'
from datasets import load_dataset
for name, kw in [
    ("cais/mmlu", dict(name="all", split="test[:20]")),
    ("openai/gsm8k", dict(name="main", split="test[:20]")),
    ("Salesforce/wikitext", dict(name="wikitext-2-raw-v1", split="test[:20]")),
]:
    try:
        n = len(load_dataset(name, **kw))
        print(f"  [ok] {name} ({n} rows sampled)")
    except Exception as e:
        print(f"  [warn] {name}: {type(e).__name__}: {e}")
PY

log "6/6 完成"
cat <<EOF

環境已就緒。下一步：
  cd $HERE
  source $VENV/bin/activate
  bash run_all.sh          # 一鍵跑完 Phase 0–4

備註：
  * 模型快取：$HF_HOME
  * 兩個 8B 模型不會同時載入 GPU；KL 階段以「先存 base logits 再算」的兩段式進行。
EOF
