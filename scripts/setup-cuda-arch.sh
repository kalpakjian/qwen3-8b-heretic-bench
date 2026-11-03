#!/usr/bin/env bash
# setup-cuda-arch.sh -- prepare an Omarchy/Arch box to run bench_arc.py on an NVIDIA GPU.
#
#   ./scripts/setup-cuda-arch.sh            # install deps + create venv + verify GPU
#   ./scripts/setup-cuda-arch.sh --check    # verify only, install nothing
#   ./scripts/setup-cuda-arch.sh --vllm     # also install vLLM (much faster runs)
#
# Safe to re-run: every step is idempotent.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO_ROOT/.venv"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
DO_CHECK_ONLY=0
WITH_VLLM=0

for arg in "$@"; do
  case "$arg" in
    --check) DO_CHECK_ONLY=1 ;;
    --vllm)  WITH_VLLM=1 ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown flag: $arg" >&2; exit 1 ;;
  esac
done

c_ok()   { printf '\033[32m[ OK ]\033[0m %s\n' "$*"; }
c_warn() { printf '\033[33m[WARN]\033[0m %s\n' "$*"; }
c_bad()  { printf '\033[31m[FAIL]\033[0m %s\n' "$*"; }
c_info() { printf '\033[90m       %s\033[0m\n' "$*"; }
hdr()    { printf '\n\033[36m== %s ==\033[0m\n' "$*"; }

need_pkg() { ! pacman -Q "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- 1. distro --
hdr "Environment"

if ! command -v pacman >/dev/null 2>&1; then
  c_bad "pacman not found. This script targets Omarchy/Arch Linux."
  exit 1
fi
c_ok "Arch-based system detected"

if [[ -f /etc/omarchy-version ]] || [[ -d "$HOME/.local/share/omarchy" ]]; then
  c_ok "Omarchy detected ($(cat /etc/omarchy-version 2>/dev/null || echo 'version file absent'))"
else
  c_info "Omarchy marker not found; continuing as plain Arch."
fi

if [[ $EUID -eq 0 ]]; then
  c_warn "Running as root. The venv will be owned by root - prefer a normal user."
fi

# ------------------------------------------------------------ 2. NVIDIA GPU --
hdr "NVIDIA driver / GPU"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  c_bad "nvidia-smi not found -- proprietary NVIDIA driver is not installed."
  c_info "Install with:"
  c_info "  sudo pacman -S --needed nvidia nvidia-utils"
  c_info "Then REBOOT (kernel module must load) and re-run this script."
  NVIDIA_READY=0
elif ! nvidia-smi >/dev/null 2>&1; then
  c_bad "nvidia-smi exists but cannot talk to the driver."
  c_info "Common cause: driver installed but kernel module not loaded, or a"
  c_info "kernel update happened without rebuilding. Try: reboot; or"
  c_info "  sudo pacman -S --needed linux-headers nvidia"
  NVIDIA_READY=0
else
  NVIDIA_READY=1
  c_ok "nvidia-smi works"
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version \
    --format=csv,noheader | sed 's/^/       /'
fi

if [[ $NVIDIA_READY -eq 1 ]]; then
  hdr "VRAM headroom"
  read -r total used <<<"$(nvidia-smi --query-gpu=memory.total,memory.used \
      --format=csv,noheader,nounits | head -1 | tr -d ',' | awk '{print $1, $2}')"
  free=$(( total - used ))
  c_info "total ${total} MiB | used ${used} MiB | free ${free} MiB"
  if (( free < 18000 )); then
    c_warn "Under 18 GB free. Qwen3-8B in bf16 needs ~16 GB plus headroom."
    c_info "Close GPU consumers first:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader 2>/dev/null | sed 's/^/       /' || true
  else
    c_ok "Enough free VRAM for a full bf16 run"
  fi

# --------------------------------------------------------------- 3. python ---
hdr "Python packages"

if [[ $DO_CHECK_ONLY -eq 0 ]]; then
  PKGS=()
  need_pkg python        && PKGS+=(python)
  need_pkg python-pip    && PKGS+=(python-pip)
  need_pkg base-devel    && PKGS+=(base-devel)
  need_pkg git           && PKGS+=(git)
  need_pkg python-virtualenv && true   # venv ships with python; no extra pkg needed

  if (( ${#PKGS[@]} > 0 )); then
    c_info "Installing: ${PKGS[*]}"
    sudo pacman -S --needed --noconfirm "${PKGS[@]}"
  else
    c_ok "Base packages already present"
  fi
else
  c_info "--check: skipping pacman"
fi

command -v python >/dev/null || { c_bad "python still missing"; exit 1; }
c_ok "python $(python --version 2>&1 | awk '{print $2}')"

# ----------------------------------------------------------------- 4. venv ---
hdr "Virtualenv"

# Arch blocks system-wide pip (PEP 668, externally-managed-environment).
# A venv is mandatory -- do NOT use --break-system-packages.
if [[ $DO_CHECK_ONLY -eq 0 ]]; then
  if [[ ! -d "$VENV" ]]; then
    c_info "Creating $VENV"
    python -m venv "$VENV"
  else
    c_ok "venv already exists: $VENV"
  fi
fi

if [[ ! -f "$VENV/bin/activate" ]]; then
  c_bad "No venv at $VENV -- rerun without --check."
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
c_ok "Activated $(python -V)"

# ----------------------------------------------------------------- 5. torch --
hdr "PyTorch (CUDA build)"

if [[ $DO_CHECK_ONLY -eq 0 ]]; then
  # Explicit index-url: the default PyPI torch on many setups resolves to a
  # CPU-only wheel, which is exactly the trap this repo exists to avoid.
  pip install --quiet --upgrade pip
  pip install --quiet --index-url "$TORCH_INDEX" torch
  pip install --quiet -r "$REPO_ROOT/requirements.txt"
  if [[ $WITH_VLLM -eq 1 ]]; then
    c_info "Installing vLLM (large download)..."
    pip install --quiet vllm || c_warn "vLLM install failed -- HF backend still works."
  fi
else
  c_info "--check: skipping pip"
fi

# --------------------------------------------------------- 6. sanity checks --
hdr "CUDA verification"

python - <<'PY'
import sys
try:
    import torch
except ImportError:
    sys.exit("torch import failed")

print(f"       torch          : {torch.__version__}")
print(f"       cuda available : {torch.cuda.is_available()}")
print(f"       cuda version   : {torch.version.cuda}")
if not torch.cuda.is_available():
    print("\n[FAIL] torch cannot see the GPU.")
    print("       A '+cpu' build is the usual cause. Reinstall with:")
    print("         pip install --force-reinstall --index-url "
          "https://download.pytorch.org/whl/cu128 torch")
    sys.exit(1)

name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info(0)
print(f"       device         : {name}")
print(f"       vram           : {free/1024**3:.1f} GiB free / {total/1024**3:.1f} GiB total")

x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().mean().item()
print(f"       bf16 matmul    : ok ({y:.4f})")
print("\n[ OK ] CUDA stack is functional")
PY

# ----------------------------------------------------------------- 7. next ---
hdr "Next steps"

cat <<EOF
  cd "$REPO_ROOT"
  source .venv/bin/activate

  # quick plumbing check (~2 min)
  python bench_arc.py --limit 20 --tag smoke

  # real runs
  python bench_arc.py --no-thinking --tag heretic
  python bench_arc.py --no-thinking --tag base --model Qwen/Qwen3-8B --root-weights

  # faster, same task/config (compare vLLM against vLLM only)
  python bench_arc.py --backend vllm --no-thinking --tag heretic
EOF

if [[ $NVIDIA_READY -eq 0 ]]; then
  printf '\n'
  c_warn "NVIDIA driver was not ready -- fix that before running benchmarks."
fi

fi
