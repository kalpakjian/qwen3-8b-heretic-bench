#!/usr/bin/env python3
"""
環境檢查：印出版本與 CUDA 狀態；有 GPU 時跑一次 bf16 matmul。

刻意做成獨立檔案（而非 `python -c "..."`）：
  Windows PowerShell 5.1 在把字串當參數傳給原生程式時會吃掉內嵌的雙引號，
  python 片段容易出現 NameError；用檔案傳遞可完全避開引號問題。

退出碼：0 = 全部正常；1 = 找不到 CUDA 或 matmul 失敗。
"""
import sys


def main() -> int:
    ok = True
    try:
        import torch
        import transformers
        import datasets
    except ImportError as e:
        print(f"  [fail] 匯入失敗：{e}")
        print("         請先跑 setup.sh，或確認 venv 已啟用。")
        return 1

    print(f"  torch        {torch.__version__}")
    print(f"  transformers {transformers.__version__}")
    print(f"  datasets     {datasets.__version__}")

    avail = torch.cuda.is_available()
    print(f"  cuda avail   {avail}")
    if not avail:
        print("  [fail] torch 看不到 GPU。若顯示 +cpu 版，代表裝到 CPU-only wheel。")
        print("         修正：uv pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cu130")
        return 1

    props = torch.cuda.get_device_properties(0)
    print(f"  gpu          {props.name}  {props.total_memory / 1024 ** 3:.1f} GB  sm_{props.major}{props.minor}")
    print(f"  cuda runtime {torch.version.cuda}")

    # bf16 是 Phase 2/3 的預設精度；若這裡失敗，後面全部不用跑。
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        b = a @ a
        torch.cuda.synchronize()
        print(f"  [ok] bf16 matmul -> {tuple(b.shape)}")
    except Exception as e:
        print(f"  [fail] bf16 matmul 失敗：{e}")
        ok = False

    # Qwen3-8B bf16 約 16.4 GB；留 4 GB 給 KV cache 與 activation 才跑得動 batching。
    free_gb = (props.total_memory - torch.cuda.memory_allocated()) / 1024 ** 3
    if free_gb < 18:
        print(f"  [warn] 可用 VRAM {free_gb:.1f} GB < 18 GB，批次大小會被壓到很低")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
