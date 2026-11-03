#!/usr/bin/env python3
"""
下載本基準測試所需的兩個模型。

  base     : Qwen/Qwen3-8B            （Apache-2.0）→ HF snapshot 根目錄
  heretic  : kalpakjian/qwen3-8b-heretic（Apache-2.0）→ 權重在 model_files/ 子目錄

刻意只抓需要的檔案：base repo 另含 bfloat16/original/ 的 .pth 與 onnx/flax 權重，
全部下載會多花約 16 GB 且完全用不到。

用法：
  python download_models.py           # 兩個模型
  python download_models.py --base-only
"""
import sys

from huggingface_hub import snapshot_download

BASE_REPO = "Qwen/Qwen3-8B"
HERETIC_REPO = "kalpakjian/qwen3-8b-heretic"

# 只要 transformers 直接吃得下的格式；排除 .pth / onnx / flax / gguf / *.bin
IGNORE = [
    "*.pth",
    "*.bin",
    "*.h5",
    "*.msgpack",
    "*.onnx*",
    "*.gguf",
    "original/*",
    "bfloat16/*",
    "onnx/*",
    "flax_model*",
    "tf_model*",
]


def main() -> int:
    repos = [BASE_REPO] if "--base-only" in sys.argv else [BASE_REPO, HERETIC_REPO]
    for repo in repos:
        print(f"  -> {repo}", flush=True)
        path = snapshot_download(repo, ignore_patterns=IGNORE)
        print(f"     {path}", flush=True)

    if HERETIC_REPO in repos:
        # 這裡刻意檢查子目錄：heretic 的權重不在 repo 根目錄，搞錯 subfolder
        # 會得到 "no file named config.json" 這種令人誤解的錯誤。
        import os

        from huggingface_hub import hf_hub_download

        cfg = hf_hub_download(HERETIC_REPO, "model_files/config.json")
        print(f"  [ok] heretic config -> {os.path.dirname(cfg)}")
        print("       （載入時務必帶 subfolder='model_files'）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
