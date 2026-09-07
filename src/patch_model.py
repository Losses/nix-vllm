#!/usr/bin/env python3
"""Patch model.py and mtp.py for int8 LM head quantization.
Supports both Qwen4Exp (newer snapshot / nightly) and Qwen3_8FlashNext.
"""
import ast
import os
import sys

def patch_file(path: str) -> bool:
    if not os.path.exists(path):
        return False
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    target = 'prefix=maybe_prefix(prefix, "lm_head"),'
    if 'quant_config=vllm_config.quant_config' in content:
        print(f"[lm_head_patch] Already patched: {path}", file=sys.stderr)
        return True

    if target not in content:
        print(f"[lm_head_patch] Target anchor not found in: {path}", file=sys.stderr)
        return False

    replacement = 'quant_config=vllm_config.quant_config,\n            prefix=maybe_prefix(prefix, "lm_head"),'
    new_content = content.replace(target, replacement)

    # Verify python syntax before writing
    ast.parse(new_content)

    if not os.path.exists(path + ".orig"):
        with open(path + ".orig", "w", encoding="utf-8") as orig_f:
            orig_f.write(content)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[lm_head_patch] Patched {path} successfully", file=sys.stderr)
    return True

def main():
    vllm_models_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if not vllm_models_dir:
        import vllm
        vllm_models_dir = os.path.join(os.path.dirname(vllm.__file__), "models")

    # Check both qwen4_exp and qwen3_8_flash_next
    candidates = ["qwen4_exp", "qwen3_8_flash_next"]
    patched_any = False

    for name in candidates:
        target_dir = os.path.join(vllm_models_dir, name, "nvidia")
        if os.path.isdir(target_dir):
            print(f"[lm_head_patch] Found architecture directory: {target_dir}", file=sys.stderr)
            for py_name in ["model.py", "mtp.py"]:
                if patch_file(os.path.join(target_dir, py_name)):
                    patched_any = True

    if not patched_any:
        print(f"[lm_head_patch] Warning: Neither qwen4_exp nor qwen3_8_flash_next found in {vllm_models_dir}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
