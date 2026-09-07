#!/usr/bin/env python3
"""Patch vLLM for single-GPU PLE CPU offload and ModelOpt checkpoint compatibility.

1. uniproc_executor.py:
   Ensures UniProcExecutor initializes the PLE CPU offload worker process
   before load_model and waits for it to become ready.
   (Previously only MultiprocExecutor had this hook in the base image).

2. ple_layer.py:
   Filters out quantization scale parameters (*.weight_scale, *.scale) for
   ngram_embedding when loading ModelOpt NVFP4 checkpoints.
"""
import ast
import os
import sys

def patch_uniproc_executor(vllm_dir: str) -> bool:
    path = os.path.join(vllm_dir, "v1", "executor", "uniproc_executor.py")
    if not os.path.exists(path):
        print(f"[-] File not found: {path}", file=sys.stderr)
        return False

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "spawn_ple_offload" in content:
        print(f"[=] uniproc_executor.py already patched", file=sys.stderr)
        return True

    target = '        self.collective_rpc("load_model")'
    if target not in content:
        print(f"[!] Target anchor not found in uniproc_executor.py", file=sys.stderr)
        return False

    replacement = '''        if envs.VLLM_PLE_CPU_OFFLOAD:
            from vllm.v1.worker.ple_offload_connector import spawn_ple_offload, wait_ple_offload_ready
            spawn_ple_offload()

        self.collective_rpc("load_model")

        if envs.VLLM_PLE_CPU_OFFLOAD:
            wait_ple_offload_ready()'''

    new_content = content.replace(target, replacement, 1)
    ast.parse(new_content)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[+] Successfully patched uniproc_executor.py", file=sys.stderr)
    return True

def patch_ple_layer(vllm_dir: str) -> bool:
    path = os.path.join(vllm_dir, "model_executor", "layers", "ple_layer.py")
    if not os.path.exists(path):
        print(f"[-] File not found: {path}", file=sys.stderr)
        return False

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "weight_scale" in content:
        print(f"[=] ple_layer.py already patched", file=sys.stderr)
        return True

    target = '            if "ngram_embedding" in name:\n                loader = AutoWeightsLoader(self.ngram_embedding)'
    if target not in content:
        print(f"[!] Target anchor not found in ple_layer.py", file=sys.stderr)
        return False

    replacement = '''            if "ngram_embedding" in name:
                if any(name.endswith(s) for s in (".weight_scale", ".weight_scale_inv", "_weight_scale", "_weight_scale_inv", ".scale", ".input_scale")):
                    continue
                loader = AutoWeightsLoader(self.ngram_embedding)'''

    new_content = content.replace(target, replacement, 1)
    ast.parse(new_content)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[+] Successfully patched ple_layer.py", file=sys.stderr)
    return True

def patch_qwen3_8_ple_layer(vllm_dir: str) -> bool:
    path = os.path.join(vllm_dir, "models", "qwen3_8_flash_next", "nvidia", "ple_layer.py")
    if not os.path.exists(path):
        return True

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "shard_.weight" in content:
        print(f"[=] qwen3_8_flash_next/nvidia/ple_layer.py already patched", file=sys.stderr)
        return True

    target = '            if name.startswith(shard_prefix) and name.endswith(".weight"):'
    if target not in content:
        print(f"[!] Target anchor not found in qwen3_8_flash_next/nvidia/ple_layer.py", file=sys.stderr)
        return False

    replacement = '''            if name in ("shard_.weight", "ngram_embedding.weight") or name.endswith(".shard_.weight"):
                embedding = self.ngram_embedding
                copy_ple_embedding_shard_(
                    embedding.weight.data,
                    loaded_weight,
                    checkpoint_start=0,
                    tp_start=embedding.shard_indices.org_vocab_start_index,
                    tp_end=embedding.shard_indices.org_vocab_end_index,
                )
                loaded.add("ngram_embedding.weight")
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):'''

    new_content = content.replace(target, replacement, 1)
    ast.parse(new_content)

    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"[+] Successfully patched qwen3_8_flash_next/nvidia/ple_layer.py", file=sys.stderr)
    return True

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    if target_dir:
        vllm_dir = os.path.join(target_dir, "vllm") if not target_dir.endswith("vllm") else target_dir
    else:
        import vllm
        vllm_dir = os.path.dirname(vllm.__file__)

    ok1 = patch_uniproc_executor(vllm_dir)
    ok2 = patch_ple_layer(vllm_dir)
    ok3 = patch_qwen3_8_ple_layer(vllm_dir)
    return 0 if (ok1 and ok2 and ok3) else 1

if __name__ == "__main__":
    sys.exit(main())
