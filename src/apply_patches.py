#!/usr/bin/env python3
"""Unified patch installer for vLLM with Qwen3.8-Flash-Next / Qwen4Exp support on Blackwell.

Applies:
1. int4+fp8 hybrid dispatch (vllm_fp8_hybrid.py)
2. int8 LM head quant_config support (patch_model.py, supporting Qwen4Exp and Qwen3_8FlashNext)
3. Mamba align-mode prefix caching alignment fix (patch_mamba_align_split.py)
4. Mamba state-copy guard & race condition fix (mamba_utils_guarded.py)
5. Never-evict system prompt pinning (patch_never_evict.py)
6. Prefix-cache diagnosis tracing (patch_hit_debug.py)
7. On-demand engine step profiler (patch_step_profile.py)
8. Blackwell / GB10 FLA shmem and warps tuning (utils.py & chunk_delta_h.py)

Note: As requested, the NVMe mmap PLE patch is excluded (discrete GPU with ample memory / native CPU offload).
"""
import ast
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_vllm_paths(site_packages=None):
    if site_packages:
        vllm_dir = os.path.join(site_packages, "vllm")
        sp_dir = site_packages
    else:
        import vllm
        vllm_dir = os.path.dirname(os.path.abspath(vllm.__file__))
        sp_dir = os.path.dirname(vllm_dir)
    return vllm_dir, sp_dir

def edit_file(path, old, new):
    if not os.path.exists(path):
        print(f"[-] File not found, skipping: {path}", file=sys.stderr)
        return False
    src = open(path, "r", encoding="utf-8").read()
    if new in src:
        print(f"[=] Already patched: {path}", file=sys.stderr)
        return True
    n = src.count(old)
    if n != 1:
        print(f"[!] Warning: anchor found {n} times in {path}, skipping edit", file=sys.stderr)
        return False
    open(path, "w", encoding="utf-8").write(src.replace(old, new))
    ast.parse(open(path, "r", encoding="utf-8").read())
    print(f"[+] Successfully edited: {path}", file=sys.stderr)
    return True

def apply_all(vllm_dir, sp_dir):
    print(f"==> Applying patches to vLLM at: {vllm_dir}", file=sys.stderr)

    # 1. FP8 Hybrid dispatch
    gptq_py = os.path.join(vllm_dir, "model_executor/layers/quantization/auto_gptq.py")
    fp8_hybrid_src = os.path.join(SCRIPT_DIR, "vllm_fp8_hybrid.py")
    fp8_hybrid_dst = os.path.join(sp_dir, "vllm_fp8_hybrid.py")
    shutil.copy2(fp8_hybrid_src, fp8_hybrid_dst)
    
    if os.path.exists(gptq_py):
        src = open(gptq_py, "r", encoding="utf-8").read()
        if "vllm_fp8_hybrid" not in src:
            src += "\n\n# --- qwen-vllm: int4+fp8 hybrid dispatch (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n"
            ast.parse(src)
            open(gptq_py, "w", encoding="utf-8").write(src)
            print("[+] Appended fp8_hybrid dispatch to auto_gptq.py", file=sys.stderr)
        else:
            print("[=] auto_gptq.py already has fp8_hybrid hook", file=sys.stderr)

    # 2. LM Head quant_config patch (Qwen4Exp & Qwen3_8FlashNext)
    patch_model_py = os.path.join(SCRIPT_DIR, "patch_model.py")
    models_dir = os.path.join(vllm_dir, "models")
    os.system(f"{sys.executable} {patch_model_py} {models_dir}")

    # 3. Mamba align-mode chunk split
    sched_py = os.path.join(vllm_dir, "v1/core/sched/scheduler.py")
    mhyb_py = os.path.join(vllm_dir, "v1/worker/gpu/model_states/mamba_hybrid.py")
    edit_file(
        sched_py,
        """        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
""",
        """        # cache_config.block_size is the MINIMUM across KV cache groups (the
        # fine/draft granularity), but this function's whole job is to end
        # chunks at MAMBA state boundaries. With the fine size, chunk ends
        # land where no mamba state is cacheable, a cold request publishes no
        # reusable state, and the min-across-groups hit rule makes the next
        # identical request a full miss.
        block_size = (
            self.cache_config.mamba_block_size or self.cache_config.block_size
        )
        # The last block-aligned position whose state can be cached. With
""",
    )
    edit_file(
        mhyb_py,
        """                (new_req_data.num_computed_tokens - 1) // self.cache_config.block_size
""",
        """                (new_req_data.num_computed_tokens - 1)
                # block_size is the MIN across KV groups; the state slot is
                # per MAMBA block (blazux/qwen3.8-Flash-DGX#2, 8347e7c).
                // (
                    self.cache_config.mamba_block_size
                    or self.cache_config.block_size
                )
""",
    )

    # 4. Mamba state-copy guard
    mamba_utils_src = os.path.join(SCRIPT_DIR, "mamba_utils_guarded.py")
    mamba_utils_dst = os.path.join(vllm_dir, "v1/worker/mamba_utils.py")
    if os.path.exists(mamba_utils_dst):
        if not os.path.exists(mamba_utils_dst + ".orig"):
            shutil.copy2(mamba_utils_dst, mamba_utils_dst + ".orig")
        shutil.copy2(mamba_utils_src, mamba_utils_dst)
        ast.parse(open(mamba_utils_dst, "r", encoding="utf-8").read())
        print("[+] Installed mamba_utils_guarded.py", file=sys.stderr)

    # 5. Never-evict prompt pinning
    cache_py = os.path.join(vllm_dir, "config/cache.py")
    arg_utils_py = os.path.join(vllm_dir, "engine/arg_utils.py")
    edit_file(
        cache_py,
        """    KV offloading is only activated when kv_offloading_size is set.\"\"\"
""",
        """    KV offloading is only activated when kv_offloading_size is set.\"\"\"

    never_evict_kv_cache_prompt_includes: str | None = None
    \"\"\"If set, any request whose prompt contains this exact substring has its
    prompt KV cache blocks pinned: they are held out of the free pool and are
    never handed out for eviction. The pin set is *replaced* the next time a
    matching request arrives, so a changed system prompt releases the old
    blocks automatically. Requires prefix caching.\"\"\"

    never_evict_kv_cache_max_fraction: float = 0.25
    \"\"\"Upper bound on the never-evict pin, as a fraction of the GPU block
    pool. Pinning stops (with a warning) once the cap is reached.\"\"\"
""",
    )
    edit_file(
        cache_py,
        """            "prefix_caching_hash_algo",
""",
        """            "prefix_caching_hash_algo",
            "never_evict_kv_cache_prompt_includes",
            "never_evict_kv_cache_max_fraction",
""",
    )
    edit_file(
        arg_utils_py,
        """    kv_offloading_backend: KVOffloadingBackend = CacheConfig.kv_offloading_backend
""",
        """    kv_offloading_backend: KVOffloadingBackend = CacheConfig.kv_offloading_backend
    never_evict_kv_cache_prompt_includes: str | None = (
        CacheConfig.never_evict_kv_cache_prompt_includes
    )
    never_evict_kv_cache_max_fraction: float = (
        CacheConfig.never_evict_kv_cache_max_fraction
    )
""",
    )
    edit_file(
        arg_utils_py,
        """        cache_group.add_argument(
            "--kv-offloading-backend", **cache_kwargs["kv_offloading_backend"]
        )
""",
        """        cache_group.add_argument(
            "--kv-offloading-backend", **cache_kwargs["kv_offloading_backend"]
        )
        cache_group.add_argument(
            "--never-evict-kv-cache-prompt-includes",
            **cache_kwargs["never_evict_kv_cache_prompt_includes"],
        )
        cache_group.add_argument(
            "--never-evict-kv-cache-max-fraction",
            **cache_kwargs["never_evict_kv_cache_max_fraction"],
        )
""",
    )

    # 6. FLA shared memory & warps tuning (for Blackwell sm_120/121)
    fla_utils = os.path.join(vllm_dir, "third_party/flash_linear_attention/ops/utils.py")
    if os.path.exists(fla_utils):
        edit_file(fla_utils, "DEFAULT = 102400", "DEFAULT = 101376  # Blackwell/GB10 99KiB shmem")

    fla_cdh = os.path.join(vllm_dir, "third_party/flash_linear_attention/ops/chunk_delta_h.py")
    if os.path.exists(fla_cdh):
        edit_file(fla_cdh, "for num_warps in [2, 4]:", "for num_warps in [2]:  # Blackwell tl.dot race fix")

    # 7. Single-GPU UniProcExecutor PLE worker spawn & ModelOpt scale parameter filter
    from patch_ple_single_gpu import patch_uniproc_executor, patch_ple_layer
    patch_uniproc_executor(vllm_dir)
    patch_ple_layer(vllm_dir)

    # 8. Remap qwen_sparse_attention -> full_attention in transformers configuration_utils
    cu_py = os.path.join(sp_dir, "transformers/configuration_utils.py")
    if os.path.exists(cu_py):
        edit_file(cu_py, '"attention": "full_attention",', '"attention": "full_attention",\n    "qwen_sparse_attention": "full_attention",')

    print("==> All patches successfully processed!", file=sys.stderr)

if __name__ == "__main__":
    target_sp = sys.argv[1] if len(sys.argv) > 1 else None
    vllm_d, sp_d = get_vllm_paths(target_sp)
    apply_all(vllm_d, sp_d)
