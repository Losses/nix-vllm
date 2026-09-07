# Qwen3.8-Flash-Next / Qwen4Exp on Blackwell with vLLM
#
# Builds from official image + third-party patches:
# - int4+fp8 hybrid dispatch
# - int8 LM head quantization support
# - prefix caching 1600-block mamba alignment & slot seed fix
# - mamba align-mode state copy guard
# - never-evict prompt pinning
# - prefix cache debug tracing
# - Blackwell / GB10 FLA shmem and warps tuning
#
# (mmap PLE patch excluded: discrete GPU with resident memory / native CPU offload)

ARG BASE_IMAGE=vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
FROM ${BASE_IMAGE}

ARG SP=/usr/local/lib/python3.12/dist-packages

# 1. spark-fla-shmem: sm120/sm121 99 KiB shared mem
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
RUN if [ -f "${FLA_UTILS}" ]; then \
      sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # Blackwell/GB10 99KiB|' ${FLA_UTILS}; \
    fi

# 2. spark-fla-warps: Blackwell tl.dot race fix
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN if [ -f "${FLA_CDH}" ]; then \
      sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # Blackwell race fix|' ${FLA_CDH}; \
    fi

# 3. int4+fp8 hybrid: dispatch blockwise-fp8 side layers from AutoGPTQConfig
ARG GPTQ_PY=${SP}/vllm/model_executor/layers/quantization/auto_gptq.py
COPY src/vllm_fp8_hybrid.py ${SP}/vllm_fp8_hybrid.py
RUN printf '\n\n# --- qwen-vllm: int4+fp8 hybrid dispatch (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> ${GPTQ_PY} \
 && python3 -c "import ast; ast.parse(open('${GPTQ_PY}').read()); print('auto_gptq.py patched OK')"

# 4. never-evict prompt pinning
COPY src/patch_never_evict.py /tmp/patch_never_evict.py
RUN python3 /tmp/patch_never_evict.py && rm /tmp/patch_never_evict.py

# 5. LM head quantization patch (supports both Qwen4Exp and Qwen3_8FlashNext)
COPY src/patch_model.py /tmp/patch_model.py
RUN python3 /tmp/patch_model.py ${SP}/vllm/models && rm /tmp/patch_model.py

# 6. mamba align-mode state-copy hardening
ARG MAMBA_UTILS=${SP}/vllm/v1/worker/mamba_utils.py
COPY src/mamba_utils_guarded.py ${MAMBA_UTILS}
RUN python3 -c "import ast; ast.parse(open('${MAMBA_UTILS}').read()); print('mamba_utils.py guarded OK')"

# 7. prefix-cache diagnosis logging
COPY src/patch_hit_debug.py /tmp/patch_hit_debug.py
RUN python3 /tmp/patch_hit_debug.py && rm /tmp/patch_hit_debug.py

# 8. prefix-cache MAMBA block alignment split
COPY src/patch_mamba_align_split.py /tmp/patch_mamba_align_split.py
RUN python3 /tmp/patch_mamba_align_split.py && rm /tmp/patch_mamba_align_split.py

# 9. on-demand step profiling
COPY src/patch_step_profile.py /tmp/patch_step_profile.py
RUN python3 /tmp/patch_step_profile.py && rm /tmp/patch_step_profile.py

