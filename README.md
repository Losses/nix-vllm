# vLLM Qwen3.8-Flash-Next / Qwen4Exp on Blackwell (NixOS)

Nix Flake for running Qwen3.8-Flash-Next (architecture name `Qwen4Exp` in newer vLLM code) on NixOS with NVIDIA Blackwell GPUs (sm_120, such as RTX PRO 6000 96GB).

Based on patches and techniques from [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) and [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX).

---

## Starting the Server

### 1. Single 96GB GPU with PLE CPU Offload (Recommended for NVFP4)

Running the 123.6 GiB NVFP4 model on a single 96GB GPU requires offloading the 44 GiB PLE n-gram table to Host RAM. The Docker target includes the required offload subsystem and all Blackwell kernel patches:

```bash
# Serve directly (mounts ~/.cache/huggingface and binds to port 8000)
nix run .#docker -- --model nvidia/Qwen3.8-Flash-Next-NVFP4 --port 8000 --mtp 0

# Or run the script directly:
./scripts/serve-docker.sh --model nvidia/Qwen3.8-Flash-Next-NVFP4 --port 8000 --mtp 0
```

Note on `--mtp 0`: The current checkpoint weights for MTP in `nvidia/Qwen3.8-Flash-Next-NVFP4` lack `w2_weight_scale_inv` parameters. Set `--mtp 0` to run base model inference without the speculative draft layer.

By default, the server runs in the foreground and streams logs directly to your terminal. Press `Ctrl+C` to cleanly stop and remove the container. Add `-d` to run in the background.

### 2. Native Nix FHS Runtime (For models fitting in VRAM or multi-GPU)

The native FHS target installs official vLLM nightly binary wheels directly into `~/.local/share/nix-vllm/venv` without compilation:

```bash
# Bootstrap prebuilt wheels and apply patches (first run only)
nix run .#setup

# Start server
nix run .#serve -- --model <model-path-or-hf-id> --port 8000

# Smoke test the running server
nix run .#smoke-test
```

---

## Hugging Face Integration

This flake provides the official Hugging Face CLI wrapped inside the FHS environment. It shares your system token and cache (`~/.cache/huggingface`).

### Via `nix run`

```bash
# Authenticate (for gated models)
nix run .#hf -- auth login
nix run .#hf -- auth whoami

# Download model snapshots
nix run .#hf -- download nvidia/Qwen3.8-Flash-Next-NVFP4
```

### Via `nix develop`

```bash
nix develop

hf auth login
hf download nvidia/Qwen3.8-Flash-Next-NVFP4
```

---

## What Problem This Repository Solves

### 1. The 120GB Memory Footprint vs 96GB Single GPU
`Qwen3.8-Flash-Next` (and `Qwen4Exp`) uses a Prompt Lookup Engine (PLE) backed by a massive n-gram hash table:
- Base model weights + MoE (NVFP4): ~76 GiB
- PLE n-gram lookup table: ~44 GiB
- Total footprint: ~123.6 GiB

On a single 96 GiB GPU (RTX PRO 6000), placing both in VRAM immediately causes a CUDA Out-of-Memory error.

### 2. Mainline vLLM Dropped PLE CPU Offloading
In mainline vLLM (including 0.28.x nightly), upstream merged `Qwen4Exp` as purely GPU-resident. The `vllm/v1/ple_offload` process tree and `PleOffloadLayer` were removed. Consequently, mainline vLLM attempts to allocate all 120+ GiB on the primary GPU.

This flake provides the patched 0.25.1 git snapshot where `VLLM_PLE_CPU_OFFLOAD=1` functions properly. The 44 GiB PLE table is held in pinned Host RAM, and lookups are synchronized asynchronously with the GPU stream via CUDA IPC and stream memory operations (`cuStreamWaitValue32`/`cuStreamWriteValue32`). GPU VRAM usage is capped at ~77 GiB, leaving headroom for KV cache.

### 3. Blackwell (sm_120) Kernel Fixes
- **FLA Shared Memory Limit**: Blackwell hardware restricts available shared memory for Flash Linear Attention kernels. The shmem threshold is adjusted to 99 KiB (101,376 bytes) to prevent Triton shared memory allocation errors.
- **Blackwell tl.dot Race Fix**: Enforces `num_warps = [2]` in `chunk_delta_h.py` to prevent warp race conditions on sm_120.
- **INT4 + FP8 Hybrid Dispatch**: Dispatches blockwise-FP8 layers from AutoGPTQ configs cleanly.
- **Prefix Caching Alignment**: Fixes Mamba 1600-block state cache alignment bugs.

---

## Flake Targets

| Command | Description |
|---|---|
| `nix run .#docker -- [args]` | Start vLLM in the Blackwell container with PLE CPU offload |
| `nix run .#docker-build` | Rebuild the Blackwell-patched Docker image locally |
| `nix run .#serve -- [args]` | Start native host server via Nix FHS |
| `nix run .#setup` | Install official prebuilt wheels and patches into user venv |
| `nix run .#hf -- [args]` | Hugging Face CLI wrapper (`auth login`, `download`, etc.) |
| `nix run .#smoke-test` | Send a test completion request to localhost:8000 |
| `nix develop` | Open dev shell containing `vllm-serve`, `vllm-docker-serve`, `hf`, and `uv` |

---

## NixOS Module

To run vLLM as a systemd service, add this flake to your `/etc/nixos/flake.nix`:

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-vllm.url = "path:/home/losses/Development/nix-vllm";
  };

  outputs = { self, nixpkgs, nix-vllm, ... }: {
    nixosConfigurations.myhost = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        nix-vllm.nixosModules.default
        {
          services.vllm-qwen = {
            enable = true;
            modelDir = "nvidia/Qwen3.8-Flash-Next-NVFP4";
            port = 8000;
            mtp = 3;
            gpuMemoryUtilization = "0.85";
            pleCpuOffload = true;
          };
        }
      ];
    };
  };
}
```
