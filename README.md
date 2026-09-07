# NixOS Flake for Qwen3.8-Flash-Next / Qwen4Exp (vLLM on Blackwell)

为 NixOS (特别是 NVIDIA RTX PRO 6000 / Blackwell sm_120 / sm_121 架构及 CUDA 13.0+) 打造的 vLLM Flake。
针对 **Qwen3.8-Flash-Next**（新版 snapshot 中注册为 **Qwen4Exp**）进行了深度性能优化与缺陷修复。

> **参考仓库与致谢：**
> 参考自 [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) 与 [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)。

---

## 核心特性与适配

1. **直接采用官方二进制 Wheels，免从头编译**：
   - vLLM Nightly cu130: `https://wheels.vllm.ai/nightly/cu130`
   - PyTorch cu130: `https://download.pytorch.org/whl/cu130`
   - FlashInfer: `https://flashinfer.ai/whl/cu130` (`flashinfer-cubin`, `flashinfer-jit-cache`)
   - 通过 Nix `buildFHSEnv` 接入宿主机 NVIDIA 驱动 (`/run/opengl-driver/lib`)。

2. **排除 NVMe mmap PLE 补丁（无需脏补丁）**：
   - 原 Spark/GB10 设备因为 128GB 统一内存受限而使用 NVMe 盘 mmap PLE n-gram 表。
   - 在独立显卡服务器（如 RTX PRO 6000 拥有 96GB 显存，且有独立的 Host RAM）上，采用 vLLM 原生 `VLLM_PLE_CPU_OFFLOAD=1`（将 ~44GB 表 pin 在 Host RAM）或直接常驻显存，无需通过 NVMe mmap 降低吞吐。

3. **保留并适配的第三方优化补丁**：
   - **Qwen4Exp / Qwen3_8FlashNext 双重自适应**：自动侦测并修补 `vllm/models/qwen4_exp` 或 `vllm/models/qwen3_8_flash_next`。
   - **int4 + int8 + fp8 混合调度 (`vllm_fp8_hybrid.py`)**：在 `AutoGPTQConfig` 中自动截获 blockwise-fp8 侧层并路由到 `Fp8Config`。
   - **int8 LM Head quant_config 补丁 (`patch_model.py`)**：为 `ParallelLMHead` 传入 `quant_config`，解锁 int8 GPTQ head，节约显存并提升 GEMV 速度。
   - **Prefix Caching 1600-token MAMBA 块边界对齐 (`patch_mamba_align_split.py`)**：修复冷启动请求无法缓存 MAMBA 状态与状态槽位除以 block_size 导致的越界计算缺陷。
   - **Mamba 对齐模式状态复制加固 (`mamba_utils_guarded.py`)**：防范 CUDA 内存非法访问与状态复制竞态。
   - **Prompt 永不淘汰锁定 (`patch_never_evict.py`)**：`--never-evict-kv-cache-prompt-includes` 确保长系统提示词 KV 缓存不被淘汰。
   - **Blackwell sm_120/121 FLA 内核优化**：降低 shmem 判定门限至 99KiB，修复 chunk_delta_h 的 tl.dot 竞态。

---

## 快速使用

### 方式一：原生 Nix Flake FHS 运行（推荐）

#### 1. 初始化安装环境（下载官方 nightly cu130 二进制并自动应用补丁）
```bash
nix run .#setup
# 或者进入开发环境后直接运行：
nix develop
vllm-setup
```

#### 2. 启动推理服务
```bash
nix run .#serve -- --model-dir /path/to/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid
# 或者：
vllm-serve --model-dir /path/to/model-dir --port 8000
```

#### 3. API 冒烟测试
```bash
nix run .#smoke-test
```

---

### 方式二：Docker 容器化运行

如果更倾向于使用容器（类似于您配置中的 `sglang` 服务）：

```bash
# 1. 构建镜像（已自动打好所有优化补丁）
nix run .#docker-build

# 2. 启动服务
MODEL_DIR=/path/to/checkpoint nix run .#docker
```

---

### 方式三：作为 NixOS Module 服务接入系统配置

在 `/etc/nixos/flake.nix` 中引入本 flake：

```nix
{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-vllm.url = "path:/home/losses/Development/nix-vllm";
  };

  outputs = { self, nixpkgs, nix-vllm, ... }: {
    nixosConfigurations.nixos = nixpkgs.lib.nixosSystem {
      system = "x86_64-linux";
      modules = [
        ./configuration.nix
        nix-vllm.nixosModules.default
        {
          services.vllm-qwen = {
            enable = true;
            package = nix-vllm.packages.x86_64-linux.default;
            modelDir = "/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid";
            port = 8000;
            mtp = 3;
            enablePrefixCaching = true;
            gpuMemoryUtilization = "0.85";
            pleCpuOffload = true;
          };
        }
      ];
    };
  };
}
```

---

## 常见环境变量与参数控制

| 变量 / 参数 | 默认值 | 说明 |
|---|---|---|
| `MODEL_DIR` | (必填) | 预处理后的模型权重目录 |
| `PORT` | `8000` | 监听端口 |
| `MTP` | `3` | Multi-Token Prediction 投机预测深度 (0 为关闭) |
| `PREFIX_CACHE` | `1` | 启用 Prefix Caching |
| `GPU_MEM` | `0.85` | GPU 显存预分配比例 |
| `KV_BYTES` | (可选) | 显式限制 KV 显存大小（如 `20g`） |
| `VLLM_PLE_CPU_OFFLOAD` | `1` | 将 PLE n-gram 词表卸载至 Host RAM (节约显存) |
| `PIN_PROMPT` | `""` | 需要常驻显存不被淘汰的系统提示词子串 |
| `TOOL_PARSER` | `qwen3_xml` | Tool Call 格式解析器 |

