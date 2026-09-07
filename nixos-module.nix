{ config, lib, pkgs, ... }:

let
  cfg = config.services.vllm-qwen;
in {
  options.services.vllm-qwen = {
    enable = lib.mkEnableOption "vLLM service for Qwen3.8-Flash-Next / Qwen4Exp";

    package = lib.mkOption {
      type = lib.types.package;
      description = "The vllm package to use.";
    };

    modelDir = lib.mkOption {
      type = lib.types.str;
      description = "Path to the model directory.";
      example = "/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "Port to listen on.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Host to bind to.";
    };

    servedName = lib.mkOption {
      type = lib.types.str;
      default = "qwen";
      description = "Model name exposed to API.";
    };

    contextLength = lib.mkOption {
      type = lib.types.int;
      default = 262144;
      description = "Maximum context length.";
    };

    maxNumSeqs = lib.mkOption {
      type = lib.types.int;
      default = 8;
      description = "Maximum concurrent sequences.";
    };

    gpuMemoryUtilization = lib.mkOption {
      type = lib.types.str;
      default = "0.85";
      description = "Fraction of GPU memory to use.";
    };

    kvCacheDtype = lib.mkOption {
      type = lib.types.str;
      default = "fp8";
      description = "KV cache data type (fp8, auto, bfloat16).";
    };

    kvCacheMemoryBytes = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Explicit KV cache memory bytes (e.g. '20g').";
    };

    mtp = lib.mkOption {
      type = lib.types.int;
      default = 3;
      description = "Number of speculative tokens (MTP). Set to 0 to disable.";
    };

    enablePrefixCaching = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Enable prefix caching.";
    };

    pinPrompt = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      description = "Never-evict KV cache prompt substring.";
    };

    pleCpuOffload = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = "Offload PLE n-gram table to host RAM.";
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Extra arguments to pass to vllm serve.";
    };

    environment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      description = "Extra environment variables.";
    };
  };

  config = lib.mkIf cfg.enable {
    systemd.services.vllm-qwen = {
      description = "vLLM Qwen Serving Engine";
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];
      wantedBy = [ "multi-user.target" ];

      environment = {
        MODEL_DIR = cfg.modelDir;
        PORT = toString cfg.port;
        HOST = cfg.host;
        SERVED_NAME = cfg.servedName;
        CTX = toString cfg.contextLength;
        SEQS = toString cfg.maxNumSeqs;
        GPU_MEM = cfg.gpuMemoryUtilization;
        MTP = toString cfg.mtp;
        PREFIX_CACHE = if cfg.enablePrefixCaching then "1" else "0";
        VLLM_PLE_CPU_OFFLOAD = if cfg.pleCpuOffload then "1" else "0";
        KV_CACHE_DTYPE = cfg.kvCacheDtype;
      } // lib.optionalAttrs (cfg.kvCacheMemoryBytes != null) {
        KV_BYTES = cfg.kvCacheMemoryBytes;
      } // lib.optionalAttrs (cfg.pinPrompt != null) {
        PIN_PROMPT = cfg.pinPrompt;
      } // cfg.environment;

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/vllm-serve ${lib.escapeShellArgs cfg.extraArgs}";
        Restart = "on-failure";
        RestartSec = "10s";
        LimitMEMLOCK = "infinity";
      };
    };
  };
}
