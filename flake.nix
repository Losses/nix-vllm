{
  description = "vLLM with Qwen3.8-Flash-Next / Qwen4Exp optimizations on Blackwell / NixOS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" ];
    in
    flake-utils.lib.eachSystem supportedSystems (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };

        vllmFhs = pkgs.callPackage ./pkgs/vllm-fhs.nix { };

        vllmDocker = pkgs.writeShellScriptBin "vllm-docker-serve" ''
          exec ${./scripts/serve-docker.sh} "$@"
        '';

        vllmDockerBuild = pkgs.writeShellScriptBin "vllm-docker-build" ''
          cd ${./.}
          exec ${pkgs.docker}/bin/docker build -t qwen-vllm-blackwell .
        '';

      in {
        packages = {
          default = vllmFhs.serve;
          serve = vllmFhs.serve;
          fhs = vllmFhs;
          setup = vllmFhs.setup;
          hf = vllmFhs.hf;
          huggingface-cli = vllmFhs.huggingface-cli;
          smoke-test = vllmFhs.smokeTest;
          docker = vllmDocker;
          docker-build = vllmDockerBuild;
        };

        apps = {
          default = {
            type = "app";
            program = "${vllmFhs.serve}/bin/vllm-serve";
            meta.description = "Serve vLLM natively in Nix FHS environment";
          };
          serve = {
            type = "app";
            program = "${vllmFhs.serve}/bin/vllm-serve";
            meta.description = "Serve vLLM natively in Nix FHS environment";
          };
          setup = {
            type = "app";
            program = "${vllmFhs.setup}/bin/vllm-setup";
            meta.description = "Bootstrap official vLLM nightly binary wheels with CUDA 13.0 and patches";
          };
          hf = {
            type = "app";
            program = "${vllmFhs.hf}/bin/hf";
            meta.description = "Hugging Face CLI tool";
          };
          huggingface-cli = {
            type = "app";
            program = "${vllmFhs.huggingface-cli}/bin/huggingface-cli";
            meta.description = "Hugging Face CLI tool";
          };
          smoke-test = {
            type = "app";
            program = "${vllmFhs.smokeTest}/bin/vllm-smoke-test";
            meta.description = "Run API smoke test against local vLLM server";
          };
          docker = {
            type = "app";
            program = "${vllmDocker}/bin/vllm-docker-serve";
            meta.description = "Serve Qwen3.8-Flash-Next in Docker container with PLE CPU offload";
          };
          docker-build = {
            type = "app";
            program = "${vllmDockerBuild}/bin/vllm-docker-build";
            meta.description = "Build Blackwell-patched Docker image";
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pkgs.python312
            pkgs.uv
            pkgs.docker
            pkgs.curl
            pkgs.jq
            pkgs.git
            pkgs.which
            vllmFhs
          ];
          shellHook = ''
            echo "================================================================"
            echo "  vLLM Qwen3.8-Flash-Next / Qwen4Exp (Blackwell / NixOS) Shell  "
            echo "================================================================"
            echo "Commands:"
            echo "  vllm-serve        - Launch vLLM server (Nix FHS native)"
            echo "  vllm-docker-serve - Launch via Blackwell container (with PLE CPU offload)"
            echo "  vllm-setup        - Install/update binary wheels & patches in venv"
            echo "  hf / huggingface-cli - Hugging Face CLI (auth, download)"
            echo "  vllm-smoke-test   - Run API test request"
            echo "================================================================"
          '';
        };
      }) // {
        nixosModules = {
          default = import ./nixos-module.nix;
          vllm-qwen = import ./nixos-module.nix;
        };
      };
}
