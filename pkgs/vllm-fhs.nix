{ lib
, stdenv
, buildFHSEnv
, writeShellScriptBin
, symlinkJoin
, python312
, uv
, coreutils
, bash
, curl
, git
, which
, procps
, zlib
, util-linux
, openssl
, libxml2
, bzip2
, xz
, numactl
, libGL
, glib
}:

let
  fhsBase = buildFHSEnv {
    name = "vllm-fhs-env";

    targetPkgs = pkgs: [
      python312
      uv
      coreutils
      bash
      curl
      git
      which
      procps
      zlib
      pkgs.stdenv.cc.cc.lib
      util-linux
      openssl
      libxml2
      bzip2
      xz
      numactl
      libGL
      glib
    ];

    profile = ''
      export PATH=/run/current-system/sw/bin:$PATH
      export LD_LIBRARY_PATH=/run/opengl-driver/lib:/run/opengl-driver-32/lib:$LD_LIBRARY_PATH
      export CUDA_HOME=/run/opengl-driver
    '';

    runScript = "bash";
  };

  vllmSetupScript = ./vllm-setup.sh;
  vllmServeScript = ./vllm-serve.sh;
  vllmSmokeTestScript = ./vllm-smoke-test.sh;

  vllmSetupBin = writeShellScriptBin "vllm-setup" ''
    exec ${fhsBase}/bin/vllm-fhs-env ${vllmSetupScript} "$@"
  '';

  vllmServeBin = writeShellScriptBin "vllm-serve" ''
    export VLLM_SETUP_CMD="${vllmSetupBin}/bin/vllm-setup"
    exec ${fhsBase}/bin/vllm-fhs-env ${vllmServeScript} "$@"
  '';

  vllmSmokeTestBin = writeShellScriptBin "vllm-smoke-test" ''
    exec ${fhsBase}/bin/vllm-fhs-env ${vllmSmokeTestScript} "$@"
  '';

  vllmBin = writeShellScriptBin "vllm" ''
    VLLM_DIR="''${VLLM_HOME:-$HOME/.local/share/nix-vllm}"
    VENV="$VLLM_DIR/venv"
    if [ ! -x "$VENV/bin/vllm" ]; then
      ${vllmSetupBin}/bin/vllm-setup
    fi
    exec ${fhsBase}/bin/vllm-fhs-env "$VENV/bin/vllm" "$@"
  '';

in symlinkJoin {
  name = "vllm-blackwell";
  paths = [ vllmServeBin vllmSetupBin vllmSmokeTestBin vllmBin fhsBase ];
  passthru = {
    fhs = fhsBase;
    setup = vllmSetupBin;
    serve = vllmServeBin;
    smokeTest = vllmSmokeTestBin;
  };
  meta = with lib; {
    description = "vLLM with Qwen3.8-Flash-Next / Qwen4Exp optimizations on Blackwell / NixOS";
    homepage = "https://github.com/vllm-project/vllm";
    license = licenses.asl20;
    platforms = [ "x86_64-linux" ];
    mainProgram = "vllm-serve";
  };
}
