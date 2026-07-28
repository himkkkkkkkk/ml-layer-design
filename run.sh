#!/usr/bin/env bash
# Run Qwen3 layer analysis with GPU acceleration
set -e
cd "$(dirname "$0")"

# NixOS library paths: nvidia drivers + gcc + zlib
export LD_LIBRARY_PATH="/run/opengl-driver/lib:/nix/store/hngmi01i8wgi25a0byrxcn4ysz5j79mw-gcc-15.2.0-lib/lib:/nix/store/dbz6pb9g67kpgpl95k8d85kzpxm1c32p-zlib-1.3.2/lib:$LD_LIBRARY_PATH"

# NixOS: triton JIT needs a C compiler
GCC_DIR="/nix/store/3wkpp7mjlh4qxij92iz99r43aifzgajd-gcc-15.2.0/bin"
export PATH="$GCC_DIR:$PATH"
export CC="$GCC_DIR/gcc"
export CXX="$GCC_DIR/g++"

if .venv/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo "GPU: $GPU_NAME"
    echo "Hint: use --dtype float16 to fit 4B model in 8GB VRAM"
else
    echo "GPU not available — running on CPU"
fi
echo ""

exec .venv/bin/python "$@"
