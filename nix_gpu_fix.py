"""
NixOS GPU fix: monkey-patch subprocess for missing /sbin/ldconfig
and disable triton JIT (needs C compiler on NixOS).

Import this BEFORE torch/transformers.
"""

import os as _os
import subprocess as _subprocess

# ── 1. Disable triton JIT (avoids "Failed to find C compiler") ──
_os.environ.setdefault("TRITON_INTERPRET", "1")
_os.environ.setdefault("TRITON_LIBCUDA_PATH", "/run/opengl-driver/lib/libcuda.so.1")

# ── 2. Fake /sbin/ldconfig if missing (NixOS has no ldconfig) ──
if not _os.path.exists("/sbin/ldconfig"):
    _ORIGINAL_RUN = _subprocess.run
    _ORIGINAL_CHECK_OUTPUT = _subprocess.check_output

    _FAKE_LDCONFIG = """\
\tlibcuda.so.1 (libc6,x86-64) => /run/opengl-driver/lib/libcuda.so.1
\tlibcuda.so (libc6,x86-64) => /run/opengl-driver/lib/libcuda.so
\tlibcudart.so.12 (libc6,x86-64) => /run/opengl-driver/lib/libcudart.so.12
"""

    def _patched_run(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and cmd[0] == "/sbin/ldconfig":
            stdout = _FAKE_LDCONFIG.encode() if "-p" in cmd else b""
            return _subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr=b"")
        return _ORIGINAL_RUN(*args, **kwargs)

    def _patched_check_output(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, (list, tuple)) and len(cmd) > 0 and cmd[0] == "/sbin/ldconfig":
            return _FAKE_LDCONFIG.encode()
        return _ORIGINAL_CHECK_OUTPUT(*args, **kwargs)

    _subprocess.run = _patched_run
    _subprocess.check_output = _patched_check_output
