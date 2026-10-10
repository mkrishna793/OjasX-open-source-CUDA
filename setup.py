"""Portable build script for OjasX C++17 / Objective-C++ TileHAL Kernels (`ojasx2.csrc`).

Supports:
  - macOS (Apple Silicon / Intel): compiles `ojasx_tile_kernels.cpp` + `ojasx_tilelang_metal.mm`
    with `-framework Accelerate -framework Metal -framework Foundation` into `libojasx_tile_kernels.dylib`.
  - Linux / Windows: compiles portable C++17 `ojasx_tile_kernels.cpp` into `libojasx_tile_kernels.so` / `.pyd`,
    or installs cleanly with the pure-PyTorch `ojasx_hal` fallback when no C++ toolchain is present.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


def build_native_tile_library() -> None:
    root = Path(__file__).resolve().parent
    csrc_dir = root / "ojasx2" / "csrc"
    tilelang_src = root / "third_party" / "tilelang" / "src"

    try:
        import torch
    except ImportError:
        return

    import sysconfig

    torch_dir = Path(torch.__file__).resolve().parent
    py_inc = sysconfig.get_paths().get("include", "")
    inc_dirs = [
        str(torch_dir / "include"),
        str(torch_dir / "include" / "torch" / "csrc" / "api" / "include"),
        str(tilelang_src),
    ]
    if py_inc:
        inc_dirs.append(py_inc)
    lib_dir = str(torch_dir / "lib")
    system = platform.system()

    if system == "Darwin":
        out_lib = csrc_dir / "libojasx_tile_kernels.dylib"
        tmp_lib = csrc_dir / "libojasx_tile_kernels.dylib.tmp"
        sdk_path = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
        sdk_flags = ["-isysroot", str(sdk_path)] if sdk_path.exists() else []
        cmd = [
            "clang++",
            *sdk_flags,
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            "-fobjc-arc",
            "-Wno-deprecated-declarations",
            str(csrc_dir / "ojasx_tile_kernels.cpp"),
            str(csrc_dir / "ojasx_tilelang_metal.mm"),
            *[f"-I{d}" for d in inc_dirs],
            f"-L{lib_dir}",
            f"-Wl,-rpath,{lib_dir}",
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
            "-framework",
            "Accelerate",
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-o",
            str(tmp_lib),
        ]
    elif system == "Linux":
        out_lib = csrc_dir / "libojasx_tile_kernels.so"
        tmp_lib = csrc_dir / "libojasx_tile_kernels.so.tmp"
        cmd = [
            os.environ.get("CXX", "g++"),
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            str(csrc_dir / "ojasx_tile_kernels.cpp"),
            *[f"-I{d}" for d in inc_dirs],
            f"-L{lib_dir}",
            f"-Wl,-rpath,{lib_dir}",
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
            "-o",
            str(tmp_lib),
        ]
    else:
        return

    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp_lib, out_lib)
    except Exception as exc:
        if tmp_lib.exists():
            tmp_lib.unlink()
        sys.stderr.write(f"[OjasX setup.py] Native C++ build skipped/fallback active: {exc}\n")


class BuildPyWithNativeKernels(build_py):
    def run(self) -> None:
        if os.environ.get("OJASX_REBUILD_NATIVE", "0") == "1":
            build_native_tile_library()
        super().run()


if __name__ == "__main__":
    setup(cmdclass={"build_py": BuildPyWithNativeKernels})
