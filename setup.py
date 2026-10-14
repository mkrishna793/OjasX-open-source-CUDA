"""Universal multi-hardware build script for OjasX C++17 / Objective-C++ TileHAL Kernels (`ojasx2.csrc`).

Supports:
  - macOS (Apple Silicon / Intel): compiles `ojasx_tile_kernels.cpp` + `ojasx_tilelang_metal.mm`
    with `-framework Accelerate -framework Metal -framework Foundation` into `libojasx_tile_kernels.dylib`.
  - Linux (x86_64 AVX2/AVX-512, ARM64 NEON, NVIDIA CUDA, AMD ROCm, Intel XPU): compiles
    `ojasx_tile_kernels.cpp` + `-x c++ ojasx_tilelang_metal.mm` into `libojasx_tile_kernels.so`.
  - Windows (x86_64 / ARM64): compiles both C++17 translation units into `libojasx_tile_kernels.pyd`
    (with automatic graceful fallback to portable PyTorch operators if a native compiler is absent).
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution


def build_native_tile_library():
    root = Path(__file__).resolve().parent
    csrc_dir = root / "ojasx2" / "csrc"
    tilelang_src = root / "third_party" / "tilelang" / "src"

    try:
        import torch
    except ImportError:
        sys.stderr.write(
            "[OjasX setup.py] torch is absent in the build environment; using PyTorch fallback. "
            "For a native build, install torch and use --no-build-isolation.\n"
        )
        return None

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
    machine = platform.machine().lower()

    abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={int(getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 1))}"

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
        simd_flags = ["-mavx2", "-mfma"] if machine in ("x86_64", "amd64") else []
        cmd = [
            os.environ.get("CXX", "g++"),
            "-std=c++17",
            "-O3",
            "-shared",
            "-fPIC",
            abi_flag,
            *simd_flags,
            str(csrc_dir / "ojasx_tile_kernels.cpp"),
            "-x",
            "c++",
            str(csrc_dir / "ojasx_tilelang_metal.mm"),
            *[f"-I{d}" for d in inc_dirs],
            f"-L{lib_dir}",
            f"-Wl,-rpath,{lib_dir}",
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
            "-o",
            str(tmp_lib),
        ]
    elif system == "Windows":
        out_lib = csrc_dir / "libojasx_tile_kernels.pyd"
        tmp_lib = csrc_dir / "libojasx_tile_kernels.pyd.tmp"
        cmd = [
            os.environ.get("CXX", "clang++"),
            "-std=c++17",
            "-O3",
            "-shared",
            str(csrc_dir / "ojasx_tile_kernels.cpp"),
            "-x",
            "c++",
            str(csrc_dir / "ojasx_tilelang_metal.mm"),
            *[f"-I{d}" for d in inc_dirs],
            f"-L{lib_dir}",
            "-ltorch",
            "-ltorch_cpu",
            "-lc10",
            "-o",
            str(tmp_lib),
        ]
    else:
        return None

    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp_lib, out_lib)
        return out_lib
    except Exception as exc:
        if tmp_lib.exists():
            tmp_lib.unlink()
        sys.stderr.write(f"[OjasX setup.py] Native C++ build skipped/fallback active: {exc}\n")
        return None


class BuildPyWithNativeKernels(build_py):
    def run(self) -> None:
        self.native_library = None
        if os.environ.get("OJASX_DISABLE_NATIVE", "0") != "1" and (
            os.environ.get("OJASX_REBUILD_NATIVE", "1") == "1"
        ):
            self.native_library = build_native_tile_library()
        super().run()
        for binary in (Path(self.build_lib) / "ojasx2" / "csrc").glob("*"):
            if binary.suffix in (".dylib", ".so", ".pyd", ".dll") and (
                self.native_library is None or binary.name != self.native_library.name
            ):
                binary.unlink()

    def find_data_files(self, package, src_dir):
        files = super().find_data_files(package, src_dir)
        built = getattr(self, "native_library", None)
        return [
            p
            for p in files
            if Path(p).suffix not in (".dylib", ".so", ".pyd", ".dll")
            or (built is not None and Path(p).resolve() == built.resolve())
        ]


class OjasXDistribution(Distribution):
    def has_ext_modules(self):
        return os.environ.get("OJASX_REBUILD_NATIVE") == "1"


if __name__ == "__main__":
    setup(
        packages=find_packages(
            where=".",
            include=["ojasx2", "ojasx2.*"],
            exclude=["third_party*", "tests*", "reports*", "rocq*", "lean*"],
        ),
        cmdclass={"build_py": BuildPyWithNativeKernels},
        distclass=OjasXDistribution,
    )
