"""Portable-by-default and explicitly requested native OjasX builds.

Portable artifacts contain the PyTorch fallback and native source files, but no
prebuilt binary. Set OJASX_BUILD_NATIVE=1 in a Torch-compatible, non-isolated
build environment to compile the C++20 tile library. The native math extension
is currently supported on Darwin; other platforms build and report tile-only
native support. Native build failures are fatal.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sysconfig
from typing import Optional

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution


ROOT = Path(__file__).resolve().parent
_BINARY_SUFFIXES = {".dylib", ".so", ".pyd", ".dll"}


def native_requested() -> bool:
    value = os.environ.get("OJASX_BUILD_NATIVE", "0").strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off", ""):
        return False
    raise ValueError("OJASX_BUILD_NATIVE must be 0 or 1")


def _torch_build_paths():
    # ImportError is intentionally fatal in native mode.
    import torch

    torch_dir = Path(torch.__file__).resolve().parent
    include_dirs = [
        str(torch_dir / "include"),
        str(torch_dir / "include" / "torch" / "csrc" / "api" / "include"),
    ]
    python_include = sysconfig.get_paths().get("include", "")
    if python_include:
        include_dirs.append(python_include)
    return torch, torch_dir, include_dirs, str(torch_dir / "lib")


def build_native_tile_library() -> Path:
    csrc_dir = ROOT / "ojasx2" / "csrc"
    tilelang_src = ROOT / "third_party" / "tilelang" / "src"
    torch, torch_dir, include_dirs, lib_dir = _torch_build_paths()
    include_dirs.append(str(tilelang_src))
    system = platform.system()
    machine = platform.machine().lower()
    abi_flag = f"-D_GLIBCXX_USE_CXX11_ABI={int(getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', 1))}"

    if system == "Darwin":
        out_lib = csrc_dir / "libojasx_tile_kernels.dylib"
        sdk_path = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
        sdk_flags = ["-isysroot", str(sdk_path)] if sdk_path.exists() else []
        cmd = [
            "clang++", *sdk_flags, "-std=c++20", "-O3", "-shared", "-fPIC", "-fobjc-arc",
            "-Wno-deprecated-declarations",
            str(csrc_dir / "ojasx_tile_kernels.cpp"), str(csrc_dir / "ojasx_tilelang_metal.mm"),
            *[f"-I{directory}" for directory in include_dirs], f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}",
            "-ltorch", "-ltorch_cpu", "-lc10", "-framework", "Accelerate", "-framework", "Metal",
            "-framework", "Foundation",
        ]
    elif system == "Linux":
        out_lib = csrc_dir / "libojasx_tile_kernels.so"
        simd_flags = ["-mavx2", "-mfma"] if machine in ("x86_64", "amd64") else []
        cmd = [
            os.environ.get("CXX", "g++"), "-std=c++20", "-O3", "-shared", "-fPIC", abi_flag, *simd_flags,
            str(csrc_dir / "ojasx_tile_kernels.cpp"), "-x", "c++", str(csrc_dir / "ojasx_tilelang_metal.mm"),
            *[f"-I{directory}" for directory in include_dirs], f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}",
            "-ltorch", "-ltorch_cpu", "-lc10",
        ]
    elif system == "Windows":
        out_lib = csrc_dir / "libojasx_tile_kernels.pyd"
        cmd = [
            os.environ.get("CXX", "clang++"), "-std=c++20", "-O3", "-shared",
            str(csrc_dir / "ojasx_tile_kernels.cpp"), "-x", "c++", str(csrc_dir / "ojasx_tilelang_metal.mm"),
            *[f"-I{directory}" for directory in include_dirs], f"-L{lib_dir}",
            "-ltorch", "-ltorch_cpu", "-lc10",
        ]
    else:
        raise RuntimeError(f"Native tile builds are unsupported on {system}")

    tmp_lib = out_lib.with_suffix(out_lib.suffix + ".tmp")
    try:
        subprocess.run([*cmd, "-o", str(tmp_lib)], check=True)
        os.replace(tmp_lib, out_lib)
    finally:
        tmp_lib.unlink(missing_ok=True)
    return out_lib


def build_native_math_library() -> Optional[Path]:
    math_csrc_dir = ROOT / "ojasx2" / "math" / "csrc"
    system = platform.system()
    if system != "Darwin":
        # The math bindings contain Objective-C++/Metal sources. Do not pretend
        # that a tile-only Linux/Windows build includes this separate component.
        return None

    torch, torch_dir, include_dirs, lib_dir = _torch_build_paths()
    include_dirs.insert(0, str(math_csrc_dir))
    out_lib = math_csrc_dir / "ojasx_math_native.so"
    tmp_lib = out_lib.with_suffix(".so.tmp")
    sdk_path = Path("/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk")
    sdk_flags = ["-isysroot", str(sdk_path)] if sdk_path.exists() else []
    cmd = [
        "clang++", *sdk_flags, "-std=c++20", "-O3", "-shared", "-fPIC", "-fobjc-arc",
        "-undefined", "dynamic_lookup", "-Wno-deprecated-declarations",
        str(math_csrc_dir / "ojasx_math_bindings.cpp"), str(math_csrc_dir / "ojasx_math_metal.mm"),
        *[f"-I{directory}" for directory in include_dirs], f"-L{lib_dir}", f"-Wl,-rpath,{lib_dir}",
        "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10", "-framework", "Metal",
        "-framework", "Foundation", "-framework", "Accelerate", "-o", str(tmp_lib),
    ]
    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp_lib, out_lib)
    finally:
        tmp_lib.unlink(missing_ok=True)
    return out_lib


class BuildPyWithNativeKernels(build_py):
    def run(self) -> None:
        self.native_library = None
        self.native_math_library = None
        self.native_mode = native_requested()
        if self.native_mode:
            self.native_library = build_native_tile_library()
            self.native_math_library = build_native_math_library()
        super().run()
        built_names = {path.name for path in (self.native_library, self.native_math_library) if path is not None}
        for binary in (Path(self.build_lib) / "ojasx2").rglob("*"):
            if binary.suffix in _BINARY_SUFFIXES and binary.name not in built_names:
                binary.unlink()
        metadata = {
            "schema_version": 1,
            "mode": "native" if self.native_mode else "portable",
            "platform": platform.system(),
            "architecture": platform.machine(),
            "cpp_standard": "c++20" if self.native_mode else None,
            "tile_library": self.native_library.name if self.native_library else None,
            "math_library": self.native_math_library.name if self.native_math_library else None,
            "torch_version": None,
        }
        if self.native_mode:
            import torch
            metadata["torch_version"] = torch.__version__
        package_dir = Path(self.build_lib) / "ojasx2"
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "native_build.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def find_data_files(self, package, src_dir):
        files = super().find_data_files(package, src_dir)
        built = {
            path.resolve() for path in (
                getattr(self, "native_library", None), getattr(self, "native_math_library", None)
            ) if path is not None
        }
        return [
            path for path in files
            if Path(path).suffix not in _BINARY_SUFFIXES or Path(path).resolve() in built
        ]


class OjasXDistribution(Distribution):
    def has_ext_modules(self):
        return native_requested()


class BuildExtWithNativeKernels(build_ext):
    def run(self) -> None:
        self.native_library = None
        self.native_math_library = None
        self.native_mode = native_requested()
        if self.native_mode:
            self.native_library = build_native_tile_library()
            self.native_math_library = build_native_math_library()
        super().run()


if __name__ == "__main__":
    setup(
        name="ojasx-verified-core",
        version="0.16.0",
        py_modules=["ojasx"],
        packages=find_packages(
            where=".", include=["ojasx2", "ojasx2.*", "ojasx3", "ojasx3.*"],
            exclude=["third_party*", "tests*", "reports*", "rocq*", "lean*"],
        ),
        package_data={
            "ojasx2": [
                "codegen/*.yaml",
                "csrc/*.cpp",
                "csrc/*.mm",
                "csrc/*.h",
                "csrc/*.dylib",
                "csrc/*.so",
                "csrc/*.pyd",
                "native_build.json",
            ],
            "ojasx2.math.csrc": [
                "*.cpp",
                "*.hpp",
                "*.h",
                "*.mm",
                "*.so",
                "*.dylib",
                "*.pyd",
            ],
        },
        cmdclass={
            "build_py": BuildPyWithNativeKernels,
            "build_ext": BuildExtWithNativeKernels,
        },
        distclass=OjasXDistribution,
    )
