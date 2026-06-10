"""Loading and building of optional native accelerators (Rust / C++).

The compiled libraries live under :mod:`worldsim.native`.  Both helpers are
optional: when the toolchain or the built artefact is missing the rest of the
package silently falls back to NumPy/CuPy or pure Python implementations.

Modules that want the current library handles must read them dynamically
(``from worldsim.core import native`` then ``native._rust_lib``) because
:func:`set_use_rust` swaps them at runtime.
"""

from __future__ import annotations

import ctypes
import multiprocessing as _mp
import os
import shutil
import subprocess
from pathlib import Path

# Root of the native source tree: worldsim/native/
NATIVE_DIR = Path(__file__).resolve().parent.parent / "native"

_lib = None
_lib_path = NATIVE_DIR / "cpp_utils" / "libcpp_utils.so"
_rust_lib = None
_rust_lib_path = NATIVE_DIR / "rust_utils" / "librust_utils.so"


def _rust_available() -> bool:
    return shutil.which("cargo") is not None


USE_RUST = True
env_flag = os.getenv("WORLDSIM_USE_RUST", "auto").lower()
if env_flag in {"0", "false", "no"}:
    USE_RUST = False
elif env_flag in {"1", "true", "yes"}:
    USE_RUST = _rust_available()
else:  # auto
    USE_RUST = _rust_available()


def _build_cpp_utils() -> None:
    """Compile optional C++ helpers if missing."""
    if _mp.current_process().name != "MainProcess" or _lib_path.exists():
        return
    cpp_file = _lib_path.with_name("utils.cpp")
    if not cpp_file.exists():
        return
    cmd = ["g++", "-O2", "-fPIC", "-shared", str(cpp_file), "-o", str(_lib_path)]
    try:
        subprocess.run(cmd, check=False)
    except Exception:
        return


def _build_rust_utils() -> None:
    """Compile optional Rust helpers if missing."""
    if not USE_RUST or _mp.current_process().name != "MainProcess" or _rust_lib_path.exists():
        return
    cargo = shutil.which("cargo")
    if cargo is None:
        return
    src_dir = _rust_lib_path.parent
    try:
        subprocess.run([cargo, "build", "--release"], cwd=src_dir, check=False)
    except Exception:
        return
    built = src_dir / "target" / "release" / "librust_utils.so"
    if built.exists():
        try:
            shutil.copy(built, _rust_lib_path)
        except Exception:
            return


_build_cpp_utils()
if _lib_path.exists():
    try:
        _lib = ctypes.CDLL(str(_lib_path))
        _lib.cpp_logistic_growth.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        _lib.cpp_logistic_growth.restype = ctypes.c_double
        _lib.cpp_distance.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        _lib.cpp_distance.restype = ctypes.c_double
    except OSError:
        _lib = None

if USE_RUST:
    _build_rust_utils()
if USE_RUST and _rust_lib_path.exists():
    try:
        _rust_lib = ctypes.CDLL(str(_rust_lib_path))
        _rust_lib.ru_logistic_growth.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int]
        _rust_lib.ru_logistic_growth.restype = ctypes.c_double
        _rust_lib.ru_distance.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        _rust_lib.ru_distance.restype = ctypes.c_double
        _rust_lib.ru_polygon_area.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int]
        _rust_lib.ru_polygon_area.restype = ctypes.c_double
        _rust_lib.ru_polygon_centroid.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        _rust_lib.ru_polygon_centroid.restype = None
        _rust_lib.ru_logistic_growth_batch.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
        ]
        _rust_lib.ru_logistic_growth_batch.restype = None
    except OSError:
        _rust_lib = None


def set_use_rust(flag: bool) -> None:
    """Enable or disable Rust helpers at runtime."""
    global USE_RUST, _rust_lib
    USE_RUST = flag and _rust_available()
    if USE_RUST:
        _build_rust_utils()
        if _rust_lib_path.exists():
            try:
                _rust_lib = ctypes.CDLL(str(_rust_lib_path))
            except OSError:
                _rust_lib = None
    else:
        _rust_lib = None
