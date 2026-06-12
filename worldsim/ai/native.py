"""Native (Rust / C++) perceptron banks for division controllers.

The ``rust_ai`` crate and the ``cpp_ai`` helper both implement a
single-output online perceptron (``rp_*`` / ``sp_*``).  This module loads
whichever is available — Rust preferred, C++ as fallback — and composes
``n_outputs`` of those heads into a :class:`NativePerceptron` bank that
mirrors the :class:`~worldsim.ai.perceptron.SimplePerceptron` surface used by
ground divisions (``predict`` / ``predict_prob`` / ``train``).

The bank is only offered for the high-volume division controllers (thousands
of tiny 3–6 input forward passes per fifth) where the per-call FFI cost beats
array-backend dispatch overhead.  The Nelder-Mead role policies keep the
vectorised NumPy/CuPy implementation.

Enablement follows the project-wide Rust convention: automatic when a
toolchain is present (``WORLDSIM_USE_RUST`` / ``worldsim.ai.set_use_rust``),
silent fallback to :class:`SimplePerceptron` otherwise.
"""

from __future__ import annotations

import ctypes
import math
import multiprocessing as _mp
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence

from ..core.native import NATIVE_DIR, USE_RUST as _CORE_USE_RUST

_RUST_DIR = NATIVE_DIR / "rust_ai"
_RUST_LIB_PATH = _RUST_DIR / "target" / "release" / "librust_ai.so"
_CPP_DIR = NATIVE_DIR / "cpp_ai"
_CPP_LIB_PATH = _CPP_DIR / "libcpp_ai.so"

_rust_lib = None
_cpp_lib = None
_loaded = False

# Mirrors core.native.USE_RUST at import; ai.set_use_rust() flips it at runtime.
_ENABLED: bool = _CORE_USE_RUST


def set_enabled(flag: bool) -> None:
    """Enable or disable the native perceptron backend at runtime."""
    global _ENABLED
    _ENABLED = bool(flag)


def _build_rust() -> None:
    if _RUST_LIB_PATH.exists() or _mp.current_process().name != "MainProcess":
        return
    if shutil.which("cargo") is None or not (_RUST_DIR / "Cargo.toml").exists():
        return
    try:
        subprocess.run(["cargo", "build", "--release"], cwd=_RUST_DIR, check=False)
    except Exception:
        return


def _build_cpp() -> None:
    if _CPP_LIB_PATH.exists() or _mp.current_process().name != "MainProcess":
        return
    src = _CPP_DIR / "ai.cpp"
    if shutil.which("g++") is None or not src.exists():
        return
    cmd = ["g++", "-O2", "-fPIC", "-shared", str(src), "-o", str(_CPP_LIB_PATH)]
    try:
        subprocess.run(cmd, check=False)
    except Exception:
        return


def _bind_perceptron(lib, prefix: str) -> None:
    create = getattr(lib, f"{prefix}_create")
    create.argtypes = [ctypes.c_int]
    create.restype = ctypes.c_void_p
    destroy = getattr(lib, f"{prefix}_destroy")
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = None
    prob = getattr(lib, f"{prefix}_predict_prob")
    prob.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    prob.restype = ctypes.c_double
    train = getattr(lib, f"{prefix}_train")
    train.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_double,
    ]
    train.restype = None


def _load_libs() -> None:
    """Build (when toolchains exist) and load the native libraries once."""
    global _rust_lib, _cpp_lib, _loaded
    if _loaded:
        return
    _loaded = True
    _build_rust()
    if _RUST_LIB_PATH.exists():
        try:
            lib = ctypes.CDLL(str(_RUST_LIB_PATH))
            _bind_perceptron(lib, "rp")
            _rust_lib = lib
        except OSError:
            _rust_lib = None
    if _rust_lib is None:
        _build_cpp()
        if _CPP_LIB_PATH.exists():
            try:
                lib = ctypes.CDLL(str(_CPP_LIB_PATH))
                _bind_perceptron(lib, "sp")
                _cpp_lib = lib
            except OSError:
                _cpp_lib = None


def native_backend() -> tuple | None:
    """Return ``(lib, prefix)`` for the active native backend, or ``None``."""
    if not _ENABLED:
        return None
    _load_libs()
    if _rust_lib is not None:
        return _rust_lib, "rp"
    if _cpp_lib is not None:
        return _cpp_lib, "sp"
    return None


class NativePerceptron:
    """One-vs-rest bank of native perceptron heads.

    Each output is a single native head; ``predict_prob`` softmax-normalises
    the head activations so callers can keep treating the result as a
    probability vector.  ``train`` applies each target component as a binary
    label to its head — equivalent to the delta rule the crates implement.
    """

    def __init__(self, n_inputs: int, *, n_outputs: int = 3, lib=None, prefix: str = "rp") -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self._lib = lib
        self._prefix = prefix
        self._create = getattr(lib, f"{prefix}_create")
        self._destroy = getattr(lib, f"{prefix}_destroy")
        self._prob = getattr(lib, f"{prefix}_predict_prob")
        self._train = getattr(lib, f"{prefix}_train")
        self._heads = [self._create(n_inputs) for _ in range(n_outputs)]

    def __del__(self) -> None:  # pragma: no cover - destructor timing
        try:
            for head in self._heads:
                if head:
                    self._destroy(head)
            self._heads = []
        except Exception:
            pass

    # Helpers -----------------------------------------------------------
    def _prepare(self, inputs: Sequence[float]):
        vec = list(inputs)[: self.n_inputs]
        if len(vec) < self.n_inputs:
            vec.extend([0.0] * (self.n_inputs - len(vec)))
        return (ctypes.c_double * self.n_inputs)(*[float(v) for v in vec])

    # Public API (SimplePerceptron-compatible) ---------------------------
    def predict(self, inputs: Sequence[float]) -> List[float]:
        arr = self._prepare(inputs)
        return [float(self._prob(head, arr)) for head in self._heads]

    def predict_prob(self, inputs: Sequence[float]) -> List[float]:
        raw = self.predict(inputs)
        if not raw:
            return []
        max_logit = max(raw)
        exp_vals = [math.exp(v - max_logit) for v in raw]
        total = sum(exp_vals) or 1.0
        return [v / total for v in exp_vals]

    def train(
        self,
        batch_inputs: Iterable[Sequence[float]],
        batch_targets: Iterable[Sequence[float]],
        lr: float = 0.1,
    ) -> None:
        for inputs, targets in zip(batch_inputs, batch_targets):
            target_list = list(targets)
            if len(target_list) != self.n_outputs:
                continue
            arr = self._prepare(inputs)
            for head, target in zip(self._heads, target_list):
                self._train(head, arr, 1 if float(target) >= 0.5 else 0, float(lr))


def make_perceptron(n_inputs: int, *, n_outputs: int = 3):
    """Return the best available small perceptron controller.

    Native (Rust, then C++) when enabled and loadable; otherwise the
    vectorised :class:`~worldsim.ai.perceptron.SimplePerceptron`.
    """
    backend = native_backend()
    if backend is not None:
        lib, prefix = backend
        try:
            return NativePerceptron(n_inputs, n_outputs=n_outputs, lib=lib, prefix=prefix)
        except Exception:
            pass
    from .perceptron import SimplePerceptron
    return SimplePerceptron(n_inputs, n_outputs=n_outputs)
