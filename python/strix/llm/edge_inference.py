# SPDX-License-Identifier: Apache-2.0

"""Edge inference engine for on-device language models.

Targets hardware platforms commonly used in autonomy and robotics deployments:

NVIDIA Jetson Orin (AGX / NX / Nano)
    - llama.cpp with CUDA backend
    - 8-64 GB unified memory
    - INT4/INT8 quantization for 3B models
    - Target: <100ms inference for short-horizon parsing

Intel NUC / x86 edge servers
    - ONNX Runtime with OpenVINO EP
    - INT8 quantization with Neural Compressor
    - Target: <200ms inference

Raspberry Pi 5 / ARM SBCs
    - llama.cpp CPU-only (ARM NEON)
    - Heavily quantized (Q2_K / Q3_K) 1-3B models
    - Target: <500ms inference (degraded capability)

The edge inference engine provides:
    1. Model format detection and automatic backend selection
    2. Warm-up and latency benchmarking
    3. Graceful degradation when hardware is insufficient
    4. Memory-mapped model loading to minimize startup time
    5. Batch inference for multi-platform decision requests

Integration with strix.llm.autonomy_llm:
    The optional public LLM interface may delegate to EdgeInference when
    running on embedded hardware. The factory function
    ``create_inference()`` auto-detects the platform and returns the
    appropriate backend.
"""

from __future__ import annotations

import logging
import platform
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger("strix.llm.edge_inference")


class InferenceBackend(Enum):
    """Supported inference backends."""

    LLAMA_CPP_CUDA = auto()
    LLAMA_CPP_CPU = auto()
    ONNX_OPENVINO = auto()
    ONNX_CPU = auto()
    STUB = auto()


@dataclass(frozen=True)
class HardwareProfile:
    """Detected hardware capabilities."""

    platform_name: str = "unknown"
    arch: str = "unknown"
    cpu_count: int = 1
    memory_gb: float = 0.0
    has_cuda: bool = False
    cuda_memory_gb: float = 0.0
    has_openvino: bool = False
    recommended_backend: InferenceBackend = InferenceBackend.STUB
    max_model_size_b: int = 0  # max parameter count


@dataclass
class InferenceResult:
    """Result from a single inference call."""

    text: str = ""
    tokens_generated: int = 0
    latency_ms: float = 0.0
    tokens_per_second: float = 0.0


class EdgeInference:
    """Edge inference engine -- manages model loading and inference.

    Stub implementation.  Production code will integrate llama.cpp
    Python bindings or ONNX Runtime.
    """

    def __init__(self, model_path: str = "", backend: InferenceBackend = InferenceBackend.STUB) -> None:
        self._model_path = model_path
        self._backend = backend
        self._loaded = False
        self._warmup_latency_ms: Optional[float] = None
        logger.info("EdgeInference created: backend=%s", backend.name)

    def detect_hardware(self) -> HardwareProfile:
        """Auto-detect hardware capabilities.

        Returns a HardwareProfile with recommended backend.
        """
        import os

        arch = platform.machine()
        cpu_count = os.cpu_count() or 1

        # Memory detection (Linux)
        memory_gb = 0.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        memory_gb = kb / (1024 * 1024)
                        break
        except (OSError, ValueError):
            pass

        # CUDA detection
        has_cuda = False
        cuda_mem = 0.0
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                has_cuda = True
                cuda_mem = float(result.stdout.strip().split("\n")[0]) / 1024
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        # Determine recommended backend
        if has_cuda and cuda_mem >= 4.0:
            backend = InferenceBackend.LLAMA_CPP_CUDA
            max_params = int(cuda_mem * 1e9)  # rough: 1B params per GB
        elif "x86" in arch or "amd64" in arch.lower():
            backend = InferenceBackend.ONNX_CPU
            max_params = int(memory_gb * 0.3 * 1e9)
        elif "aarch64" in arch or "arm" in arch:
            backend = InferenceBackend.LLAMA_CPP_CPU
            max_params = int(memory_gb * 0.2 * 1e9)
        else:
            backend = InferenceBackend.STUB
            max_params = 0

        # Platform name heuristic
        pname = "unknown"
        if has_cuda:
            pname = "NVIDIA GPU system"
        elif "aarch64" in arch:
            pname = "ARM64 SBC" if memory_gb < 16 else "Jetson/ARM server"
        elif "x86" in arch:
            pname = "x86 edge server"

        profile = HardwareProfile(
            platform_name=pname,
            arch=arch,
            cpu_count=cpu_count,
            memory_gb=round(memory_gb, 1),
            has_cuda=has_cuda,
            cuda_memory_gb=round(cuda_mem, 1),
            has_openvino=False,  # TODO: detect OpenVINO
            recommended_backend=backend,
            max_model_size_b=max_params,
        )

        logger.info(
            "Hardware detected: %s (%s), %.1f GB RAM, CUDA=%s, recommended=%s",
            profile.platform_name,
            profile.arch,
            profile.memory_gb,
            profile.has_cuda,
            profile.recommended_backend.name,
        )
        return profile

    def load_model(self) -> bool:
        """Load the model into memory using the configured backend.

        Returns True if successful.
        """
        if not self._model_path:
            logger.warning("No model path specified -- running in stub mode")
            self._loaded = True
            return True

        logger.info("Loading model: %s (backend=%s)", self._model_path, self._backend.name)
        # TODO: Implement actual model loading
        # if self._backend == InferenceBackend.LLAMA_CPP_CUDA:
        #     from llama_cpp import Llama
        #     self._model = Llama(model_path=self._model_path, n_gpu_layers=-1)
        self._loaded = True
        return True

    def warmup(self, n_runs: int = 3) -> float:
        """Run warmup inferences and return average latency in ms."""
        if not self._loaded:
            logger.warning("Model not loaded -- cannot warm up")
            return 0.0

        latencies = []
        for _ in range(n_runs):
            t0 = time.monotonic()
            self.infer("warmup test query")
            latencies.append((time.monotonic() - t0) * 1000)

        avg = sum(latencies) / len(latencies)
        self._warmup_latency_ms = avg
        logger.info("Warmup complete: %.1fms avg latency (%d runs)", avg, n_runs)
        return avg

    def infer(self, prompt: str, max_tokens: int = 128) -> InferenceResult:
        """Run inference on a single prompt.

        Returns an InferenceResult with generated text and timing.
        """
        t0 = time.monotonic()

        # TODO: Implement actual inference
        # if self._backend in (InferenceBackend.LLAMA_CPP_CUDA, InferenceBackend.LLAMA_CPP_CPU):
        #     output = self._model(prompt, max_tokens=max_tokens)
        #     text = output["choices"][0]["text"]

        text = f"[stub response to: {prompt[:50]}...]"
        tokens = len(text.split())
        latency = (time.monotonic() - t0) * 1000

        return InferenceResult(
            text=text,
            tokens_generated=tokens,
            latency_ms=latency,
            tokens_per_second=tokens / max(latency / 1000, 1e-6),
        )

    def unload(self) -> None:
        """Release model from memory."""
        self._loaded = False
        logger.info("Model unloaded")


def create_inference(model_path: str = "") -> EdgeInference:
    """Factory: auto-detect hardware and create the best inference engine.

    Usage::

        engine = create_inference("/models/phi-3-mini-q4.gguf")
        engine.load_model()
        engine.warmup()
        result = engine.infer("Classify threat at bearing 045")
    """
    engine = EdgeInference(model_path=model_path)
    profile = engine.detect_hardware()
    engine._backend = profile.recommended_backend
    return engine
