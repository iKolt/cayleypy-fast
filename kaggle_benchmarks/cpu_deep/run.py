"""cayleypy-fast CPU DEEP kernel: cube333 iterated/iterated_batched at bw=2^18.

Mirrors the GPU deep tier (cayleypy AGENTS.md section 10): 100 steps with a
hashed MITM neighbourhood of radius 3, one measured run. Legacy at bw=2^18 is
the slow reference; the engine is expected to dominate. CPU-only (no GPU pin
needed).

Write: cpu_bench_deep_result.json. Kernel id: ivankolt/cayleypy-cpu-deep.
"""

import json
import statistics
import subprocess
import sys
import time

_CAYLEYPY_SHA = "0b7e109ff2d379fb2509f9fb14f7686e64453503"
_CAYLEYPY_FAST_REF = "45c5985986c10f520113863547bc8fae4f82fc32"  # runaway-beam fix

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "h5py", "numba", "kagglehub"])
subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        f"git+https://github.com/cayleypy/cayleypy.git@{_CAYLEYPY_SHA}",
        f"git+https://github.com/iKolt/cayleypy-fast.git@{_CAYLEYPY_FAST_REF}",
    ]
)

import torch  # noqa: E402

from cayleypy import CayleyGraph, Puzzles  # noqa: E402

import cayleypy_fast  # noqa: E402
from cayleypy_fast.engine import create_engine  # noqa: E402

assert cayleypy_fast.run_probe().ok, cayleypy_fast.run_probe().problems

print(f"torch={torch.__version__}, cpu_count={__import__('os').cpu_count()}", flush=True)

# fmt: off
_CUBE333_START = [3, 3, 1, 0, 0, 2, 1, 0, 4, 4, 2, 0, 5, 1, 4, 5, 5, 3,
                  3, 3, 5, 0, 2, 5, 4, 2, 0, 2, 4, 0, 2, 3, 3, 2, 5, 5,
                  2, 1, 0, 0, 4, 1, 2, 4, 1, 4, 3, 5, 1, 5, 1, 3, 4, 1]
# fmt: on

graph = CayleyGraph(Puzzles.rubik_cube(3, metric="QTM"), dtype=torch.int8, bit_encoding_width=None)
wrapped = cayleypy_fast.wrap(graph)

# (name, mode, history_depth, beam_width, max_steps, mitm_radius, measured_runs)
CONFIGS = [
    ("cube333_iterated_deep", "iterated", 2, 2**18, 100, 3, 1),
    ("cube333_iterated_batched_deep", "iterated_batched", 2, 2**18, 100, 3, 1),
]


def run_timed(beam_search_fn, start_state, beam_mode, history_depth, beam_width, max_steps, mitm, runs):
    kwargs = {
        "beam_mode": beam_mode,
        "beam_width": beam_width,
        "max_steps": max_steps,
        "history_depth": history_depth,
    }
    if mitm is not None:
        kwargs["hashed_neigbourhood"] = mitm
    times = []
    last = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last = beam_search_fn(start_state=start_state, **kwargs)
        times.append(time.perf_counter() - t0)
    stats = {
        "min": min(times),
        "mean": statistics.mean(times),
        "path_found": last.path_found,
        "path_length": last.path_length,
    }
    return stats, last


results = {}
for name, mode, hd, bw, steps, mitm, runs in CONFIGS:
    assert create_engine(graph, mode, None) is not None, "engine unavailable for cube333"

    legacy_stats, legacy_result = run_timed(graph.beam_search, _CUBE333_START, mode, hd, bw, steps, mitm, runs)
    engine_stats, engine_result = run_timed(wrapped.beam_search, _CUBE333_START, mode, hd, bw, steps, mitm, runs)

    assert engine_result.path_found == legacy_result.path_found, f"{name}: path_found mismatch"
    speedup = legacy_stats["mean"] / engine_stats["mean"]
    results[name] = {
        "graph": "cube333",
        "mode": mode,
        "history_depth": hd,
        "beam_width": bw,
        "max_steps": steps,
        "hashed_neigbourhood": mitm,
        "legacy": legacy_stats,
        "engine": engine_stats,
        "speedup_mean": speedup,
        "engine_path_length": engine_result.path_length,
        "legacy_path_length": legacy_result.path_length,
    }
    print(
        f"{name}: legacy={legacy_stats['mean']:.2f}s engine={engine_stats['mean']:.2f}s "
        f"speedup={speedup:.2f}x found={legacy_result.path_found}",
        flush=True,
    )

results["meta"] = {
    "torch": torch.__version__,
    "device": "cpu",
    "kernel": "cpu-deep",
    "cayleypy_sha": _CAYLEYPY_SHA,
    "cayleypy_fast_ref": _CAYLEYPY_FAST_REF,
}
with open("cpu_bench_deep_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Wrote cpu_bench_deep_result.json", flush=True)
