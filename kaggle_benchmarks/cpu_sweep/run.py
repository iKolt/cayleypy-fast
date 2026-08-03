"""cayleypy-fast CPU SWEEP kernel: beam-width scaling on cube333 iterated_batched.

bw in {256, 1e3, 1e4, 1e5}, 30 steps, 2 measured runs each, legacy vs engine.
The bw=256 row is the negative control for the size gate: 256*12 = 3072 < 2^16
and 256 < 512, so the engine must NOT engage there (expected speedup ~= 1.0,
i.e. legacy vs legacy). bw=1e3 clearly exceeds both thresholds.

Write: cpu_bench_sweep_result.json. Kernel id: ivankolt/cayleypy-cpu-sweep.
"""

import json
import statistics
import subprocess
import sys
import time

_CAYLEYPY_SHA = "0b7e109ff2d379fb2509f9fb14f7686e64453503"
_CAYLEYPY_FAST_REF = "0732d801addce60d0d2eafeb6efe51aa5b5c61f7"  # runaway-beam fix

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
assert create_engine(graph, "iterated_batched", None) is not None, "engine unavailable for cube333"

_N_GENS = graph.definition.n_generators


def _engine_expected(bw: int) -> bool:
    """Mirror of the size gate: engine iff bw*n_generators >= 2^16 or bw >= 512."""
    return bw * _N_GENS >= 2**16 or bw >= 512


# (name, beam_width, history_depth, max_steps, measured_runs)
CONFIGS = [(f"cube333_iterated_batched_bw{bw}", bw, 2, 30, 2) for bw in (256, 10**3, 10**4, 10**5)]


def run_timed(beam_search_fn, start_state, beam_width, runs):
    kwargs = {"beam_mode": "iterated_batched", "beam_width": beam_width, "max_steps": 30, "history_depth": 2}
    times = []
    last = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last = beam_search_fn(start_state=start_state, **kwargs)
        times.append(time.perf_counter() - t0)
    return {
        "min": min(times),
        "mean": statistics.mean(times),
        "path_found": last.path_found,
        "path_length": last.path_length,
    }, last


results = {}
for name, bw, hd, steps, runs in CONFIGS:
    legacy_stats, legacy_result = run_timed(graph.beam_search, _CUBE333_START, bw, runs)
    engine_stats, engine_result = run_timed(wrapped.beam_search, _CUBE333_START, bw, runs)

    assert engine_result.path_found == legacy_result.path_found, f"{name}: path_found mismatch"
    speedup = legacy_stats["mean"] / engine_stats["mean"]
    engine_expected = _engine_expected(bw)
    results[name] = {
        "graph": "cube333",
        "mode": "iterated_batched",
        "history_depth": hd,
        "beam_width": bw,
        "max_steps": steps,
        "legacy": legacy_stats,
        "engine": engine_stats,
        "speedup_mean": speedup,
        "engine_expected": engine_expected,
        "engine_path_length": engine_result.path_length,
        "legacy_path_length": legacy_result.path_length,
    }
    print(
        f"{name}: legacy={legacy_stats['mean']:.2f}s engine={engine_stats['mean']:.2f}s "
        f"speedup={speedup:.2f}x (engine_expected={engine_expected})",
        flush=True,
    )

    # Negative control: below the gate the two paths are identical (legacy vs
    # legacy), so the speedup must be ~= 1.0 (0.7-1.4 tolerance).
    if not engine_expected:
        assert 0.7 < speedup < 1.4, f"{name}: expected legacy-vs-legacy at bw={bw}, got speedup {speedup:.2f}"

results["meta"] = {
    "torch": torch.__version__,
    "device": "cpu",
    "kernel": "cpu-sweep",
    "cayleypy_sha": _CAYLEYPY_SHA,
    "cayleypy_fast_ref": _CAYLEYPY_FAST_REF,
}
with open("cpu_bench_sweep_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Wrote cpu_bench_sweep_result.json", flush=True)
