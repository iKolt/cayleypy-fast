"""cayleypy-fast CPU HEAVY kernel: legacy vs engine, cube444/cube555 x 3 runs.

Companion of kaggle_benchmarks/cpu (full 11-row matrix). This kernel repeats
the two heavy rows with 3 measured runs each (vs 1 in the matrix kernel) for
statistical robustness, plus their iterated_batched counterparts.

Write: cpu_bench_heavy_result.json.

Run cycle: see kaggle_benchmarks/cpu/run.py (same CLI flow; kernel id
ivankolt/cayleypy-cpu-heavy).
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
_CUBE444_START = [1, 5, 0, 0, 2, 0, 2, 2, 0, 0, 2, 2, 4, 0, 0, 0, 5, 1,
                  4, 4, 1, 4, 1, 1, 2, 1, 1, 2, 4, 1, 4, 4, 1, 5, 5, 2,
                  2, 5, 5, 2, 3, 3, 2, 0, 1, 1, 1, 1, 3, 3, 4, 2, 3, 3,
                  2, 2, 4, 3, 4, 4, 2, 5, 3, 2, 0, 0, 3, 3, 0, 0, 3, 4,
                  1, 0, 1, 1, 3, 3, 3, 0, 3, 0, 3, 5, 5, 5, 5, 5, 5, 4,
                  4, 5, 5, 4, 4, 5]
_CUBE555_START = [5, 5, 5, 5, 5, 2, 2, 2, 0, 2, 3, 0, 0, 0, 0, 2, 2, 2,
                  2, 2, 3, 0, 0, 0, 0, 2, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0,
                  1, 1, 1, 1, 3, 3, 3, 3, 3, 0, 1, 1, 1, 1, 2, 5, 2, 5,
                  4, 2, 5, 2, 5, 4, 2, 5, 2, 5, 4, 2, 4, 4, 5, 4, 2, 1,
                  2, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                  0, 1, 1, 1, 1, 5, 5, 5, 5, 4, 2, 0, 2, 0, 2, 0, 0, 4,
                  0, 4, 2, 0, 4, 0, 4, 2, 3, 3, 4, 3, 4, 1, 5, 3, 4, 1,
                  5, 5, 5, 5, 0, 4, 4, 4, 4, 1, 5, 5, 5, 5, 0, 4, 4, 5,
                  4, 0, 4, 4, 4]
# fmt: on

_graphs = {
    "cube444": CayleyGraph(Puzzles.rubik_cube(4, metric="QTM"), dtype=torch.int8, bit_encoding_width=None),
    "cube555": CayleyGraph(Puzzles.rubik_cube(5, metric="QTM"), dtype=torch.int8, bit_encoding_width=None),
}
_wrapped = {name: cayleypy_fast.wrap(graph) for name, graph in _graphs.items()}

# (name, graph, start, mode, history_depth, beam_width, max_steps, measured_runs)
CONFIGS = [
    ("cube444_iterated", "cube444", _CUBE444_START, "iterated", 2, 10**4, 30, 3),
    ("cube444_iterated_batched", "cube444", _CUBE444_START, "iterated_batched", 2, 10**4, 30, 3),
    ("cube555_iterated", "cube555", _CUBE555_START, "iterated", 2, 10**4, 30, 3),
    ("cube555_iterated_batched", "cube555", _CUBE555_START, "iterated_batched", 2, 10**4, 30, 3),
]


def run_timed(beam_search_fn, start_state, beam_mode, history_depth, beam_width, max_steps, runs):
    kwargs = {"beam_mode": beam_mode, "beam_width": beam_width, "max_steps": max_steps, "history_depth": history_depth}
    times = []
    last = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last = beam_search_fn(start_state=start_state, **kwargs)
        times.append(time.perf_counter() - t0)
    stats = {
        "min": min(times),
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        "path_found": last.path_found,
        "path_length": last.path_length,
    }
    return stats, last


results = {}
for name, graph_key, start_state, mode, hd, bw, steps, runs in CONFIGS:
    graph = _graphs[graph_key]
    assert create_engine(graph, mode, None) is not None, f"engine unavailable for {graph_key}"

    legacy_stats, legacy_result = run_timed(graph.beam_search, start_state, mode, hd, bw, steps, runs)
    engine_stats, engine_result = run_timed(_wrapped[graph_key].beam_search, start_state, mode, hd, bw, steps, runs)

    assert engine_result.path_found == legacy_result.path_found, f"{name}: path_found mismatch"
    speedup = legacy_stats["mean"] / engine_stats["mean"]
    results[name] = {
        "graph": graph_key,
        "mode": mode,
        "history_depth": hd,
        "beam_width": bw,
        "max_steps": steps,
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
    "kernel": "cpu-heavy",
    "cayleypy_sha": _CAYLEYPY_SHA,
    "cayleypy_fast_ref": _CAYLEYPY_FAST_REF,
}
with open("cpu_bench_heavy_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Wrote cpu_bench_heavy_result.json", flush=True)
