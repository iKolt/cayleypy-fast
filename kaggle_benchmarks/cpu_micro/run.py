"""cayleypy-fast CPU MICRO kernel: re-validate the two ms-scale regression rows.

lrx8_simple (0.91x) and lrx16_advanced (0.74x) were below 1.0 in cpu-bench v1
because per-call create_engine setup dominated searches that finish in 6-16
steps. engine@3cd0773 caches the engine per graph; this kernel re-measures
those exact rows (5 runs each so steady-state vs first-call setup is visible).

Write: cpu_bench_micro_result.json. Kernel id: ivankolt/cayleypy-cpu-micro.
"""

import json
import statistics
import subprocess
import sys
import time

_CAYLEYPY_SHA = "0b7e109ff2d379fb2509f9fb14f7686e64453503"
_CAYLEYPY_FAST_REF = "3cd0773dcd0942420dfe42e4895ca9898122d044"  # per-graph engine cache

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

from cayleypy import CayleyGraph, PermutationGroups  # noqa: E402

import cayleypy_fast  # noqa: E402
from cayleypy_fast.engine import create_engine  # noqa: E402

assert cayleypy_fast.run_probe().ok, cayleypy_fast.run_probe().problems

print(f"torch={__import__('torch').__version__}, cpu_count={__import__('os').cpu_count()}", flush=True)

_LRX8_START = [3, 5, 7, 1, 0, 6, 4, 2]
_LRX16_START = list(range(10, 16)) + list(range(0, 10))

_graphs = {
    "lrx8": CayleyGraph(PermutationGroups.lrx(8)),
    "lrx16": CayleyGraph(PermutationGroups.lrx(16)),
}
_wrapped = {name: cayleypy_fast.wrap(graph) for name, graph in _graphs.items()}

# (name, graph, start, mode, history_depth, beam_width, max_steps, measured_runs)
CONFIGS = [
    ("lrx8_simple", "lrx8", _LRX8_START, "simple", 0, 10**5, 30, 5),
    ("lrx16_advanced", "lrx16", _LRX16_START, "advanced", 2, 10**5, 20, 5),
]


def run_timed(beam_search_fn, start_state, beam_mode, history_depth, beam_width, max_steps, runs):
    kwargs = {"beam_mode": beam_mode, "beam_width": beam_width, "max_steps": max_steps, "history_depth": history_depth}
    times = []
    last = None
    for _ in range(runs):
        t0 = time.perf_counter()
        last = beam_search_fn(start_state=start_state, **kwargs)
        times.append(time.perf_counter() - t0)
    return {
        "times": times,
        "min": min(times),
        "mean": statistics.mean(times),
        "path_found": last.path_found,
        "path_length": last.path_length,
    }, last


results = {}
for name, graph_key, start_state, mode, hd, bw, steps, runs in CONFIGS:
    graph = _graphs[graph_key]
    assert create_engine(graph, mode, None) is not None, f"engine unavailable for {graph_key}"

    legacy_stats, legacy_result = run_timed(graph.beam_search, start_state, mode, hd, bw, steps, runs)
    engine_stats, engine_result = run_timed(_wrapped[graph_key].beam_search, start_state, mode, hd, bw, steps, runs)

    assert engine_result.path_found == legacy_result.path_found, f"{name}: path_found mismatch"
    results[name] = {
        "graph": graph_key,
        "mode": mode,
        "beam_width": bw,
        "legacy": legacy_stats,
        "engine": engine_stats,
        "speedup_min": legacy_stats["min"] / engine_stats["min"],
        "speedup_mean": legacy_stats["mean"] / engine_stats["mean"],
        "engine_path_length": engine_result.path_length,
        "legacy_path_length": legacy_result.path_length,
    }
    print(
        f"{name}: legacy(min={legacy_stats['min']*1000:.1f}ms mean={legacy_stats['mean']*1000:.1f}ms) "
        f"engine(min={engine_stats['min']*1000:.1f}ms mean={engine_stats['mean']*1000:.1f}ms) "
        f"speedup_min={results[name]['speedup_min']:.2f}x speedup_mean={results[name]['speedup_mean']:.2f}x",
        flush=True,
    )

results["meta"] = {
    "device": "cpu",
    "kernel": "cpu-micro",
    "cayleypy_sha": _CAYLEYPY_SHA,
    "cayleypy_fast_ref": _CAYLEYPY_FAST_REF,
}
with open("cpu_bench_micro_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Wrote cpu_bench_micro_result.json", flush=True)
