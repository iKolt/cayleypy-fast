"""cayleypy-fast CPU benchmark kernel: legacy vs engine, same process.

Runs the CPU benchmark matrix (mirroring cayleypy/algo/beam_search_benchmark.py)
twice per config — once legacy (graph.beam_search) and once with the fast engine
(patch-averse, via cayleypy_fast.wrap(graph).beam_search) — and writes
cpu_benchmark_result.json. Also asserts: engine actually engaged per graph, and
path_found parity legacy vs engine.

Environment: Kaggle CPU session (4 vCPU, 30 GB RAM, up to 12 h). No GPU, no
torch pin needed (the torch==2.5.1+cu121 pin in cayleypy's GPU baseline kernel
is only for Tesla P100 sm_60; CPU kernels run fine on the default torch).

Run cycle (PowerShell, see cayleypy AGENTS.md section 10 for kaggle CLI quirks):
    $env:KAGGLE_API_TOKEN="<key from ~/.kaggle/kaggle.json>"
    kaggle kernels push -p kaggle_benchmarks/cpu
    kaggle kernels status ivankolt/cayleypy-cpu-bench
    # download via the Python API (CLI output download has a charmap bug on Windows):
    python -c "from kaggle import KaggleApi; api=KaggleApi(); api.authenticate(); \
        api.kernels_output('ivankolt/cayleypy-cpu-bench', path='./kaggle_out', force=True, quiet=True)"
"""

import json
import statistics
import subprocess
import sys
import time

# cayleypy pinned to an immutable commit SHA (feature/beam-search-perf tip;
# includes Task 6 helpers + audit fixes, notably A1: _restore_path takes
# destination_state — the engine's probe requires this signature).
_CAYLEYPY_SHA = "0b7e109ff2d379fb2509f9fb14f7686e64453503"
# cayleypy-fast pinned to an immutable commit SHA: runaway-beam fix
# (_GlobalTopK first-chunk top-k cap) + size gate + pytest plugin.
_CAYLEYPY_FAST_REF = "45c5985986c10f520113863547bc8fae4f82fc32"

# Install cayleypy with --no-deps so pip does not re-resolve torch (cayleypy
# declares torch>=2.6; on the CPU kernel the preinstalled torch is fine).
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cayleypy import CayleyGraph, PermutationGroups, Puzzles, prepare_graph  # noqa: E402

import cayleypy_fast  # noqa: E402
from cayleypy_fast.engine import create_engine  # noqa: E402

assert cayleypy_fast.run_probe().ok, cayleypy_fast.run_probe().problems

print(f"torch={torch.__version__}, cpu_count={__import__('os').cpu_count()}", flush=True)

# --- Scenario constants (keep in sync with cayleypy/algo/beam_search_benchmark.py) ---
_LRX8_START = [3, 5, 7, 1, 0, 6, 4, 2]
_LRX16_START = list(range(10, 16)) + list(range(0, 10))
_LRX32_START = list(range(16, 32)) + list(range(0, 16))
# fmt: off
_CUBE333_START = [3, 3, 1, 0, 0, 2, 1, 0, 4, 4, 2, 0, 5, 1, 4, 5, 5, 3,
                  3, 3, 5, 0, 2, 5, 4, 2, 0, 2, 4, 0, 2, 3, 3, 2, 5, 5,
                  2, 1, 0, 0, 4, 1, 2, 4, 1, 4, 3, 5, 1, 5, 1, 3, 4, 1]
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
                  4, 0, 4, 4, 4, 4]
# fmt: on

_graphs = {
    "lrx8": CayleyGraph(PermutationGroups.lrx(8)),
    "lrx16": CayleyGraph(PermutationGroups.lrx(16)),
    "lrx32": CayleyGraph(PermutationGroups.lrx(32)),
    "cube222": CayleyGraph(prepare_graph("cube_2/2/2_6gensQTM")),
    "cube333": CayleyGraph(Puzzles.rubik_cube(3, metric="QTM"), dtype=torch.int8, bit_encoding_width=None),
    "cube444": CayleyGraph(Puzzles.rubik_cube(4, metric="QTM"), dtype=torch.int8, bit_encoding_width=None),
    "cube555": CayleyGraph(Puzzles.rubik_cube(5, metric="QTM"), dtype=torch.int8, bit_encoding_width=None),
}
_wrapped = {name: cayleypy_fast.wrap(graph) for name, graph in _graphs.items()}

np.random.seed(12345)
_CUBE222_START = _graphs["cube222"].random_walks(width=1, length=20)[0][-1]

# (name, graph, start, mode, history_depth, beam_width, max_steps, measured_runs)
# Beams are 10^4-10^5 per plan section "Goal" item 3; measured_runs=1 for the heavy rows.
CONFIGS = [
    ("lrx8_simple", "lrx8", _LRX8_START, "simple", 0, 10**5, 30, 2),
    ("lrx8_advanced", "lrx8", _LRX8_START, "advanced", 2, 10**5, 30, 2),
    ("lrx8_iterated", "lrx8", _LRX8_START, "iterated", 2, 10**5, 30, 2),
    ("lrx16_advanced", "lrx16", _LRX16_START, "advanced", 2, 10**5, 20, 2),
    ("lrx32_iterated", "lrx32", _LRX32_START, "iterated", 2, 10**5, 30, 2),
    ("cube222_simple", "cube222", _CUBE222_START, "simple", 0, 10**5, 20, 2),
    ("cube333_simple", "cube333", _CUBE333_START, "simple", 0, 10**5, 30, 2),
    ("cube333_advanced", "cube333", _CUBE333_START, "advanced", 2, 10**5, 30, 2),
    ("cube333_iterated", "cube333", _CUBE333_START, "iterated", 2, 10**5, 30, 2),
    ("cube444_iterated", "cube444", _CUBE444_START, "iterated", 2, 10**4, 30, 1),
    ("cube555_iterated", "cube555", _CUBE555_START, "iterated", 2, 10**4, 30, 1),
]


def run_timed(beam_search_fn, start_state, beam_mode, history_depth, beam_width, max_steps, runs):
    """Time `runs` beam searches; return (stats dict, last result)."""
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
        "path_found": last.path_found,
        "path_length": last.path_length,
    }
    return stats, last


results = {}
for name, graph_key, start_state, mode, hd, bw, steps, runs in CONFIGS:
    graph = _graphs[graph_key]
    # Guard: the engine must actually be engaged (otherwise we would benchmark legacy vs legacy).
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
    "cayleypy_sha": _CAYLEYPY_SHA,
    "cayleypy_fast_ref": _CAYLEYPY_FAST_REF,
}
with open("cpu_benchmark_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Wrote cpu_benchmark_result.json", flush=True)
