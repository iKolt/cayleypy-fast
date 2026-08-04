"""cayleypy-fast GPU perf kernel: legacy vs engine (Triton tier), same process, on Tesla P100.

Stages (each numbered section below):
  0. Smoke + test tripwire: Triton backend engaged, parity smoke (lrx8), and
     the CUDA-only pytest selection (tests/test_triton_tier.py,
     tests/test_nn_parity.py) executed in a clone of the repo at the pinned SHA.
  1. Quick mirror of the immutable GPU baseline: lrx8 + cube333, bw=1e5,
     max_steps=30, modes simple/advanced(hd=2)/iterated(hd=2), hamming,
     legacy-vs-engine in-process diff (warmup 2 + measured 3, cuda-synced).
  2. Deep mirror: cube333 iterated bw=2^18 mitm=3 return_path=False.
  2b. NN-predictor rows (the representative workload; hamming is
     de-prioritized by user decision): pretrained lrx16/lrx32 models (kagglehub)
     + seeded-random MLP on cube333/cube444. Parity policy: validate_path is a
     HARD assert; path_found equality is record-and-loudly-warn (fp reduction
     order is not batch-invariant on CUDA, so strict path_found equality is not
     a legal hard assert on NN rows).
  3. High-beam rows: cube333 + lrx32 iterated bw=2^24, engine-only absolute
     step time (legacy infeasible at 2^24: ~64x the 20-min deep baseline row).

Writes gpu_perf_result.json. Download via the Python API only (the
`kaggle kernels output` CLI has a Windows charmap bug).
"""

import gc
import json
import statistics
import subprocess
import sys
import time
import traceback

# --- Installs (order matters; mirrors the immutable baseline kernel) ---
# torch 2.5.1+cu121 FIRST: Kaggle's default torch crashes on Tesla P100 (sm_60).
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"]
)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "h5py", "numba", "kagglehub", "pytest"])

# cayleypy pinned to the dev pin (probe-verified target); --no-deps keeps the torch pin.
_CAYLEYPY_SHA = "0b7e109ff2d379fb2509f9fb14f7686e64453503"
# cayleypy-fast pinned to an immutable commit SHA (T3 Triton tier).
_CAYLEYPY_FAST_REF = "33197dbfba0dd30de57a2104b804920b3ee1b1f4"
_CLONE_DIR = "/kaggle/working/cayleypy-fast-clone"

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
from cayleypy.models.models import MlpModel, ModelConfig  # noqa: E402
from cayleypy.predictor import Predictor  # noqa: E402

import cayleypy_fast  # noqa: E402
from cayleypy_fast.engine import create_engine  # noqa: E402

assert torch.cuda.is_available(), "GPU perf kernel requires CUDA"
assert cayleypy_fast.run_probe().ok, cayleypy_fast.run_probe().problems

DEVICE_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"torch={torch.__version__}, device={DEVICE_NAME}, vram={TOTAL_VRAM_GB:.1f} GB", flush=True)

_KERNEL_T0 = time.time()
_KERNEL_BUDGET_SEC = 11.0 * 3600  # Kaggle GPU sessions cap at 12 h; keep reserve.

results: dict = {"meta": {"stages": {}}}


def budget_left() -> float:
    return _KERNEL_BUDGET_SEC - (time.time() - _KERNEL_T0)


def reset_mem():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# --- Scenario constants (keep in sync with kaggle_benchmarks/cpu/run.py) ---
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
# fmt: on


def make_graph(name: str) -> CayleyGraph:
    if name == "lrx8":
        return CayleyGraph(PermutationGroups.lrx(8))
    if name == "lrx16":
        return CayleyGraph(PermutationGroups.lrx(16))
    if name == "lrx32":
        return CayleyGraph(PermutationGroups.lrx(32))
    if name == "cube333":
        return CayleyGraph(Puzzles.rubik_cube(3, metric="QTM"), dtype=torch.int8, bit_encoding_width=None)
    if name == "cube444":
        return CayleyGraph(Puzzles.rubik_cube(4, metric="QTM"), dtype=torch.int8, bit_encoding_width=None)
    raise ValueError(name)


def run_timed(beam_search_fn, start_state, beam_mode, hd, beam_width, max_steps, predictor=None,
              return_path=False, warmup=2, measured=3):
    """Time `measured` beam searches (cuda-synced bracketing); return (stats, last result)."""
    kwargs = {
        "beam_mode": beam_mode,
        "beam_width": beam_width,
        "max_steps": max_steps,
        "history_depth": hd,
        "return_path": return_path,
    }
    if predictor is not None:
        kwargs["predictor"] = predictor
    for _ in range(warmup):
        beam_search_fn(start_state=start_state, **kwargs)
    times = []
    last = None
    for _ in range(measured):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        last = beam_search_fn(start_state=start_state, **kwargs)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return {
        "min": min(times),
        "mean": statistics.mean(times),
        "path_found": last.path_found,
        "path_length": last.path_length,
    }, last


def engine_backend(graph) -> str:
    engine = create_engine(graph, "iterated", None)
    assert engine is not None, "engine must be available on these graphs"
    return engine.hv.backend or "matmul(cpu-none)"


def compare_row(name, graph, wrapped, start_state, mode, hd, bw, steps, predictor=None,
                return_path=False, warmup=2, measured=3, nn_row=False):
    """Legacy vs engine in-process; hard path_found assert for hamming rows, record+warn for NN rows."""
    legacy_stats, legacy_result = run_timed(
        graph.beam_search, start_state, mode, hd, bw, steps, predictor, return_path, warmup, measured
    )
    reset_mem()
    engine_stats, engine_result = run_timed(
        wrapped.beam_search, start_state, mode, hd, bw, steps, predictor, return_path, warmup, measured
    )
    row = {
        "mode": mode,
        "history_depth": hd,
        "beam_width": bw,
        "max_steps": steps,
        "legacy": legacy_stats,
        "engine": engine_stats,
        "speedup_mean": legacy_stats["mean"] / engine_stats["mean"],
        "speedup_min": legacy_stats["min"] / engine_stats["min"],
        "engine_peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    diverged = legacy_result.path_found != engine_result.path_found
    if nn_row:
        row["nn_parity_diverged"] = diverged
        if diverged:
            print(f"WARNING: {name}: NN path_found parity diverged: {legacy_result.path_found} vs "
                  f"{engine_result.path_found}", flush=True)
        if legacy_result.path_found:
            graph.validate_path(start_state, legacy_result.path)
        if engine_result.path_found:
            graph.validate_path(start_state, engine_result.path)
    else:
        assert not diverged, f"{name}: path_found mismatch (hamming rows stay strict)"
    print(
        f"{name}: legacy={legacy_stats['mean']:.2f}s engine={engine_stats['mean']:.2f}s "
        f"speedup={row['speedup_mean']:.2f}x found={legacy_result.path_found}/{engine_result.path_found}",
        flush=True,
    )
    return row


# =============================================================================
# Stage 0: smoke + test tripwire.
# =============================================================================
print("=== Stage 0: smoke ===", flush=True)
_g0 = make_graph("lrx8")
_w0 = cayleypy_fast.wrap(_g0)
assert engine_backend(_g0) == "triton", "Triton tier must engage on this runner (see gpu_sanity kernel)"
_smoke_kwargs = dict(beam_mode="simple", beam_width=1000, max_steps=20, return_path=True)
_r_legacy = _g0.beam_search(start_state=_LRX8_START, **_smoke_kwargs)
_r_engine = _w0.beam_search(start_state=_LRX8_START, **_smoke_kwargs)
assert _r_engine.path_found == _r_legacy.path_found
if _r_engine.path_found:
    _g0.validate_path(_LRX8_START, _r_engine.path)
print(f"stage0 smoke ok (found={_r_legacy.path_found}, backend=triton)", flush=True)
results["meta"]["hash_backend"] = engine_backend(_g0)

print("=== Stage 0: pytest-in-clone (CUDA-only tests) ===", flush=True)
subprocess.check_call(["git", "clone", "-q", "https://github.com/iKolt/cayleypy-fast.git", _CLONE_DIR])
subprocess.check_call(["git", "-C", _CLONE_DIR, "checkout", "-q", _CAYLEYPY_FAST_REF])
subprocess.check_call(
    [sys.executable, "-m", "pytest", "tests/test_triton_tier.py", "tests/test_nn_parity.py", "-x", "-q"],
    cwd=_CLONE_DIR,
)
print("stage0 pytest-in-clone ok", flush=True)
results["meta"]["stages"]["smoke"] = "ok"


def write_results():
    results["meta"].update(
        {
            "torch": torch.__version__,
            "device": DEVICE_NAME,
            "vram_gb": TOTAL_VRAM_GB,
            "cayleypy_sha": _CAYLEYPY_SHA,
            "cayleypy_fast_ref": _CAYLEYPY_FAST_REF,
            "hash_backend": results["meta"].get("hash_backend"),
            "elapsed_sec": time.time() - _KERNEL_T0,
        }
    )
    with open("gpu_perf_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


# =============================================================================
# Stage 1: quick mirror of the immutable GPU baseline (hamming rows).
# =============================================================================
print("=== Stage 1: quick mirror (hamming) ===", flush=True)
QUICK = [
    ("lrx8", _LRX8_START),
    ("cube333", _CUBE333_START),
]
QUICK_MODES = [("simple", 0), ("advanced", 2), ("iterated", 2)]
for graph_name, start in QUICK:
    graph = make_graph(graph_name)
    wrapped = cayleypy_fast.wrap(graph)
    backend = engine_backend(graph)
    results["meta"]["hash_backend"] = backend
    for mode, hd in QUICK_MODES:
        if budget_left() < 1800:
            print(f"budget guard: skipping {graph_name}/{mode}", flush=True)
            continue
        name = f"{graph_name}_{mode}_bw1e5"
        reset_mem()
        results[name] = compare_row(name, graph, wrapped, start, mode, hd, 10**5, 30)
        write_results()
    del graph, wrapped
    reset_mem()

# =============================================================================
# Stage 2: deep mirror (cube333 iterated bw=2^18 mitm=3); legacy ~20 min on P100.
# =============================================================================
print("=== Stage 2: deep mirror ===", flush=True)
graph = make_graph("cube333")
wrapped = cayleypy_fast.wrap(graph)
_deep = dict(beam_mode="iterated", beam_width=2**18, max_steps=100, history_depth=2, hashed_neigbourhood=3,
             return_path=False)
if budget_left() < 3600:
    results["cube333_deep"] = {"skipped": "budget"}
else:
    reset_mem()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    e_res = wrapped.beam_search(start_state=_CUBE333_START, **_deep)
    torch.cuda.synchronize()
    e_time = time.perf_counter() - t0
    e_peak = torch.cuda.max_memory_allocated() / 2**30
    reset_mem()
    t0 = time.perf_counter()
    l_res = graph.beam_search(start_state=_CUBE333_START, **_deep)
    torch.cuda.synchronize()
    l_time = time.perf_counter() - t0
    assert l_res.path_found == e_res.path_found, "cube333_deep: path_found mismatch"
    results["cube333_deep"] = {
        "legacy": {"time_sec": l_time, "path_found": l_res.path_found, "path_length": l_res.path_length},
        "engine": {"time_sec": e_time, "path_found": e_res.path_found, "path_length": e_res.path_length},
        "speedup": l_time / e_time,
        "engine_peak_vram_gb": e_peak,
    }
    print(f"cube333_deep: legacy={l_time:.1f}s engine={e_time:.1f}s speedup={l_time/e_time:.2f}x", flush=True)
    write_results()
del graph, wrapped
reset_mem()

# =============================================================================
# Stage 2b: NN-predictor rows (representative workload).
# =============================================================================
print("=== Stage 2b: NN rows ===", flush=True)


def pretrained_or_none(graph):
    try:
        return Predictor.pretrained(graph)
    except Exception:  # kagglehub failure must never fail the kernel.
        traceback.print_exc()
        print(f"WARNING: pretrained predictor unavailable for {graph.definition.name}; skipping row", flush=True)
        return None


def seeded_mlp(graph, layers=(256, 256)):
    """Deterministic untrained MLP matching the graph's state encoding (shared legacy/engine)."""
    torch.manual_seed(20240)
    n_colors = int(torch.as_tensor(graph.definition.central_state).max()) + 1
    model = MlpModel(
        ModelConfig(
            model_type="MLP",
            input_size=graph.definition.state_size,
            num_classes_for_one_hot=n_colors,
            layers_sizes=list(layers),
        )
    )
    model.eval()
    return model.to(graph.device)


NN_ROWS = []
for _lrx_name, _start in (("lrx16", _LRX16_START), ("lrx32", _LRX32_START)):
    if budget_left() < 3600:
        print(f"budget guard: skipping NN row {_lrx_name}", flush=True)
        break
    graph = make_graph(_lrx_name)
    predictor = pretrained_or_none(graph)
    if predictor is not None:
        wrapped = cayleypy_fast.wrap(graph)
        for mode, hd in (("iterated", 2), ("simple", 0)):
            name = f"{_lrx_name}_nn_{mode}_bw1e5"
            reset_mem()
            results[name] = compare_row(
                name, graph, wrapped, _start, mode, hd, 10**5, 30, predictor=predictor, return_path=True,
                warmup=1, measured=2, nn_row=True,
            )
            write_results()
        del wrapped
    del graph, predictor
    reset_mem()

for _cube_name, _start in (("cube333", _CUBE333_START), ("cube444", _CUBE444_START)):
    if budget_left() < 3600:
        print(f"budget guard: skipping NN row {_cube_name}", flush=True)
        break
    graph = make_graph(_cube_name)
    model = seeded_mlp(graph)
    wrapped = cayleypy_fast.wrap(graph)
    for mode, hd in (("iterated", 2), ("simple", 0)):
        name = f"{_cube_name}_nn_{mode}_bw1e5"
        reset_mem()
        results[name] = compare_row(
            name, graph, wrapped, _start, mode, hd, 10**5, 30, predictor=model, return_path=True,
            warmup=1, measured=2, nn_row=True,
        )
        write_results()
    del graph, wrapped, model
    reset_mem()

# =============================================================================
# Stage 3: high-beam rows (the 5-10x criterion). Engine-only; legacy infeasible.
# =============================================================================
print("=== Stage 3: high-beam bw=2^24 ===", flush=True)
for graph_name, start in (("cube333", _CUBE333_START), ("lrx32", _LRX32_START)):
    if budget_left() < 3600:
        print(f"budget guard: skipping high-beam {graph_name}", flush=True)
        break
    graph = make_graph(graph_name)
    wrapped = cayleypy_fast.wrap(graph)
    name = f"{graph_name}_iterated_bw2p24"
    reset_mem()
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = wrapped.beam_search(
            start_state=start, beam_mode="iterated", beam_width=2**24, max_steps=30,
            history_depth=2, return_path=False,
        )
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        steps_run = res.path_length if res.path_found else 30
        results[name] = {
            "beam_width": 2**24,
            "legacy": "skipped_infeasible",
            "engine": {"time_sec": total, "path_found": res.path_found, "path_length": res.path_length,
                       "avg_step_sec": total / steps_run},
            "engine_peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
            "hash_backend": engine_backend(graph),
        }
        print(f"{name}: engine={total:.1f}s ({total/steps_run:.2f}s/step), found={res.path_found}", flush=True)
    except (RuntimeError, MemoryError) as exc:
        results[name] = {"beam_width": 2**24, "legacy": "skipped_infeasible", "engine": f"error: {exc}"}
        print(f"{name}: engine row failed: {exc}", flush=True)
    write_results()
    del graph, wrapped
    reset_mem()

write_results()
print("Wrote gpu_perf_result.json", flush=True)
