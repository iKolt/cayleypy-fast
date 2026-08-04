"""cayleypy-fast GPU sweep kernel: max-beam ladder on Tesla P100 (engine-only).

Measures beam/memory FEASIBILITY + per-step time at large beams (predictor
choice is irrelevant here ??? default hamming). Per graph, ascending beam-width
ladder; stop at the first OOM/timeout row. Mode: iterated hd=2,
return_path=False, max_steps=10. Between rows: gc + cuda cache reset; records
achieved max beam, peak memory and avg step time.

Design targets (radical-plan Goal): cube333 >= 2^26, lrx32 >= 2^26,
cube555 >= 2^25 (stretch 2^27/2^27/2^26).

Writes gpu_sweep_result.json. Download via the Python API only.
"""

import gc
import json
import subprocess
import sys
import time

# torch 2.5.1+cu121 FIRST (sm_60 P100 support), then the rest (install order as
# in the gpu perf kernel; --no-deps keeps the torch pin).
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"]
)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "h5py", "numba", "kagglehub"])

_CAYLEYPY_SHA = "0b7e109ff2d379fb2509f9fb14f7686e64453503"
_CAYLEYPY_FAST_REF = "ae3989f5f53cb371d5efdda5dc719462d8a466db"

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

from cayleypy import CayleyGraph, PermutationGroups, Puzzles  # noqa: E402

import cayleypy_fast  # noqa: E402
from cayleypy_fast.engine import create_engine  # noqa: E402

assert torch.cuda.is_available(), "GPU sweep kernel requires CUDA"
assert cayleypy_fast.run_probe().ok, cayleypy_fast.run_probe().problems

DEVICE_NAME = torch.cuda.get_device_name(0)
TOTAL_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"torch={torch.__version__}, device={DEVICE_NAME}, vram={TOTAL_VRAM_GB:.1f} GB", flush=True)

_ROW_TIME_CAP_SEC = 30 * 60  # One ladder row (10 steps) must fit in 30 min.
_KERNEL_T0 = time.time()
_KERNEL_BUDGET_SEC = 11.0 * 3600

# Fixed start states (keep in sync with kaggle_benchmarks/cpu/run.py).
_LRX32_START = list(range(16, 32)) + list(range(0, 16))
# fmt: off
_CUBE333_START = [3, 3, 1, 0, 0, 2, 1, 0, 4, 4, 2, 0, 5, 1, 4, 5, 5, 3,
                  3, 3, 5, 0, 2, 5, 4, 2, 0, 2, 4, 0, 2, 3, 3, 2, 5, 5,
                  2, 1, 0, 0, 4, 1, 2, 4, 1, 4, 3, 5, 1, 5, 1, 3, 4, 1]
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
assert len(_CUBE555_START) == 150


def make_graph(name: str) -> CayleyGraph:
    if name == "lrx32":
        return CayleyGraph(PermutationGroups.lrx(32))
    if name == "cube333":
        return CayleyGraph(Puzzles.rubik_cube(3, metric="QTM"), dtype=torch.int8, bit_encoding_width=None)
    if name == "cube555":
        return CayleyGraph(Puzzles.rubik_cube(5, metric="QTM"), dtype=torch.int8, bit_encoding_width=None)
    raise ValueError(name)


# Per-graph (start, max_steps, beam ladder). The sweep measures feasibility +
# per-step time at FULL beams, not solves:
#   - Cube graphs (G=12/24) saturate any ladder beam within ~7 steps of the
#     fixed scrambles; max_steps=10 is enough.
#   - lrx32 has only 3 generators, so 10 steps reach just 3^10 ~= 59k states ???
#     it needs ~20 steps against a deep (nbt, length-80) scramble to fill
#     2^20+ beams. A hamming-guided lrx search may still solve early; steps_run
#     is recorded, so ladder rows with few saturated steps are still readable.
LADDERS = {
    "cube333": (_CUBE333_START, 10, [2**20, 2**22, 2**24, 2**25, 2**26, 2**27, 2**28]),
    "lrx32": (None, 20, [2**20, 2**22, 2**24, 2**25, 2**26, 2**27, 2**28]),  # start: nbt-walk, below
    "cube555": (_CUBE555_START, 10, [2**22, 2**23, 2**24, 2**25, 2**26]),
}

results: dict = {}
for graph_name, (start_state, max_steps, ladder) in LADDERS.items():
    if time.time() - _KERNEL_T0 > _KERNEL_BUDGET_SEC:
        print(f"budget guard: skipping ladder {graph_name}", flush=True)
        break
    graph = make_graph(graph_name)
    if start_state is None:  # lrx32: deep non-backtracking scramble (see LADDERS note)
        np = __import__("numpy")
        np.random.seed(12345)
        torch.manual_seed(12345)
        start_state = graph.random_walks(width=1, length=81, mode="nbt", nbt_history_depth=1)[0][-1]
    wrapped = cayleypy_fast.wrap(graph)
    engine = create_engine(graph, "iterated", None)
    assert engine is not None, f"engine unavailable for {graph_name}"
    graph_rows = []
    achieved = 0
    for bw in ladder:
        if time.time() - _KERNEL_T0 > _KERNEL_BUDGET_SEC:
            print(f"budget guard: stopping ladder {graph_name} at bw=2^{bw.bit_length()-1}", flush=True)
            break
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = {"beam_width": bw}
        try:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            res = wrapped.beam_search(
                start_state=start_state, beam_mode="iterated", beam_width=bw, max_steps=max_steps,
                history_depth=2, return_path=False,
            )
            torch.cuda.synchronize()
            total = time.perf_counter() - t0
            steps_run = res.path_length if res.path_found else max_steps
            row.update(
                {
                    "status": "ok" if total <= _ROW_TIME_CAP_SEC else "timeout",
                    "time_sec": total,
                    "steps_run": steps_run,
                    "avg_step_sec": total / steps_run,
                    "path_found": res.path_found,
                    "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
            )
            if total > _ROW_TIME_CAP_SEC:
                graph_rows.append(row)
                break  # Timeout: stop the ladder here.
            achieved = bw
        except (RuntimeError, MemoryError) as exc:
            row.update({"status": "oom" if "out of memory" in str(exc).lower() else "error", "error": str(exc)[:400]})
            graph_rows.append(row)
            print(f"{graph_name} bw=2^{bw.bit_length()-1}: {row['status']} ({str(exc)[:120]})", flush=True)
            break  # OOM/error: stop the ladder here.
        graph_rows.append(row)
        print(
            f"{graph_name} bw=2^{bw.bit_length()-1}: {row['time_sec']:.1f}s "
            f"({row['avg_step_sec']:.2f}s/step, {row['steps_run']} steps) peak={row['peak_vram_gb']:.1f}GB",
            flush=True,
        )
    results[graph_name] = {
        "rows": graph_rows,
        "max_beam_achieved": achieved,
        "max_beam_log2": achieved.bit_length() - 1 if achieved else None,
        "hash_backend": engine.hv.backend,
    }
    with open("gpu_sweep_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)  # Write incrementally: keep partial results on failure.
    del graph, wrapped, engine
    gc.collect()
    torch.cuda.empty_cache()

results["meta"] = {
    "torch": torch.__version__,
    "device": DEVICE_NAME,
    "vram_gb": TOTAL_VRAM_GB,
    "cayleypy_sha": _CAYLEYPY_SHA,
    "cayleypy_fast_ref": _CAYLEYPY_FAST_REF,
    "elapsed_sec": time.time() - _KERNEL_T0,
}
with open("gpu_sweep_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Wrote gpu_sweep_result.json", flush=True)
