# cayleypy-fast

Pluggable high-performance beam search for [CayleyPy](https://github.com/cayleypy/cayleypy).

Separate add-on project (no cayleypy core changes). Three performance tiers:
a pure-torch engine (device-agnostic reference, always available), Triton CUDA
kernels, and numba CPU kernels. Pure Python + JIT sources: **no native build
step** (`py3-none-any` wheel).

## Usage

```python
import cayleypy_fast
cayleypy_fast.enable()   # All graph.beam_search(...) calls now use the fast engine when available.
```

If the installed cayleypy version is incompatible, `enable()` emits a warning
and leaves cayleypy untouched (legacy behaviour). Set the environment variable
`CAYLEYPY_FAST_DISABLE=1` to force the legacy implementation. `cayleypy_fast.disable()`
restores the originals at runtime.

Patch-averse API:

```python
wrapped = cayleypy_fast.wrap(graph)
result = wrapped.beam_search(start_state=...)   # Same signature as CayleyGraph.beam_search.
```

## Install order (torch pin)

Kaggle pins `torch==2.5.1+cu121` while cayleypy declares `torch>=2.6`; this
package declares `torch>=2.5`. To keep a pinned torch, install in this order:

```bash
pip install --no-deps "cayleypy @ git+https://github.com/cayleypy/cayleypy.git@<sha>"
pip install h5py kagglehub numba numpy scipy   # cayleypy's other runtime deps
pip install "cayleypy-fast @ git+https://github.com/cayleypy/cayleypy-fast.git@<sha>"
pip install "torch==2.5.1+cu121" --index-url https://download.pytorch.org/whl/cu121   # torch pinned LAST
```

## Status

- **T0** (scaffold): patch plumbing (`enable()` / `disable()` / `wrap()` /
  compatibility probe) complete.
- **T1**: hash bit-equality property tests (engine matmul vs
  `hasher.make_hashes` on materialized neighbors) over 12 permutation graphs x
  {int8, int64}.
- **T2**: pure-torch engine (all 4 modes) — neighbor states are never
  materialized (one matmul per tile hashes all `B x G` candidates); chunked
  dedup into per-step `TorchHashSet`; lazy scoring; lazy survivor
  materialization. Parity vs legacy validated in `tests/test_engine_parity.py`.

Design doc: [docs/radical-speedup-plan.md](docs/radical-speedup-plan.md).

### Documented deviations from legacy (parity non-goals)

- Tie-breaking under chunked dedup may select different (equally-scored)
  candidates than legacy; invariant-level guarantees (valid path, path length
  arithmetic `i_step + bfs_layer_id`) are preserved.
- The legacy `iterated_batched` CUDA memory gate (raising `MemoryError`) is
  replaced by chunk-bounded candidate accounting; beyond budget the engine uses
  per-generator (iterated) selection for that search instead of failing.
- Legacy `advanced` never resets its history-depth slots; the engine
  reproduces the resulting semantics exactly as a single ever-growing seen set
  (cumulative ban). `iterated` / `iterated_batched` keep the sliding-window
  semantics (slot reset per step).
