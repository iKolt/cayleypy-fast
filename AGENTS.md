# AGENTS.md

Guidance for AI agents (and humans) working on the cayleypy-fast codebase.
Read this before making any changes to the engine, the patch layer, or the
benchmark kernels. Line anchors are stamped as of `33cae8b`; re-verify with
grep before citing them in new docs.

## 1. Project at a glance

cayleypy-fast is a **drop-in beam-search accelerator for cayleypy**: a separate
pure-Python package (`py3-none-any` wheel, no native build step) that replaces
the beam-search hot path with a chunked matmul-hash engine plus optional numba
(CPU) / Triton (GPU) tiers. Zero changes to cayleypy core.

- **Activation**: `import cayleypy_fast; cayleypy_fast.enable()` patches
  `cayleypy.cayley_graph.BeamSearchAlgorithm` in place (guarded monkeypatch,
  probe-gated). Patch-averse alternative: `cayleypy_fast.wrap(graph)`. Pytest
  plugin: `pytest -p cayleypy_fast.pytest_plugin <host suite>`.
- **Maturity**: working prototype — CPU: T0/T1/T2/T4 done; GPU: T3 Triton tier,
  T5 parity+smoke, T7-GPU benchmarks done on Kaggle P100 (see §13).
- **Upstream pin**: developed and parity-tested against cayleypy @ `0b7e109`
  (editable install from the sibling checkout `C:\Users\xiaomi\cayleypy`).
- **License:** MIT.

## 2. Environment & commands

Venv `.venv/` (measured): Python 3.11.7, torch 2.13.0+cpu, numba 0.66.0,
numpy 2.4.6, pytest 9.1.1 + pytest-benchmark 5.2.3, mypy 1.15.0, pylint
4.0.6, black 26.5.1. cayleypy is installed editable from the sibling checkout
(`pip install --no-deps -e C:\Users\xiaomi\cayleypy` — `--no-deps` keeps pip
from upgrading torch to satisfy cayleypy's `torch>=2.6` metadata).

```powershell
pip install -e .[lint,test,numba]     # dev install
pytest                                # 162 tests, 8 files, offline (21 CUDA-gated skips on CPU)
bash lint.sh                          # black --check + pylint + mypy (PR gate)

# Run the UPSTREAM cayleypy suite with the engine active (in-process enable):
python -m pytest -p cayleypy_fast.pytest_plugin C:\Users\xiaomi\cayleypy\cayleypy\algo\beam_search_test.py
```

Measured upstream suite timings at cayleypy@0b7e109 with the patch enabled:
65.7s for the beam_search test file, 159.2s for the full upstream suite.

**Pitfall:** `pytest -q` with stdout redirected/buffered hides progress — the
dots arrive only at process exit. Measure with in-script `time.perf_counter()`
or tail a log file, never by watching the dots.

## 3. Repository map (as of 33cae8b)

```
cayleypy_fast/
  __init__.py          # Public API: enable/disable/is_enabled/wrap, run_probe, ProbeResult
  _patch.py            # Guarded monkeypatch layer
  _probe.py            # Compatibility probe against the live cayleypy internals
  engine.py            # FastBeamEngine: chunked matmul-hash beam engine (all 4 modes)
  numba_kernels.py     # Optional numba CPU tier for the dual-int32 hash primitive
  triton_kernels.py    # Optional Triton GPU tier for the int64 hash (broadcast-mul + tl.sum)
  pytest_plugin.py     # In-process enable() for host test suites
tests/                 # 162 tests: delegation, enable, engine_parity, hash_parity, numba_tier, probe, triton_tier, nn_parity
docs/radical-speedup-plan.md   # Design doc + measured T2/T4 benchmark tables
kaggle_benchmarks/     # 8 Kaggle kernels: 5 CPU (cpu, cpu_heavy, cpu_deep, cpu_sweep, cpu_micro) + 3 GPU (gpu_sanity, gpu, gpu_sweep)
kaggle_out/            # Downloaded kernel logs + result JSONs (gitignored)
conftest.py            # Deterministic seeding (DETERMINISTIC_SEED = 12345)
lint.sh                # black 120 + pylint + mypy
```

### Code anchors (grep-verified at 33cae8b)

| `engine.py` | Line | Role |
|---|---|---|
| `class PermutedHashVectors` | 115 | Permuted hash vector table `VH[j, g] = v[p_g^-1[j]]` (+ `backend` slot for the CUDA ladder) |
| `build_permuted_hash_vectors` | 158 | Builds VH from the live `graph.hasher` |
| `hash_neighbors_mulsum` | 191 | Rung-3 int64 fallback (never materializes neighbors) |
| `_hash_int64_dispatch` | 220 | CUDA int64 dispatch with mid-search demotion (warn once, recompute) |
| `resolve_backend` | 251 | Once-per-engine ladder resolution: triton → matmul → mulsum probe (bit-checked vs mulsum) |
| `hash_neighbors_tiled` | 297 | Tiled neighbor hashing; CPU dual-int32 → numba, CUDA int64 → ladder |
| `engine_available` | 330 | Gate: permutation group + dot-product hasher + probe |
| `class _GlobalTopK` | 340 | Running global topk (simple/advanced) |
| `class _PerGenTopK` | 422 | Per-generator running topk (iterated modes) |
| `class _BatchedRedistribute` | 453 | Legacy surplus redistribution (iterated_batched) |
| `class FastBeamEngine` | 518 | The engine; `_hash_tile_flat`:575, `_tile_candidates`:584, `_apply_cumulative_ban`:612, `_apply_window_ban`:621, `_make_score_fn`:633, `_materialize`:681 (chunked), `_run`:693 |
| `search_simple/advanced/iterated/iterated_batched` | 883/914/945/976 | Forked mode methods |
| `create_engine` + per-graph cache | 1039 | WeakKeyDictionary-keyed engine cache |

| `triton_kernels.py` | Line | Role |
|---|---|---|
| `_ENV_DISABLE` | 47 | `CAYLEYPY_FAST_TRITON_DISABLE` kill-switch |
| `triton_available` | 112 | Optional-import gate |
| `_pick_config` | 119 | Fixed config table by (rows, G); no autotune |
| `hash_neighbors_int64_triton` | 142 | Fused int64 hash kernel wrapper (broadcast-mul + tl.sum) |

| `_patch.py` | Line | Role |
|---|---|---|
| `ENV_DISABLE` | 22 | `CAYLEYPY_FAST_DISABLE` |
| `ENV_MIN_BG` / `ENV_MIN_BEAM` | 31/32 | Size-gate env vars; defaults 2^16 / 512 |
| `class FastBeamSearchAlgorithm` | 38 | Subclass of legacy BeamSearchAlgorithm; size gate lives in `_fast_engine`:56 |
| `enable` / `disable` / `is_enabled` | 108/138/152 | Probe-gated patch/unpatch |
| `class FastGraphWrapper` / `wrap` | 157/181 | Patch-averse API |

| `_probe.py` | Line | Role |
|---|---|---|
| `_EXPECTED_RESTORE_PATH_PARAMS` | 57 | `_restore_path` param names incl. `destination_state` |
| `PATCH_POINTS` | 80 | Modules whose `BeamSearchAlgorithm` symbol gets patched |
| `class ProbeResult` | 84 | Probe outcome |
| signature checks | 124-143 | `_check_search_signature`, `_check_restore_path_signature` |
| `run_probe` | 190 | Full probe |

| `numba_kernels.py` | Line | Role |
|---|---|---|
| `_ENV_DISABLE` | 34 | `CAYLEYPY_FAST_NUMBA_DISABLE` |
| `numba_available` | 71 | Optional-import + kill-switch gate |
| `hash_neighbors_dual_int32_numba` | 88 | Fused parallel dual-int32 hash kernel |

`pytest_plugin.py:19` — `pytest_configure` calls `cayleypy_fast.enable()` once.

## 4. Architecture & data flow

Three activation paths, all funneling into the same engine:
1. `enable()` — in-place symbol patch (probe-gated; failure ⇒ total fallback to
   legacy, never partial).
2. `wrap(graph)` — per-graph wrapper, no patching.
3. `pytest -p cayleypy_fast.pytest_plugin` — in-process `enable()`.

Engine step pipeline (`FastBeamEngine._run`, per step, per tile × generator
chunk, gen-major flat layout):
VH matmul hash (never materialize neighbor states) → chunk dedup vs step
`TorchHashSet` → MITM check (`isin_via_searchsorted` vs BFS layer hashes,
strictly BEFORE the ban) → non-backtracking ban (cumulative/window semantics
replicated per mode) → lazy scoring (only candidates surviving dedup+ban) →
per-mode selection → end-of-step one-gather materialization
(`new_beam = gather(beam_states[parents], 1, perms[gens])`).

Key insight: for permutation groups with the dot-product hasher,
`hash(g(s)) = s · (v ∘ p_g^{-1})`, so all `B × G` neighbor hashes are one
matmul against precomputed VH. VH is derived **from the live
`graph.hasher` instance** (never regenerated) ⇒ chunk hashes bit-match
`hasher.make_hashes`. On CPU the hashers dual-int32 path is dispatched to the
numba kernel inside `hash_neighbors_tiled`. On CUDA the int64 path dispatches
through a per-engine backend ladder resolved by `resolve_backend`:
triton (probe-compiled + bit-checked) → torch matmul (mirrors the hasher's own
try/except) → mulsum (per-generator multiply+sum, bit-equal floor), with
mid-search demotion (warn once, demote, recompute — search never aborts).

## 5. Upstream contract snapshot (pinned cayleypy @ 0b7e109)

Critical contracts the engine depends on (anchors into the sibling checkout;
peripheral upstream detail — BFS internals, datasets, model registry — see
`cayleypy/AGENTS.md`).

- **Sorted-hash invariant**: `get_unique_states` (`cayley_graph.py:122`)
  returns hashes in sorted non-decreasing order (`torch.sort(hashes,
  stable=True)` at `cayley_graph.py:131`). Feeds `isin_via_searchsorted`
  (`torch_utils.py:4`, `test_elements_sorted` MUST be sorted) and MITM
  `_check_path_found` (`beam_search.py:82-86`). The engine mirrors this by
  keeping hashes sorted through `_tile_candidates`.
- **`_restore_path` takes `destination_state`** (fix A1,
  `beam_search.py:89`): when `found_layer_id == 0` the path is restored to
  `destination_state`, not `central_state`.
- **`encode_states` preserves the INPUT dtype** (`cayley_graph.py:143`,
  `torch.as_tensor(states)` with no cast): a Python-list start state enters the
  whole search as int64 EVEN IF the graph was built with `dtype=torch.int8`.
  Both legacy and the engine carry it through; hashes are unaffected (values
  are small), but state tensors cost 8× memory (see the dtype trap in §7).
- **The 6 BeamSearchResult invariants** (from cayleypy/AGENTS.md §6, verbatim):
  1. `graph.apply_path(start_state, path)` equals `graph.central_state` (or `destination_state` if specified).
  2. `len(path) == path_length` (enforced by `BeamSearchResult.__post_init__`).
  3. All elements of `path` are valid generator ids: `0 <= g < graph.definition.n_generators`.
  4. When `hashed_neigbourhood` is provided as a `BfsResult`, its graph must match the search graph (`_precompute_mitm` at `beam_search.py:177` raises `ValueError` otherwise).
  5. Meet-in-the-middle path length = `i_step + bfs_layer_id` (the beam step plus the BFS layer where intersection was found).
  6. Non-backtracking (`history_depth > 0`) must never ban the destination neighborhood — otherwise the search could not find the goal.
- **Replicated legacy quirks (do NOT fix here)**:
  - Hamming predictor always measures distance to `graph.central_state`, even
    with a custom `destination_state` (`predictor.py:42`). Reproduce
    byte-for-byte; flag with `# TODO(char-spec)`.
  - `advanced` mode ban = cumulative `TorchHashSet`; `iterated` modes = window
    (slot) ban with legacy slot bookkeeping (reset current slot per step;
    query all slots before adding the chunk; add post-dedup hashes to the
    current slot).
  - B2: iterated modes raise `ValueError` when `beam_width < n_generators`.
  - B3: simple/advanced exit early on empty beam after dedup.
  - The `iterated_batched` CUDA memory gate (1.5x materialized-states
    estimate) is invalid under the engine → replaced by chunk-bounded
    accounting in the forked mode method.
  - `debug_scores` legacy formulas per mode: advanced = min of selected;
    iterated_batched = global min of survivor scores; iterated = min over chunks.
- **Probe surface**: `_restore_path` param names
  (`_EXPECTED_RESTORE_PATH_PARAMS`), `_cuda_sync`,
  `_BeamSearchProfile[reset_step, format_line]`, `BeamSearchResult`, and the
  `PATCH_POINTS` modules (`cayleypy.cayley_graph`, `cayleypy.algo`,
  `cayleypy.algo.beam_search`).

## 6. cayleypy-fast invariants (DO NOT break)

- **Outcome parity**: same `path_found` / `path_length` as legacy on the same
  inputs (tie policy may differ; set equality, not path identity).
- Found path is valid: `apply_path(start_state, path) == destination`;
  `len(path) == path_length`; all generator ids in range.
- Hashes stay **sorted** through `_tile_candidates` (engine-side mirror of the
  upstream sorted-hash invariant).
- `_GlobalTopK.finalize` postcondition: `hashes.numel() <= k`.
- **Bit-equality (T1 property)**: engine chunk hashes bit-equal
  `hasher.make_hashes(materialized neighbors)` on seeded random states; the
  numba tier is bit-equal to the torch reference.
- Probe failure at `enable()` ⇒ **total fallback** to legacy (warning +
  no-op), never a partially patched state.

## 7. Danger zones / perf candidates + bug history

- **Runaway-beam bug** (fixed @`45c5985`): `_GlobalTopK.offer` first-chunk
  path stored chunks larger than `beam_width` wholesale (a typical step has
  exactly ONE offer), so the beam grew ~G× per step — lrx16+MLP reached 839K
  states at step 21 (814s vs 2.5s legacy). Fixed with an immediate top-k cap +
  `finalize` invariant + a regression test with a wall-clock bound. Lesson:
  small-graph parity tests could NOT see it (lrx6 state space = 720) — parity
  harness requires a long-scramble big-graph test.
- **Per-graph engine cache** (@`3cd0773`): ms-scale rows (path found in 6–16
  steps) paid `create_engine` setup (VH build, argsort, central_by_gen) per
  call. WeakKeyDictionary cache keeps overhead ≤1ms on those rows.
- **numba tier** (@`0732d80`): torch int32 matmul does NOT dispatch to an
  optimized BLAS integer GEMM (0.75–2 GMAC/s); the numba kernel with
  transposed (2G, n) layout + fused int8 read is 13–39× faster, bit-equal.
  Pitfall: JIT ~0.5s on the first call pollutes `mean` on ms rows — always
  compare `min` (or drop warmup) on ms-scale benchmark rows.
- **`_CUBE555_START` 149-vs-150** (@`927b165`): a hand-copied cube555 start
  state constant was one element short and crashed the kernel. Lesson: every
  hand-copied constant gets a length assert
  (`assert len(_CUBE555_START) == 150`).
- **Size gate**: production thresholds `CAYLEYPY_FAST_MIN_BG` (default 2^16)
  and `CAYLEYPY_FAST_MIN_BEAM` (default 512) keep the engine off tiny searches
  where setup dominates. The parity fixture forces `"0"`/`"0"`
  (`tests/test_engine_parity.py:36-37`) to exercise the engine everywhere.
- **Kill switches**: `CAYLEYPY_FAST_DISABLE=1` (enable() no-op),
  `CAYLEYPY_FAST_NUMBA_DISABLE=1` (numba tier off, torch fallback),
  `CAYLEYPY_FAST_TRITON_DISABLE=1` (triton tier off, ladder demotes at resolve).
- **int64-state dtype trap** (@`33cae8b`, Kaggle P100 OOM): list start states
  search as int64 end-to-end (see §5); at bw=2^24 cube333 the (2^24×54) int64
  survivor batch is a single 6.75 GiB alloc on a 16 GB GPU. Fix convention in
  the GPU kernels: cast starts to `graph.dtype`; the engine itself must stay
  dtype-faithful to legacy (drop-in parity promise).
- **_materialize gather index is (K, n) int64** (@`ae3989f`): `perms[gens]` on
  the full survivor batch was multi-GB at 2^24+ beams; now chunked under
  `_MATERIALIZE_CHUNK_BYTES` (bit-identical, parity test monkeypatches the
  constant tiny).
- **Legacy `iterated` has NO within-chunk dedup** (verified
  `beam_search.py:903`): it dedups later generators' chunks only against the
  post-topk survivors of earlier generators; the engine's tile-level dedup is
  strictly stronger. Consequence: NN `debug_scores`-level parity is legal for
  simple/advanced/iterated_batched only — iterated keeps outcome parity.
- **CUDA test RNG ≠ CPU test RNG** (Philox): fixed-instance parity tests must
  either tune instances per device or probe-select them
  (`test_nn_parity._find_solvable_start` — nbt walks, classic walks on LRX
  collapse via inverse generators to distance ≤ 2).
- **GPU max-beam wall = candidate-hash storage** (T7-GPU sweep): live memory
  per step ≈ 3 × `bw·G·8B` (step dedup set + hd=2 ban slots) + beam + ~1.5 GB
  tiles/sort. cube333 caps at 2^25, cube555 at 2^22, lrx32 untested-ceiling
  (2^28, beams stay small under hamming). See the design doc T7 section.

## 8. Measured benchmarks

Final combined run (engine @`0732d80` = runaway fix + per-graph cache + numba
tier; Kaggle 4 vCPU CPU, torch 2.10.0+cpu, cayleypy@0b7e109; all 5 CPU kernels
COMPLETE). Result JSONs traceable in `kaggle_out/` (see §14). Full tables with
T2-vs-final history and the SHAs of each measurement: `docs/radical-speedup-plan.md`.

| Kernel | Row | legacy | engine | speedup |
|---|---|---|---|---|
| cpu | lrx8/lrx32 iterated | 55ms / 175ms | 29ms / 65ms | **1.93× / 2.68×** |
| cpu | cube222 simple | 1.08s | 0.58s | 1.86× |
| cpu | cube333 simple/advanced/iterated | 41.3s / 61.3s / 38.2s | 26.2s / 36.2s / 25.7s | 1.58× / 1.69× / 1.49× |
| cpu | cube444/cube555 iterated | 12.2s / 17.1s | 4.67s / 6.32s | **2.62× / 2.71×** |
| cpu-heavy (3 runs) | cube444 iterated / it_batched bw=1e4 | 11.63s / 3.99s | 4.75s / 1.52s | **2.45× / 2.63×** |
| cpu-heavy (3 runs) | cube555 iterated / it_batched bw=1e4 | 1.91s / 3.90s | 0.69s / 1.32s | **2.76× / 2.96×** |
| cpu-deep | cube333 iterated/it_batched bw=2^18 mitm=3 | 266.6s / 380.5s | 186.0s / 191.1s | 1.43× / **1.99×** |
| cpu-sweep | cube333 it_batched bw 1e4/1e5 | 5.85s / 65.1s | 2.55s / 30.4s | **2.30× / 2.15×** |
| cpu-sweep | bw=256 gate control / bw=1e3 | 0.180s / 0.485s | 0.143s / 0.296s | 1.26× / 1.64×(min), 0.76×(mean, JIT) |

Net: 16/21 engine rows ≥2× on mean; all ≥2× on min where warmup is excluded.
Heavy rows confirmed with 3-run statistics (stdev ≤0.2s). Remaining sub-1×
rows are ms-scale searches (path found in 6–16 steps; ≤1ms absolute overhead
post-cache). **ms-row caveat:** first-call numba JIT (~0.5s) inflates `mean` —
compare `min`. Parity: all parity harnesses green (see §10).

Reproduce: push the 5 kernels (§11), download outputs, read the
`cpu_bench*_result.json` files.

### GPU (Kaggle P100, torch 2.5.1+cu121, cayleypy@0b7e109, engine in `ae3989f`)

| Kernel | Row | legacy | engine | speedup |
|---|---|---|---|---|
| gpu | lrx8 simple/advanced/iterated bw=1e5 | 13/22/53 ms | 21/15/27 ms | 0.63×(ms row) / 1.42× / 1.98× |
| gpu | cube333 simple/advanced/iterated bw=1e5 | 0.40/0.53/1.13 s | 0.27/0.36/0.36 s | 1.45× / 1.48× / **3.10×** |
| gpu | cube333 iterated bw=2^18 mitm=3 | 1.68 s | 0.65 s | **2.56×** |
| gpu | lrx16/lrx32 iterated bw=1e5, pretrained MLP | 21/77 ms | 14/37 ms | 1.53× / 2.09× |
| gpu | cube333/cube444 NN rows bw=1e5 | 3.9–10.9 s | 3.0–8.9 s | 1.02–1.25× (NN-forward dominated) |
| gpu | cube333 iterated bw=2^24 (engine-only) | infeasible | 48.6 s, **1.68 s/step**, found | — |
| gpu_sweep | cube333 / lrx32 / cube555 max beam (iterated hd=2) | — | 2^25 / 2^28 / 2^22 | wall: candidate-hash sets ≈3×bw·G·8B |

Sanity kernel (`gpu_sanity`): 5/5 shape bit-equality for the Triton kernel on
P100, ~3.6× over the mulsum micro baseline; clone-side pytest of
test_triton_tier+test_nn_parity inside the perf kernel: 40/40 green.

## 9. Contribution rules

- **Before commit:** `pytest` green + `bash lint.sh` green (black line-length
  120, pylint 10/10, mypy clean).
- **Git mutations only on explicit user authorization.**
- Commit style per history: conventional-ish prefixes (`docs:`, `kaggle ...:`,
  task-tagged messages like `T4 numba tier: ...`), body with motivation +
  numbers.
- **Update this AGENTS.md** whenever engine structure, gates, env vars, or the
  kernel set change.

## 10. Testing conventions

- **Seeds:** `conftest.py` autouse fixture seeds numpy/random/torch to 12345
  (mirrors cayleypy). Graph construction in tests uses seed 42.
- **Isolation:** `tests/conftest.py` autouse fixture calls
  `cayleypy_fast.disable()` before AND after every test — the patch must never
  leak across tests.
- **Parity harness rules:** assert hash **set equality** per step, not path
  identity (tie policy may diverge); iterated modes also assert per-generator
  survivor counts; `validate_path` on found paths. Small graphs are
  insufficient — include a long-scramble big-graph parity test (they miss
  whole-space-collision bugs, see the runaway-beam lesson §7).
- **numba tests:** `pytest.mark.skipif` on `numba_available()`; 17 tier tests
  (bit-equality × 12 shapes, dispatch, kill-switch, cache, speed guard >2×).
- **Speed guards:** tier tests assert >2× thresholds so regressions fail CI.
- **No network in the default suite.**

## 11. Kaggle workflow

- **Auth (one-time per session):** `~/.kaggle/kaggle.json` holds
  `{"username":"ivanKolt","key":"KGAT_..."}`. The kaggle CLI v2.x does NOT
  read `kaggle.json` — export the key:
  `$env:KAGGLE_API_TOKEN="KGAT_..."` (PowerShell).
- **Cycle** (GPU kernels run ~5–15 min; CPU kernels ~40–60 min per push → queue
  → run → download):
  ```powershell
  kaggle kernels push -p kaggle_benchmarks/cpu          # or any other kernel dir
  kaggle kernels status ivankolt/cayleypy-cpu-bench
  # Download ONLY via the Python API (the `kaggle kernels output` CLI AND the
  # plain API hit a Windows charmap bug when logs contain non-ASCII — use
  # PYTHONUTF8 + -X utf8):
  $env:PYTHONUTF8='1'
  python -X utf8 -c "from kaggle import KaggleApi; api=KaggleApi(); api.authenticate(); api.kernels_output('ivankolt/cayleypy-gpu-perf', path='./kaggle_out/gpu_perf_v9', force=True, quiet=True)"
  ```
- **Pretrained model weights need `model_sources`**: kagglehub's cache resolver
  cannot attach un-attached models in non-interactive sessions (BackendError
  code 9). The gpu kernel attaches `fedimser/lrx-16/pyTorch/ep60/1` and
  `fedimser/lrx-32-by-mrnnnn/pyTorch/model_final/1` in `kernel-metadata.json`.
- **Parallel-5 rule:** up to 5 kernels run in parallel on a free account —
  push all 5 at once.
- **Bump `_CAYLEYPY_FAST_REF` consistently across ALL kernel run.py files** when
  re-measuring (GPU kernels currently `ae3989f5f53cb371d5efdda5dc719462d8a466db`).
  Kernels install `git+https://github.com/iKolt/cayleypy-fast.git@<ref>`.

| Kernel id | Dir | Matrix / intent | Result file (kaggle_out/) |
|---|---|---|---|
| `ivankolt/cayleypy-cpu-bench` | `cpu/` | lrx8/16/32, cube222/333/444/555 × 4 modes, hamming | `cpu_bench{,_v2}/cpu_benchmark_result.json` |
| `ivankolt/cayleypy-cpu-heavy` | `cpu_heavy/` | cube444/555 iterated+it_batched bw=1e4, 3 runs | `cpu_heavy_v{2,3}/cpu_bench_heavy_result.json` |
| `ivankolt/cayleypy-cpu-deep` | `cpu_deep/` | cube333 iterated/it_batched bw=2^18, mitm=3 | `cpu_deep{,_v2}/cpu_bench_deep_result.json` |
| `ivankolt/cayleypy-cpu-sweep` | `cpu_sweep/` | cube333 it_batched bw 1e3/1e4/1e5 + bw=256 gate control | `cpu_sweep{,_v2}/cpu_bench_sweep_result.json` |
| `ivankolt/cayleypy-cpu-micro` | `cpu_micro/` | ms-scale rows (engine-cache re-validation) | `cpu_micro_v2`, `cpu_micro_cache/cpu_bench_micro_result.json` |

- **Quota:** ~30 GPU-hours/week free; CPU kernels are cheaper — still
  budget cycles for before/after pairs.
- **GPU kernels:** the immutable baseline lives in the **cayleypy repo**
  (`kaggle_benchmarks/baseline`, id `ivankolt/cayleypy-gpu-baseline`, installs
  cayleypy pinned @`4ba6b04`). The perf GPU kernel (id
  `ivankolt/cayleypy-gpu-perf`) belongs to **cayleypy-fast** (per the
  cayleypy AGENTS.md §10 two-kernel sync rule: copy + change install SHA and
  kernel id only). Kaggle allocates Tesla P100 (sm_60) → pin
  `torch==2.5.1+cu121` (default Kaggle torch crashes with
  `cudaErrorNoKernelImageForDevice`).

| GPU kernel id | Dir | Intent | Result (kaggle_out/) |
|---|---|---|---|
| `ivankolt/cayleypy-gpu-sanity` | `gpu_sanity/` | Triton kernel bit-equality micro (5 shapes, dtypes) | `gpu_sanity{,_v2}/gpu_sanity_result.json` |
| `ivankolt/cayleypy-gpu-perf` | `gpu/` | 4-stage perf: smoke+clone-pytest → quick mirror → deep + NN rows → bw=2^24 engine-only | `gpu_perf_v{3..9}/gpu_perf_result.json` |
| `ivankolt/cayleypy-gpu-sweep` | `gpu_sweep/` | Engine-only max-beam ladders 2^20…2^28 (cube333/lrx32/cube555) | `gpu_sweep_v1/gpu_sweep_result.json` |

## 12. Windows/PowerShell pitfalls

- No `&&` in PowerShell 5.1 — use `; if ($?) { ... }`.
- Stdout redirects (`> file`, `Out-File`) buffer until completion — use
  file-based logs + tail, or measure with `time.perf_counter()`/TimeSpans
  in-script. Never judge progress by piped output.
- `git push` prints to stderr → spurious `NativeCommandError`; benign.
- venv launcher PID shows CPU≈0 — the real work is the child `python.exe`.
- No `Start-Sleep` polling — run long jobs (pytest, kaggle cycles) as
  background processes.

## 13. Task tracker

Cross-ref: `docs/radical-speedup-plan.md` (full design + risk matrix).

| Task | Status |
|---|---|
| T0 scaffold (package, probe, enable/wrap/plugin, lint) | DONE |
| T1 baseline & characterization (bit-equality property tests, parity harness) | DONE |
| T2 torch engine, 4 modes, hamming (measured table in design doc) | DONE @`45c5985` |
| T4 numba tier (13–39× hash kernel, bit-equal, 17 tests) | DONE @`0732d80` |
| T3 Triton GPU tier (ladder + demotion, 24 tier tests, Kaggle P100 validated) | DONE @`79808f8` (+OOM fixes @`ae3989f`/`33cae8b`) |
| T5 parity+smoke NN tests (trained lrx8 + untrained cube222, 4 modes × CPU/CUDA) | DONE @`79808f8` (+probe fixes @`356b201`–`d171fad`) |
| T5 NN *perf* tuning (separate task) | NOT DONE — deferred; engine ≥1.0× legacy on NN rows (NN-forward dominated) |
| T6 encoded-state path (spike-gated, low priority) | TODO |
| T7-GPU scaling sweep + GPU benchmarks + docs refresh | DONE @`33cae8b` (cubes capped 2^25/2^22 by candidate-hash sets; lrx32 2^28) |
| T7 remaining | candidate-hash memory redesign only if bigger GPU beams wanted |

## 14. Where's what

| What | Where |
|---|---|
| Old task plans (audit fixes, radical speedup) | `C:\Users\xiaomi\cayleypy\.kilo\plans\` (read-only history; do NOT move/edit) |
| New task plans | `C:\Users\xiaomi\cayleypy-fast\.kilo\plans\` (planner writes here when sessions open in this repo) |
| Benchmark result JSONs + kernel logs | `kaggle_out/` (gitignored) — subdirs per kernel/version |
| Legacy pytest-benchmark baseline | `cayleypy/.benchmarks/Windows-CPython-3.11-64bit/0004_R0-pre.json` |
| GPU baseline kernel | `cayleypy/kaggle_benchmarks/baseline` (id `ivankolt/cayleypy-gpu-baseline`, pinned `4ba6b04`) |
| Upstream guidance (BFS, models, all 6 helpers' detail) | `C:\Users\xiaomi\cayleypy\AGENTS.md` |
| Design doc + full benchmark tables | `docs/radical-speedup-plan.md` |
