# AGENTS.md

Guidance for AI agents (and humans) working on the cayleypy-fast codebase.
Read this before making any changes to the engine, the patch layer, or the
benchmark kernels. Line anchors are stamped as of `c237ca6`; re-verify with
grep before citing them in new docs.

## 1. Project at a glance

cayleypy-fast is a **drop-in beam-search accelerator for cayleypy**: a separate
pure-Python package (`py3-none-any` wheel, no native build step) that replaces
the beam-search hot path with a chunked matmul-hash engine plus optional numba
CPU tier. Zero changes to cayleypy core.

- **Activation**: `import cayleypy_fast; cayleypy_fast.enable()` patches
  `cayleypy.cayley_graph.BeamSearchAlgorithm` in place (guarded monkeypatch,
  probe-gated). Patch-averse alternative: `cayleypy_fast.wrap(graph)`. Pytest
  plugin: `pytest -p cayleypy_fast.pytest_plugin <host suite>`.
- **Maturity**: working prototype, T0/T1/T2/T4 done (see §13). T3 Triton GPU
  tier is next.
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
pytest                                # 118 tests, 6 files, offline
bash lint.sh                          # black --check + pylint + mypy (PR gate)

# Run the UPSTREAM cayleypy suite with the engine active (in-process enable):
python -m pytest -p cayleypy_fast.pytest_plugin C:\Users\xiaomi\cayleypy\cayleypy\algo\beam_search_test.py
```

Measured upstream suite timings at cayleypy@0b7e109 with the patch enabled:
65.7s for the beam_search test file, 159.2s for the full upstream suite.

**Pitfall:** `pytest -q` with stdout redirected/buffered hides progress — the
dots arrive only at process exit. Measure with in-script `time.perf_counter()`
or tail a log file, never by watching the dots.

## 3. Repository map (as of c237ca6)

```
cayleypy_fast/
  __init__.py          # Public API: enable/disable/is_enabled/wrap, run_probe, ProbeResult
  _patch.py            # Guarded monkeypatch layer
  _probe.py            # Compatibility probe against the live cayleypy internals
  engine.py            # FastBeamEngine: chunked matmul-hash beam engine (all 4 modes)
  numba_kernels.py     # Optional numba CPU tier for the dual-int32 hash primitive
  pytest_plugin.py     # In-process enable() for host test suites
tests/                 # 118 tests: delegation, enable, engine_parity, hash_parity, numba_tier, probe
docs/radical-speedup-plan.md   # Design doc + measured T2/T4 benchmark tables
kaggle_benchmarks/     # 5 Kaggle CPU kernels: cpu, cpu_heavy, cpu_deep, cpu_sweep, cpu_micro
kaggle_out/            # Downloaded kernel logs + result JSONs (gitignored)
conftest.py            # Deterministic seeding (DETERMINISTIC_SEED = 12345)
lint.sh                # black 120 + pylint + mypy
```

### Code anchors (grep-verified at c237ca6)

| `engine.py` | Line | Role |
|---|---|---|
| `class PermutedHashVectors` | 112 | Permuted hash vector table `VH[j, g] = v[p_g^-1[j]]` |
| `build_permuted_hash_vectors` | 145 | Builds VH from the live `graph.hasher` |
| `hash_neighbors_tiled` | 170 | Tiled neighbor hashing; dispatches CPU dual-int32 to numba |
| `engine_available` | 200 | Gate: permutation group + dot-product hasher + probe |
| `class _GlobalTopK` | 210 | Running global topk (simple/advanced) |
| `class _PerGenTopK` | 292 | Per-generator running topk (iterated modes) |
| `class _BatchedRedistribute` | 323 | Legacy surplus redistribution (iterated_batched) |
| `class FastBeamEngine` | 388 | The engine; `_hash_tile_flat`:442, `_tile_candidates`:451, `_apply_cumulative_ban`:479, `_apply_window_ban`:488, `_make_score_fn`:500, `_materialize`:542, `_run`:560 |
| `search_simple/advanced/iterated/iterated_batched` | 750/781/812/843 | Forked mode methods |
| `create_engine` + per-graph cache | 906 | WeakKeyDictionary-keyed engine cache |

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
numba kernel inside `hash_neighbors_tiled`.

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
  `CAYLEYPY_FAST_NUMBA_DISABLE=1` (numba tier off, torch fallback).

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
- **Cycle** (~40–60 min per push → queue → run → download on CPU kernels):
  ```powershell
  kaggle kernels push -p kaggle_benchmarks/cpu          # or any other kernel dir
  kaggle kernels status ivankolt/cayleypy-cpu-bench
  # Download ONLY via the Python API (the `kaggle kernels output` CLI has a
  # Windows charmap bug when logs contain non-ASCII):
  python -c "from kaggle import KaggleApi; api=KaggleApi(); api.authenticate(); api.kernels_output('ivankolt/cayleypy-cpu-bench', path='./kaggle_out/cpu_bench_v3', force=True, quiet=True)"
  ```
- **Parallel-5 rule:** up to 5 kernels run in parallel on a free account —
  push all 5 at once.
- **Bump `_CAYLEYPY_FAST_REF` consistently across ALL 5 run.py files** when
  re-measuring (currently `0732d801addce60d0d2eafeb6efe51aa5b5c61f7`).
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
  `ivankolt/cayleypy-gpu-perf`, SHA TBD) belongs to **cayleypy-fast** (per the
  cayleypy AGENTS.md §10 two-kernel sync rule: copy + change install SHA and
  kernel id only). Kaggle allocates Tesla P100 (sm_60) → pin
  `torch==2.5.1+cu121` (default Kaggle torch crashes with
  `cudaErrorNoKernelImageForDevice`).

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
| T3 Triton GPU tier | TODO — needs the Kaggle GPU baseline/perf kernel pair |
| T5 hybrid NN predictors streaming scoring | TODO |
| T6 encoded-state path (spike-gated, low priority) | TODO |
| T7 scaling sweep 2^20…2^28 + GPU before/after + docs refresh | TODO |

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
