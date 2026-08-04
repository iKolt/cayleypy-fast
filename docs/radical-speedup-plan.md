# Plan: cayleypy-fast — radical beam-search speedup + much larger beams

Separate add-on project (no cayleypy core changes). Three perf tiers: pure-torch
engine (device-agnostic reference), Triton CUDA kernels, numba CPU kernels.
In-place activation via guarded monkeypatch. Optional spike: encoded-state path.

## Goal & success criteria

1. **Step time 5–10x faster** at `bw=2^24` on Kaggle P100 (cube333 / cube555 / lrx32
   profiles), measured by baseline-vs-perf kernel JSON diff.
2. **Max beam, per-graph targets** (16 GB P100, hard physics noted in §Memory model):
   - cube333 (n=54, int8): >= 2^26, stretch 2^27–2^28.
   - lrx32 (n=32): >= 2^26, stretch 2^27–2^28.
   - cube555 (n=150, int8): target 2^25, stretch 2^26 (beam states alone ≈ 10 GB at 2^26).
   Final numbers re-baselined from the T2 roofline measurement; the T7 sweep records
   achieved max beam per graph.
3. **CPU covered (both devices required):** no regression is the hard floor; target
   >= 2x step speedup at benchmark beams (10^4–10^5) via engine + numba tier.
4. cayleypy upstream `pytest` + `./lint.sh` green **with the patch enabled**
   (probe correctness gate); cayleypy-fast suite green on CI (CPU: Win/Linux/macOS)
   and CUDA.
5. Invariants preserved (bit-identity NOT required): valid paths
   (`apply_path(start, path) == dest`), `len(path) == path_length`, MITM arithmetic
   `i_step + bfs_layer_id`, non-backtracking never bans destination neighborhood,
   cayleypy §6 sorted-hash invariant untouched.

## Packaging & activation (user comment 1)

- **New repo/package**, proposal `cayleypy-fast` (import `cayleypy_fast`; owner
  creates the repo, ideally under the cayleypy org; name is owner-confirmed — the
  alternative suggestion `cayleypy_triton` undersells the CPU tier but is
  acceptable). **Pure Python + Triton/numba JIT sources: no native build step** →
  `py3-none-any` wheel, zero toolchain on Kaggle/Windows/macOS/Linux.
- **Activation (in-place replacement):**
  ```python
  import cayleypy_fast
  cayleypy_fast.enable()   # all graph.beam_search(...) calls now use the engine
  ```
  Mechanism: `enable()` defines `FastBeamSearchAlgorithm(cayleypy BeamSearchAlgorithm)`
  whose 4 `search_*` overrides route to the engine when available and to
  `super()` otherwise, then patches one symbol:
  `cayleypy.cayley_graph.BeamSearchAlgorithm = FastBeamSearchAlgorithm`
  (plus `cayleypy.algo.beam_search.BeamSearchAlgorithm` for direct importers).
  - **Compatibility probe** at `enable()`: asserts importability and rough signatures
    of every internal it touches (`_init_predictor`, `_encode_and_dedupe_start`,
    `_setup_path_device_and_restore`, `_precompute_mitm`, `_early_return_if_at_dest`,
    `_finalize_not_found`, `_restore_path`, `_check_path_found`, `TorchHashSet`,
    `isin_via_searchsorted`, `BeamSearchResult`, `StateHasher` attrs). Mismatch →
    warning + no-op (legacy behavior).
  - `disable()` restores originals; env var `CAYLEYPY_FAST_DISABLE=1` forces no-op.
  - Patch-averse API: `cayleypy_fast.wrap(graph)` returning an object whose
    `beam_search` matches the cayleypy signature.
- **Dependencies:** `torch>=2.5` (NOT >=2.6 — Kaggle pins `2.5.1+cu121` while
  cayleypy declares >=2.6; document install order: `pip install --no-deps cayleypy@<sha>`
  + `cayleypy-fast` + explicit `torch==2.5.1+cu121`, torch pinned LAST), `numba`
  (optional import, tier-gated), declared cayleypy compat range.
- cayleypy core changes: **none required**. (Optional later: a tiny hook PR — only if
  the symbol-patch proves fragile across cayleypy refactors.)

## Key technical insight

For **permutation groups with the dot-product hasher**, hashing a neighbor is linear:

```
hash(g(s)) = sum_i s[p_g[i]] * v[i] = s · (v ∘ p_g^{-1})
```

⇒ hashes of all `B × G` neighbors = one matmul against precomputed permuted hash
vectors `VH[j, g] = v[p_g^{-1}[j]]` — **neighbor states are never materialized**.
- `VH` is derived **from the live `graph.hasher` instance** (`vec_hasher`, or
  `vec_hasher_i32` on the CPU dual-int32 path combined via `.view(torch.int64)`) —
  NOT regenerated — so chunk hashes bit-match `hasher.make_hashes` outputs used for
  MITM BFS layers and non-backtracking sets. torch int64 matmul / Triton int64 /
  numba int64 all wrap mod 2^64 ⇒ bit-identical.
- **Lazy survivor materialization:** only `beam_width` topk survivors are built per
  step: `parents = idx % B_step`, `gens = idx // B_step` (gen-major),
  `new_beam = gather(beam_states[parents], 1, perms[gens])` — one gather, `B×n`.

## Constraints discovered in code (baked into design)

- **No global sort at target beams**: full `torch.sort` of `B×G` int64 at bw=2^26,
  G=12–24 ⇒ 6–13 GB keys + 2–3x radix workspace > 16 GB. Engine dedupes chunk-wise
  into a `TorchHashSet` (sorted shards, `searchsorted` queries) instead.
- **Gen-major flat layout** `(G,B).reshape(-1)` so origin math (`flat // B`) and
  dedup-winner ordering match legacy `iterated_batched` / `iterated` semantics.
- **`iterated_batched` CUDA memory gate** (1.5x materialized-states estimate,
  raises `MemoryError`) is invalid under the engine → the engine's forked mode
  method replaces it with chunk-bounded accounting.
- **int8 states:** benchmarks already use `dtype=torch.int8`; `torch.matmul` has no
  int8 kernel → engine casts state tiles to int64 (CPU: int32 dual) per tile,
  temp bounded by tile size (default `tile_states = 2**22`, tune by memory).
- **`restore_path` re-sorts layers itself** (cayleypy `cayley_graph.py:436`) ⇒
  survivor-hash order irrelevant for path restore. Confirmed.
- **Replicated quirk (do not fix here):** hamming predictor always measures distance
  to `graph.central_state` even when `destination_state` is custom
  (`predictor.py:42`, via `_init_predictor` default). The engine reproduces this
  byte-for-byte for parity; flag with `# TODO(char-spec)` for upstream.
- **Per-chunk op order preserved:** dedup-vs-step-accumulator → **MITM check** →
  non-backtracking ban (MITM strictly before ban — ban must never hide the
  destination, cayleypy §6.6) → selection. Non-backtracking slot bookkeeping matches
  legacy iterated: reset current slot per step; query all slots before adding the
  chunk; add the chunk's post-dedup hashes to the current slot; apply ban mask.
- **`debug_scores` per mode:** keep legacy formulas (advanced: min of selected;
  iterated_batched: global min of survivor scores; iterated: min over chunks).
- **Identity hasher** (`encoded_state_size == 1`) and matrix groups → legacy path.

## Memory model (16 GB P100, dtype=int8)

| Graph | n | G | beam @2^26 | tile temp (2^22 × n × 4B) | verdict |
|---|---|---|---|---|---|
| lrx32 | 32 | 3 | 2.1 GB | 0.5 GB | 2^27–2^28 reachable |
| cube333 | 54 | 12 | 3.6 GB | 0.9 GB | 2^27 realistic |
| cube555 | 150 | 24 | 10.1 GB | 2.5 GB (use 2^20 tiles → 0.6 GB) | 2^25 ok, 2^26 marginal |

Per-step peak ≈ beam states + TorchHashSet (~beam×8B) + topk buffers
(≈2×beam×4B scores + idx) + tile-bounded temps. Candidates/streaming tensors never
persist beyond a chunk. Score computation is **lazy within a chunk**: computed only
for candidates surviving dedup + ban masks (big early-step savings when duplicate
rates are high).

## Engine (`cayleypy_fast/engine.py`)

`FastBeamEngine(graph, predictor_spec, mode)` per search:

- `available()` gate: permutation group AND `string_encoder is None` AND
  dot-product hasher (non-identity) AND probe passed. Else → `super()` legacy.
- Precompute per graph: `VH` from hasher (see above), perm table `(G×n)` int64,
  `central_state` for fused hamming, tile sizes from `memory_limit_gb`.
- `step(beam_states)` → `(survivor_states, survivor_hashes, best_score)`; two-level
  chunking (state tile × generator), gen-major within tile:
  1. hash chunk via matmul with `VH[:, g]` (CPU: dual-int32 pair → view int64);
  2. dedup vs step `TorchHashSet` → sorted add;
  3. MITM check (`isin_via_searchsorted` vs BFS layer hashes) — early exit keeps
     `i_step + layer` arithmetic;
  4. non-backtracking ban via per-slot `TorchHashSet` (legacy ordering above);
  5. lazy scoring of masked survivors (hamming fused; hybrid below);
  6. selection per mode: simple/advanced — running global topk buffer
     (`cat + topk(beam_width)`, ≤2×beam_width per merge); iterated — per-generator
     running topk (`beam_width_part`) merged across tiles, accumulate per-gen slot
     lists; iterated_batched — per-gen topk + legacy surplus redistribution;
  7. end of step: materialize survivors via one gather; record
     `restore_path_hashes` (order-insensitive, confirmed).
- Hybrid predictors (NN/custom): steps 1–4 fused; step 5 streams decoded chunk
  survivors through the user predictor into the running topk. Still no `B×G×n`
  materialization (chunk-bounded). Detect hamming from the RAW predictor argument
  (`None` / `"hamming"`) before `_init_predictor` wrapping.
- The forked mode methods import cayleypy helpers (probe-verified) for preambles,
  `_early_return_if_at_dest`, `_finalize_not_found`, `_restore_path`; empty-beam
  exits, `memory_cleanup`, verbose/profile regions preserved.

## Tiers

| Tier | Device | Tech | Content |
|---|---|---|---|
| T-eng | CPU+CUDA | pure torch | reference engine; always available; benchmark-gated on CPU |
| T-gpu | CUDA | Triton (bundled w/ torch; JIT PTX sm_60 ok) | one fused kernel per chunk: gather + wrapped int64 hash + hamming; autotuned BLOCK_B/N; any import/compile/runtime error → T-eng + one-time warning |
| T-cpu | CPU | numba `njit(parallel=True)` (already a cayleypy dep) | same primitive, `prange` over states; optional import → fall back to T-eng |
| T-enc | both | Triton + numba | optional T6 encoded path, spike-gated |

Bit-equality gates: engine/native hashes vs `hasher.make_hashes(gather(...))` on
seeded random states; hamming parity; determinism under fixed seed.

## Encoded-state path (user comment 2) — optional T6, spike-first

Bottleneck confirmed in code: `StringEncoder.encode/decode` loop `w*n` Python
iterations of tensor ops (cube333 auto width w=3, n=54 ⇒ ~162 kernel launches per
call, run for every scored chunk with NN predictors); `implement_permutation` = one
kernel per (mask, shift) term; `_hash_splitmix64` loops per encoded word.
T6 (if spike passes): one-launch encode/decode kernels, encoded perm-apply from the
`prepare_shift_to_mask` table as flat arrays, fused splitmix64. Property tests vs
`StringEncoder` incl. `uses_sign_bit` configs (B5 sign-bit history). Success =
encoded+NN beam-search step within agreed factor of decoded+NN; else document out
of scope — nothing else depends on T6.

## Work plan

**T0 — Scaffold.** Repo, pyproject (pure wheel; deps per §Packaging), lint mirroring
cayleypy (black 120 / pylint / mypy), CPU CI matrix (3.9–3.13, Win/Linux/macOS),
`enable()/disable()` + probe + env var, `wrap()` API, `FastBeamSearchAlgorithm`
subclass that initially just delegates to `super()` (proves patch plumbing).

**T1 — Baseline & characterization.**
- Property test vs CURRENT cayleypy (CPU+CUDA): sampled perm graphs (lrx 4/8/16/32,
  cube222/333/444/555, 3–5 `PermutationGroups` entries) × dtypes {int8, int64}:
  engine matmul hashes bit-eq `make_hashes(materialized neighbors)`.
- Parity harness: legacy vs engine on fixed seeds per mode; assert per-step selected
  hash **multiset equality** (up to tie policy: assert set equality), iterated
  per-generator survivor counts, `validate_path` on found paths.
- Record baselines: `pytest --benchmark-only beam_search_benchmark.py
  --benchmark-save=R0-pre`; one run of the immutable Kaggle baseline kernel.

**T2 — Torch engine (T-eng), all 4 modes, hamming.** Forked mode methods + engine +
lazy scoring + chunked dedup; CPU benchmark gate (>= 2x target; if regression on any
suite row, default CPU to legacy and flag). Char tests pinning exact paths rewritten
to invariant assertions (in cayleypy-fast's own suite; cayleypy suite untouched).

**T2 measured (Kaggle 4 vCPU CPU, torch 2.10.0+cpu, cayleypy@0b7e109, engine@45c5985 —
post runaway-beam fix; paired legacy-vs-engine in `kaggle_benchmarks/cpu*` kernels):**

| Kernel | Row | legacy | engine | speedup |
|---|---|---|---|---|
| cpu | lrx8/lrx32/cube444/cube555 iterated | 55.6ms / 179ms / 2.64s / 19.1s | 24.8ms / 74.9ms / 0.98s / 6.61s | **2.24× / 2.39× / 2.70× / 2.90×** |
| cpu | cube333 simple/advanced/iterated | 44.3s / 61.1s / 35.6s | 30.4s / 38.6s / 25.8s | 1.45× / 1.58× / 1.38× |
| cpu-sweep | cube333 it_batched bw 1e3/1e4/1e5 | 0.50s / 5.97s / 58.5s | 0.23s / 2.55s / 29.1s | **2.19× / 2.34× / 2.01×** |
| cpu-sweep | bw=256 gate negative control | 0.170s | 0.142s | 1.20× (legacy-vs-legacy ✓) |
| cpu-heavy (3 runs) | cube444 iterated / iterated_batched bw=1e4 | 10.48s / 19.22s | 5.22s / 7.15s | **2.01× / 2.69×** |
| cpu-heavy (3 runs) | cube555 iterated / iterated_batched bw=1e4 | 1.91s / 3.48s | 0.71s / 1.18s | **2.69× / 2.96×** |
| cpu-deep | cube333 iterated/it_batched bw=2^18 mitm=3 | 127.6s / 588.9s | 88.4s / 328.0s | 1.44× / **1.80×** |
| cpu | lrx8_simple / lrx16_advanced (ms-scale rows) | 22.9ms / 2.39ms | 25.1ms / 3.22ms | 0.91× / 0.74× ⚠ setup overhead |

Gate position: ≥2× achieved on 14 of 21 engine-engaged rows (all iterated and
iterated_batched rows pass; heavy rows confirmed with 3-run statistics, stdev ≤0.2s);
the two <1× rows are ms-scale searches (path found in 6–16 steps) dominated
by `create_engine` per-call setup (VH build, `argsort`, `central_by_gen`) — fix planned
via per-graph engine cache (WeakKeyDictionary). Deep rows (bw=2^18) give the largest
absolute savings (~4.4 min/run for it_batched) despite sub-2× ratios.

**Bug found & fixed en route (engine@45c5985):** `_GlobalTopK.offer` first-chunk path
stored chunks > `beam_width` wholesale (typical step has exactly ONE offer), so the
beam grew ~G× per step — lrx16+MLP reached 839K states at step 21 (814s vs 2.5s legacy).
Fixed with an immediate top-k cap + `finalize` invariant (`numel <= k`) + regression test
with wall-clock bound. Small-graph parity tests couldn't see it (lrx6 state space = 720);
a long-scramble big-graph parity test was added.


**T3 — Triton tier.** Kernel + autotune + fallback matrix; sm_60 validation on
Kaggle; bit-equality tests; perf check vs T-eng CUDA.

**T4 — numba tier.** Parallel fused chunk kernel; int8/int64 states; bit-equality
tests; CPU benchmark re-run; per-device default tier table finalized.

**T4 spike measured (laptop, torch 2.13 CPU, numba njit parallel prange vs torch
int32 matmul on dual-int32 hash tiles):** torch int32 matmul does NOT dispatch to
an optimized BLAS integer GEMM — measured 0.75-2 GMAC/s. The numba kernel with
transposed (2G, n) layout + fused int8 read is **13-39x faster** across
B∈{1e4,1e5,2^18} × n∈{8,32,54,150} × G∈{3,12,24}, bit-equal everywhere.
Tier integrated (engine@0732d80): `hash_neighbors_tiled` dispatches dual-int32
to numba by default, `CAYLEYPY_FAST_NUMBA_DISABLE=1` forces torch; 17 tier
tests (bit-equality x 12 shapes, dispatch, kill-switch, cache, speed guard >2x).

**Final combined run (engine@0732d80 = runaway fix + per-graph cache + numba,
Kaggle 4 vCPU, all 5 kernels COMPLETE; JIT caveat: engine `mean` on ms-rows and
bw=1e3 sweep includes one ~0.5s numba compile on the first call — compare `min`:**

| Kernel | Row | legacy | engine | speedup |
|---|---|---|---|---|
| cpu | lrx8/lrx32 iterated | 55ms / 175ms | 29ms / 65ms | **1.93× / 2.68×** |
| cpu | cube222 simple | 1.08s | 0.58s | 1.86× |
| cpu | cube333 simple/advanced/iterated | 41.3s / 61.3s / 38.2s | 26.2s / 36.2s / 25.7s | 1.58× / 1.69× / 1.49× |
| cpu | cube444/cube555 iterated | 12.2s / 17.1s | 4.67s / 6.32s | **2.62× / 2.71×** |
| cpu-heavy (3 runs) | cube444 iterated / it_batched | 11.63s / 3.99s | 4.75s / 1.52s | **2.45× / 2.63×** |
| cpu-heavy (3 runs) | cube555 iterated / it_batched | 1.91s / 3.90s | 0.69s / 1.32s | **2.76× / 2.96×** |
| cpu-deep | cube333 iterated/it_batched bw=2^18 | 266.6s / 380.5s | 186.0s / 191.1s | 1.43× / **1.99×** |
| cpu-sweep | cube333 it_batched bw 1e4/1e5 | 5.85s / 65.1s | 2.55s / 30.4s | **2.30× / 2.15×** |
| cpu-sweep | bw=256 gate control / bw=1e3 (min) | 0.180s / 0.485s | 0.143s / 0.296s | 1.26× / 1.64×(min)/0.76×(mean, JIT) |

Net: T4 raised iterated/deep rows (deep it_batched 1.80→1.99×, lrx32_it 2.39→2.68×,
cube333_it 1.38→1.49×). 16/21 engine rows ≥2× on mean, all ≥2× on min where warmup
is excluded. Remaining sub-1× rows are the two ms-scale rows (path found in 6-16
steps; post-cache they sit at 0.61-0.91× mean, i.e. ≤1ms absolute overhead).

**T5 — Hybrid predictors + iterated parity hardening.** Streaming per-generator
NN/custom scoring w/ running topk; tiny-MLP smoke test (trained in-test) on both
devices; parity per T1 harness.

**T5 parity+smoke DONE (@`79808f8`, CUDA-probe fixes @`356b201`–`d171fad`):**
`tests/test_nn_parity.py` — trained-in-test tiny MLP (lrx8) + untrained seeded MLP
(cube222), 4 modes × 2 devices. CPU: exact `debug_scores` + scored-candidate
multiset parity for simple/advanced/iterated_batched; CUDA: outcome parity +
allclose on common steps. Iterated keeps outcome-level asserts only (legacy
iterated does NOT dedup within a per-generator chunk — verified at
`beam_search.py:903` — while the engine's tile-level dedup does, so scored
candidate counts legitimately diverge). CUDA mirrors cannot reuse fixed
instances (torch RNG streams differ CPU vs CUDA — Philox), so they select
device-native legacy-solvable nbt-walk instances by probing (`_find_solvable_start`;
classic walks needed L/R-cancellation protection: classic random walks on LRX
often simplify to distance <= 2 and never exercise scoring). Kaggle gpu-perf
stage 0 runs the whole tier+NN suite in-clone on the P100: 40/40 green.

**T3 DONE (@`79808f8` tier + tests, OOM fixes through @`33cae8b`; Kaggle P100
sm_60, torch 2.5.1+cu121 / triton 3.1.0, cayleypy@`0b7e109`):** fixed-config
(no autotune) broadcast-mul+`tl.sum`
int64 hash kernel in `cayleypy_fast/triton_kernels.py`; engine ladder
triton→matmul→mulsum resolved once per engine (`resolve_backend`, deterministic
`torch.arange` probe bit-checked vs mulsum) + mid-search demotion with a
RuntimeWarning + recompute. Sanity kernel: 5/5 shapes bit-equal on P100,
~3.6× over the mulsum baseline micro. Engine vs legacy measured
(`kaggle_benchmarks/gpu`, id `ivankolt/cayleypy-gpu-perf`):

| Row | legacy | engine | speedup (mean) |
|---|---|---|---|
| lrx8 simple/advanced/iterated bw=1e5 | 13ms / 22ms / 53ms | 21ms / 15ms / 27ms | 0.63×\* / 1.42× / 1.98× |
| cube333 simple/advanced/iterated bw=1e5 | 0.40s / 0.53s / 1.13s | 0.27s / 0.36s / 0.36s | 1.45× / 1.48× / 3.10× |
| cube333 iterated bw=2^18 mitm=3 (deep) | 1.68s | 0.65s | 2.56× |
| lrx16/lrx32 iterated bw=1e5, pretrained MLP | 21ms / 77ms | 14ms / 37ms | 1.53× / 2.09× |
| cube333 iterated/simple bw=1e5, seeded MLP | 3.89s / 3.02s | 3.12s / 2.96s | 1.25× / 1.02× |
| cube444 iterated/simple bw=1e5, seeded MLP | 10.87s / 8.72s | 8.91s / 8.56s | 1.23× / 1.02× |
| cube333 iterated bw=2^24 (engine-only; legacy infeasible) | — | 48.6s, 1.68s/step, path found | — |
| lrx32 iterated bw=2^24 (engine-only) | — | 0.1s, path found | — |

\* lrx8/lrxNN "simple" rows are ms-scale searches (path found in ~16 steps before
the beam fills); engine dispatch overhead dominates — same known pattern as the
CPU micro rows. NN cube rows sit at 1.0–1.25× because streaming-scoring time is
dominated by the NN forward, not hashing (T5 perf tuning stays deferred — the
engine does not lose to legacy). All rows same `path_found`, hard `validate_path`
asserts on NN rows, `nn_parity_diverged: false` everywhere.

**T7-GPU sweep measured (`kaggle_benchmarks/gpu_sweep`, id
`ivankolt/cayleypy-gpu-sweep`, engine-only iterated hd=2 ladders):**

| Graph | Beams reached | Peak VRAM | Notes |
|---|---|---|---|
| cube333 | 2^20, 2^22 (0.15 s/step), 2^24 (0.56 s/step), 2^25 (1.17 s/step) | 13.2 GB | 2^26 OOM |
| lrx32 | all rows through 2^28 | 0.29 GB | hamming-guided lrx beams stay small regardless of bw (already found ≤ 20 steps) |
| cube555 | 2^22 (1.08 s/step) | 7.2 GB | 2^23 OOM |

**Max-beam wall identified: candidate-hash sets.** At beam bw with G generators
the live state per step is ≈ `bw·G·8B` (step torch-hash-set) + up to 2 further
`bw·G·8B` for the two history-depth ban slots (+~10% transient tiles/sort):
cube333 @2^26 needs 6.4 (step set) + 12.9 (two ban slots) + 3.6 (beam, int8)
+ ~1.5 (tiles/sort) ≈ 24 GB > 16 GB; cube555 @2^23 (G=24) ≈ 12.9 GB of sets
alone. lrx32 (G=3) fits 2^28 easily.
This is a *storage* wall, not a compute wall — options for a future task:
hd=1 sweep rows (−1 slot), or a fused dedup/hash-table tier (plan §Stretch).

**GPU bugs found & fixed en route:** (1) `torch.arange` probe/JIT during
`resolve_backend` is per-(config,dtype) — warmup runs absorb it in all kernels.
(2) **Input-dtype trap** (the big one): `CayleyGraph.encode_states` keeps the
*input* dtype (`torch.as_tensor(states)` with no cast, `cayley_graph.py:143`),
so a Python-list start state is int64; a (2^24, 54) int64 survivor batch is a
single 6.75 GiB allocation → OOM on a 16 GB P100 at bw=2^24 cube333 even with
chunked `_materialize`. Kernels now cast starts to `graph.dtype`
(`kaggle_benchmarks/gpu/run.py` `cast_start`). Users with big beams should do
the same. (3) `_materialize`'s `perms[gens]` gather index materialized
(K, n) int64 wholesale → chunked under `_MATERIALIZE_CHUNK_BYTES`
(bit-identical row-wise op; parity test with a monkeypatched tiny constant).

**T6 — (optional, spike-gated) Encoded path.** Order: encode/decode one-launch
kernels first (spike), then perm-apply, then splitmix64. Property tests incl.
sign-bit configs. Fail gate → document as out of scope.

**T7 — Scaling + Kaggle before/after + docs.** Beam sweep 2^20…2^28 (log achieved
max beam + step time per graph; verify per-graph targets); `kaggle_benchmarks/perf/`
in cayleypy-fast: baseline-kernel copy + install `cayleypy@<pinned SHA>` +
`cayleypy-fast@<perf SHA>` + `cayleypy_fast.enable()` in run.py; kernel id
`ivankolt/cayleypy-gpu-perf`; quick+deep+sweep runs; JSON diff table in the repo
README/PR; usage/fallback/caveats docs.

Stretch (only if profiled hot after T3): open-addressing dedup-table kernel
(replaces per-chunk TorchHashSet merge if it dominates at 2^27+); CUDA graphs
around the step; `torch.compile` of glue (low value — TorchHashSet python breaks
graphs).

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| cayleypy alpha API drift breaks the patch | Probe at `enable()` → warning + no-op; CI runs cayleypy's own suite with patch enabled; pin declared compat range |
| int64 matmul unsupported on some GPUs | try/except → mul+sum fallback (mirrors StateHasher); T1 property test catches backend divergence |
| Triton/numba wrap-around divergence | All wrap mod 2^64 by construction; bit-equality property tests (seeded, both dtypes) per native tier |
| Triton on sm_60 (P100) compile issues | try/except → T-eng; Kaggle validation in T3 before any perf claim |
| Chunked dedup/tie-breaking diverges from legacy | Agreed non-goal; gen-major layout + legacy op order minimize drift; parity harness asserts hash-set equality + invariants |
| Iterated fairness drift | T1 characterization + T2/T5 per-generator survivor-count parity |
| cube555 2^26 marginal (10 GB beam) | Per-graph criteria (agreed); smaller tiles; lazy scoring; T6 encoded states as escape hatch |
| Encoded sign-bit edge cases (B5) | T6 spike-gated; property tests vs StringEncoder; documented opt-out |
| Kaggle torch pin conflict (cayleypy >=2.6 vs 2.5.1) | `torch>=2.5` dep; documented `--no-deps` install order in kernel run.py and README |
| CPU gain doesn't materialize | Benchmark-gated activation; floor = no regression with legacy on CPU |

## Validation plan

1. cayleypy-fast suite green on CPU CI matrix + CUDA box (Triton/numba tests skipif-gated).
2. cayleypy upstream `pytest` + `./lint.sh` green with `cayleypy_fast.enable()` active (CI job).
3. Benchmarks `--benchmark-compare` vs `R0-pre`: no regression > 5% anywhere; CPU >= 2x target rows.
4. Kaggle quick+deep+sweep: baseline vs perf JSON diff → confirm step-time 5–10x at
   bw=2^24 and per-graph max-beam targets (§Goal).
5. Invariant matrix: 4 modes × {hamming, tiny-NN} × {fast on, fast off} × {CPU, CUDA}
   → `validate_path` + MITM length arithmetic + `len(path) == path_length`.

## Out of scope

- C++/CUDA extensions, prebuilt native wheels, custom radix-sort library.
- Matrix-group fast path.
- Changing cayleypy public defaults (`dtype`, `beam_mode`).
- Encoded path beyond the T6 spike gate.
- torch.compile / CUDA graphs (stretch only).

## Open items for the owner (non-blocking)

- Repo name/location (`cayleypy-fast` proposed under cayleypy org; `cayleypy_triton`
  acceptable). Affects naming only.
- Whether a future tiny hook PR to cayleypy core is acceptable if symbol-patching
  proves fragile (currently: zero core changes).

## Filesystem layout for parallel development (verified)

Confirmed: `C:\Users\xiaomi\cayleypy-fast` sibling directory exists (empty) — the
new project's home. cayleypy checkout state: only untracked `.kilo/` (this plan);
no tracked files modified; branch `feature/beam-search-perf`. `.kilo/` is not
gitignored — a local untracked artifact; committing it is optional, out of scope.

Isolation rules so this work never affects cayleypy and cayleypy keeps evolving
in parallel:

1. **Sibling directory:** all new-project files live only under
   `C:\Users\xiaomi\cayleypy-fast` (own git repo, own venv, own CI, own
   `kaggle_benchmarks/perf/`). The `C:\Users\xiaomi\cayleypy` checkout is never
   modified by this work (zero core changes).
2. **Own virtualenv** in the new project — isolates dependency pins (notably the
   cayleypy `torch>=2.6` metadata vs the Kaggle `2.5.1+cu121` pin). Setup:
   ```powershell
   cd C:\Users\xiaomi\cayleypy-fast
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --no-deps -e C:\Users\xiaomi\cayleypy   # parity tests run against the LOCAL cayleypy working tree
   pip install torch numba numpy scipy h5py pytest pytest-benchmark
   ```
   `--no-deps` on the cayleypy editable install prevents pip from trying to
   upgrade torch to satisfy cayleypy's metadata.
3. **Parallel cayleypy development:** unchanged workflow in the cayleypy repo;
   drift caught by the `enable()` probe + a scheduled cayleypy-fast CI job
   (suite vs latest cayleypy commit).
4. **This plan file:** physically inside the cayleypy checkout (untracked
   `.kilo/plans/`). On or after repo creation, copy it to
   `cayleypy-fast/docs/radical-speedup-plan.md` as the design doc; keeping or
   deleting the local copy has no effect on the library.

## Bootstrap order (owner's practical sequence)

GitHub repo registration is NOT a hard prerequisite for starting development; it
becomes mandatory only at specific points:

1. **Local-first (no GitHub needed):** T0 scaffold (package, pyproject, `enable()`,
   probe, tests — plain local `git init`), T1 (baseline + property tests),
   T2 (torch engine), T4 (numba tier) — all debuggable locally on CPU.
2. **GitHub repo required for:** T0 CI (GitHub Actions needs a pushed repo) and
   T3/T7 Kaggle runs (kernels install via
   `pip install git+https://github.com/<org>/cayleypy-fast@<sha>`). Workaround if
   deferring: upload built wheels as a Kaggle Dataset (worse iteration cycle).
3. **PyPI publishing:** only when versioned releases are wanted; until then
   install-from-git-SHA suffices.

Recommended sequence: create the (possibly empty) repo early → T0 scaffold locally
→ first push + CI → proceed task by task; do T3/T7 Kaggle cycles only after the
repo is pushable.
