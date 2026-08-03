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

**T3 — Triton tier.** Kernel + autotune + fallback matrix; sm_60 validation on
Kaggle; bit-equality tests; perf check vs T-eng CUDA.

**T4 — numba tier.** Parallel fused chunk kernel; int8/int64 states; bit-equality
tests; CPU benchmark re-run; per-device default tier table finalized.

**T5 — Hybrid predictors + iterated parity hardening.** Streaming per-generator
NN/custom scoring w/ running topk; tiny-MLP smoke test (trained in-test) on both
devices; parity per T1 harness.

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
