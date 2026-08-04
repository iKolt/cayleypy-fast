"""cayleypy-fast Triton sanity kernel (derisk BEFORE the full GPU perf kernel).

Validates on the Kaggle Tesla P100 (sm_60):
  1. torch 2.5.1+cu121 actually runs on sm_60 (Kaggle's default torch 2.10
     crashes with cudaErrorNoKernelImageForDevice);
  2. the engine's int64 Triton neighbor-hash kernel JIT-compiles on sm_60;
  3. its output is BIT-EQUAL to the mul+sum reference
     ``(states.to(int64)[:,None,:] * vh[None,:,:]).sum(-1)`` (mod-2^64 wrap).

The kernel source below MUST stay byte-identical in behavior to
``cayleypy_fast/triton_kernels.py`` (it is vendored here so the sanity kernel
has zero dependence on the cayleypy-fast install).

~5 min of GPU time. If this fails, the engine ships with the matmul/mulsum
ladder only (Triton deferred).
"""

import json
import subprocess
import sys
import time

# Pin torch 2.5.1+cu121 FIRST (sm_60 support), before any torch import.
subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "torch==2.5.1", "--index-url", "https://download.pytorch.org/whl/cu121"]
)

import torch  # noqa: E402

assert torch.cuda.is_available(), "GPU sanity kernel requires CUDA"
device_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"torch={torch.__version__}, device={device_name}, capability={capability}", flush=True)

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

print(f"triton={triton.__version__}", flush=True)


@triton.jit
def _hash_neighbors_int64_jit(
    states_ptr,
    vh_t_ptr,
    out_ptr,
    n_rows,
    n_gens,
    state_size,
    block_b: tl.constexpr,
    block_g: tl.constexpr,
    block_k: tl.constexpr,
):
    """out[i, g] = sum_k int64(states[i, k]) * vh_t[g, k] with mod-2^64 wraparound.

    Vendored copy of the kernel in cayleypy_fast/triton_kernels.py.
    """
    pid_b = tl.program_id(axis=0)
    pid_g = tl.program_id(axis=1)
    offs_b = pid_b * block_b + tl.arange(0, block_b)
    offs_g = pid_g * block_g + tl.arange(0, block_g)
    b_mask = offs_b < n_rows
    g_mask = offs_g < n_gens
    acc = tl.zeros((block_b, block_g), dtype=tl.int64)
    for k0 in range(0, state_size, block_k):
        offs_k = k0 + tl.arange(0, block_k)
        k_mask = offs_k < state_size
        x = tl.load(
            states_ptr + offs_b[:, None] * state_size + offs_k[None, :],
            mask=b_mask[:, None] & k_mask[None, :],
            other=0,
        )
        v = tl.load(
            vh_t_ptr + offs_g[:, None] * state_size + offs_k[None, :],
            mask=g_mask[:, None] & k_mask[None, :],
            other=0,
        )
        acc += tl.sum(x.to(tl.int64)[:, None, :] * v[None, :, :], axis=2)
    tl.store(
        out_ptr + offs_b[:, None] * n_gens + offs_g[None, :],
        acc,
        mask=b_mask[:, None] & g_mask[None, :],
    )


def triton_hash(states, vh_t, block_b, block_g, block_k=32, num_warps=4):
    n_rows, state_size = states.shape
    n_generators = vh_t.shape[0]
    out = torch.empty((n_rows, n_generators), dtype=torch.int64, device=states.device)
    grid = (triton.cdiv(n_rows, block_b), triton.cdiv(n_generators, block_g))
    _hash_neighbors_int64_jit[grid](
        states.contiguous(), vh_t, out, n_rows, n_generators, state_size,
        block_b=block_b, block_g=block_g, block_k=block_k, num_warps=num_warps,
    )
    return out


def mulsum_reference(states, vh_t):
    """(B, n) x (G, n) -> (B, G), mul+sum, mod-2^64 wrap."""
    return (states.to(torch.int64)[:, None, :] * vh_t[None, :, :]).sum(-1)


torch.manual_seed(12345)
checks = []
# (n_rows, n_gens, state_size, dtype, block_b, block_g): small/large shapes,
# both input dtypes, both config buckets, edge masks (odd sizes).
for n_rows, n_gens, state_size, dtype, block_b, block_g in [
    (8, 3, 8, torch.int64, 64, 4),
    (257, 12, 54, torch.int8, 64, 4),
    (2**14, 3, 32, torch.int8, 128, 4),
    (2**16 + 1, 48, 150, torch.int8, 64, 4),
    (3, 7, 150, torch.int64, 64, 4),
]:
    # Values mimic permutation indices (< state_size); int8 tops out at 127.
    hi = min(state_size, 100) if dtype == torch.int8 else state_size
    states = torch.randint(0, hi, (n_rows, state_size), dtype=dtype, device="cuda")
    vh = torch.randint(-(2**62), 2**62, (state_size, n_gens), dtype=torch.int64, device="cuda")
    vh_t = vh.t().contiguous()
    t0 = time.perf_counter()
    got = triton_hash(states, vh_t, block_b, block_g)
    torch.cuda.synchronize()
    compile_sec = time.perf_counter() - t0
    ref = mulsum_reference(states, vh_t)
    bit_equal = bool(torch.equal(got, ref))
    checks.append(
        {
            "shape": [n_rows, n_gens, state_size],
            "dtype": str(dtype),
            "config": [block_b, block_g],
            "bit_equal": bit_equal,
            "first_call_sec": compile_sec,
        }
    )
    print(
        f"shape=({n_rows},{n_gens},{state_size}) dtype={dtype} cfg=({block_b},{block_g},32) "
        f"bit_equal={bit_equal} first_call={compile_sec:.2f}s",
        flush=True,
    )

# Speed smoke: benchmark-scale tile (2^16 x 12 gens x 54) triton vs mul+sum.
n_rows, n_gens, state_size = 2**16, 12, 54
states = torch.randint(0, state_size, (n_rows, state_size), dtype=torch.int8, device="cuda")
vh_t = torch.randint(-(2**62), 2**62, (state_size, n_gens), dtype=torch.int64, device="cuda").t().contiguous()


def timed(fn, runs=5):
    fn()  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs


t_triton = timed(lambda: triton_hash(states, vh_t, 64, 4))
t_mulsum = timed(lambda: mulsum_reference(states, vh_t))
print(f"speed smoke ({n_rows}x{n_gens}x{state_size}): triton={t_triton*1e3:.2f}ms mulsum={t_mulsum*1e3:.2f}ms", flush=True)

ok = all(c["bit_equal"] for c in checks)
result = {
    "ok": ok,
    "device": device_name,
    "capability": list(capability),
    "torch": torch.__version__,
    "triton": triton.__version__,
    "checks": checks,
    "speed_smoke": {"triton_ms": t_triton * 1e3, "mulsum_ms": t_mulsum * 1e3},
}
with open("gpu_sanity_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print(f"Wrote gpu_sanity_result.json ok={ok}", flush=True)
assert ok, "Triton int64 kernel NOT bit-equal to mul+sum reference on this device"
