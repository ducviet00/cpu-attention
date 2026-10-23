# C++ backend — state, numbers, and what did not work

`cpu_attention/csrc/` is a second backend beside the Mojo kernel, with the same
Python API (`flash_attn_func`, `flash_attn_with_kvcache`, and the engine aliases
`self_attention` / `cross_attention` / `reshape_and_cache`).
`CPU_ATTENTION_BACKEND=mojo|cpp` selects one; both shared objects load in one
process, and `tests/bench_cpp.py` alternates them for an in-process A/B.
The default is still `mojo`.

```bash
uv run --extra build python scripts/build_cpp.py          # -> cpu_attention/_C_cpp.so
CPU_ATTENTION_BACKEND=cpp uv run python tests/test_dense.py
CPU_ATTENTION_BACKEND=cpp uv run python tests/test_flash_api.py
CPU_ATTENTION_BACKEND=cpp uv run python tests/fuzz_dense.py --cases 80 --max-seq 900
```

## Build

One artifact per architecture, baseline ISA plus runtime dispatch (CPUID /
`arch_prctl` on x86, HWCAP on aarch64). No `-march=native`.

| TU | flags |
|---|---|
| everything else | `-O3 -fPIC -std=c++20 -fopenmp` |
| `vec.cpp` | + `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vnni -mfma -mavx2` |
| `amx.cpp` | the vec flags + `-mamx-tile -mamx-bf16 -mamx-int8` |
| `neon.cpp` | `-march=armv8.2-a+bf16+i8mm` |

AMX needs `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` before the first
tile instruction; `cpu_isa.cpp::amx_init()` does it once, from the plan build.

## Design

`cpu_attn_impl.hpp` is the shared core: the job descriptions, the per-thread
workspace, and the two loop skeletons. Each ISA provides a *policy* — the packed
K/V layout, `qk`, `softmax`, `pv`, and the paged `store_head`/`attend_head`.

* **Dense work item = (batch, q head, `rq` consecutive query rows).** Inside it
  the key range is walked in tiles of `kt` and the rows in chunks of `RQT`. K/V
  are read once per `rq` rows instead of once per 8, which is what the long
  D=128 rows are bound by. Two dispatch shapes: *fused* (a thread packs the head
  it then attends, no barrier, K/V in its own L2) and *split* (every head packed
  once, concurrent threads sharing one head's slab through L3). Fused wins below
  `workers * slab <= 0.6 * LLC`, split above it — at 4096:16:128 fp32 the fused
  form is 1.26x slower because 8 threads × 4 MB exceeds a 16 MB L3.
* **Paged work item = (sequence, q head) end to end**, one scan of the head's
  cache, accumulator in registers.
* **Causal bound:** the key loop stops at the last row's limit and each row
  chunk clamps to its own limit, so a causal call never computes a masked block.
  Leaving that out cost 4.8x at rq=32.
* **Call path:** the dense path uses a 5-argument plan-buffer entry
  (`dense_make_plan` + `dense_forward`) and the paged path a per-geometry plan
  object in Python that patches only the data pointers.
* **Workspace** is per thread, per geometry (a mixed 64-bit key), never returned
  to the allocator: a fresh 33 MB `torch.empty` is served by mmap and faults in
  page by page.

AVX-512 policy layouts: fp32 `K [hd][tokens]`, `V [tokens][hd]`; bf16
`K [hd/2][tokens][2]` (the pair `dpbf16ps` contracts), `V [tokens][hd]`. Q@K^T is
`RQT=8` rows against 16 positions — one K load plus 8 broadcast-FMAs — and P@V is
an 8×32 register tile. The K pack is a 16×16 blocked transpose (four shuffle
stages, one 64-byte store per output row).

## Numbers — Ryzen 7 7840H, 8 threads, causal

`tests/bench_cpp.py --isa vec --rounds 21`, one process, arms rotated, **min**,
`ctrl` is a duplicate `cpp` arm and is the noise floor for this run. µs, and
`x/mojo` / `x/vllm` are ratios (`< 1` is faster). Ours includes the K/V pack and
the output allocation; vLLM's cache is staged and its output preallocated.

**Caveat: this run was taken while the box carried a load average of ~2 (the two
per-ISA agents were active). The duplicate-arm floor reads 1.25 on
`128:8:64 bf16` and 1.12 on `4096:8:64 fp32`, i.e. those two rows are not
resolvable in this artifact.** Re-run with `scripts/verify.sh`-style isolation
before quoting them.

| row | dtype | cpp | mojo | vllm | cpp/mojo | cpp/vllm | mojo/vllm | ctrl/mojo |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 32.9 | 26.4 | 30.7 | 1.246 | 1.073 | 0.861 | 1.251 |
| 128:8:64 | fp32 | 40.6 | 30.7 | 27.9 | 1.321 | 1.454 | 1.100 | 1.297 |
| 512:8:64 | bf16 | 266.8 | 266.2 | 379.5 | 1.002 | 0.703 | 0.701 | 1.012 |
| 512:8:64 | fp32 | 349.5 | 350.6 | 345.1 | 0.997 | 1.013 | 1.016 | 1.006 |
| 1024:8:64 | bf16 | 1041.5 | 1035.8 | 1451.6 | 1.005 | 0.717 | 0.714 | 1.004 |
| 1024:8:64 | fp32 | 1368.3 | 1349.4 | 1336.8 | 1.014 | 1.024 | 1.009 | 0.999 |
| 2048:8:64 | bf16 | 3995.6 | 4096.4 | 5697.5 | 0.975 | 0.701 | 0.719 | 0.981 |
| 2048:8:64 | fp32 | 5271.3 | 5305.6 | 5102.8 | 0.994 | 1.033 | 1.040 | 1.002 |
| 4096:8:64 | bf16 | 16090.8 | 16168.1 | 22758.5 | 0.995 | 0.707 | 0.710 | 1.001 |
| 4096:8:64 | fp32 | 23060.6 | 20599.6 | 20008.3 | 1.119 | 1.153 | 1.030 | 1.118 |
| 256:16:128 | bf16 | 285.1 | 353.9 | 396.4 | 0.806 | 0.719 | 0.893 | 0.797 |
| 256:16:128 | fp32 | 377.2 | 427.9 | 420.3 | 0.882 | 0.897 | 1.018 | 0.906 |
| 1024:16:128 | bf16 | 4152.1 | 5153.1 | 5658.4 | 0.806 | 0.734 | 0.911 | 0.803 |
| 1024:16:128 | fp32 | 6374.0 | 6228.8 | 5119.7 | 1.023 | 1.245 | 1.217 | 1.038 |
| 2048:16:128 | bf16 | 16131.5 | 19623.4 | 22440.5 | 0.822 | 0.719 | 0.874 | 0.829 |
| 2048:16:128 | fp32 | 23358.4 | 22623.4 | 19847.9 | 1.032 | 1.177 | 1.140 | 1.047 |
| 4096:16:128 | bf16 | 65780.1 | 91447.7 | 88830.7 | 0.719 | 0.741 | 1.029 | 0.720 |
| 4096:16:128 | fp32 | 93236.9 | 119597.3 | 80103.3 | 0.780 | 1.164 | 1.493 | 0.777 |
| 512:32:128 | bf16 | 2355.1 | 2904.3 | 3021.2 | 0.811 | 0.780 | 0.961 | 0.794 |
| 512:32:128 | fp32 | 3780.0 | 3746.3 | 2779.0 | 1.009 | 1.360 | 1.348 | 1.035 |
| 1024:8:32 | bf16 | 589.1 | 653.0 | 811.3 | 0.902 | 0.726 | 0.805 | 0.927 |
| 1024:8:32 | fp32 | 748.9 | 736.1 | 751.5 | 1.017 | 0.997 | 0.979 | 1.034 |

| paged (heads=16, dim=64, page 32) | dtype | cpp | mojo | vllm | cpp/mojo | cpp/vllm | mojo/vllm | ctrl/mojo |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| self b=1 ctx=256 | bf16 | 8.1 | 8.0 | 6.6 | 1.011 | 1.227 | 1.214 | 0.999 |
| cross b=1 ctx=256 | bf16 | 6.5 | 6.4 | 4.8 | 1.009 | 1.354 | 1.341 | 1.005 |
| self b=8 ctx=256 | bf16 | 34.7 | 58.7 | 51.6 | 0.592 | 0.673 | 1.138 | 0.512 |
| cross b=8 ctx=256 | bf16 | 32.0 | 54.3 | 44.5 | 0.588 | 0.719 | 1.222 | 0.512 |
| self b=8 ctx=1024 | bf16 | 465.8 | 626.3 | 608.2 | 0.744 | 0.766 | 1.030 | 0.748 |
| cross b=8 ctx=1024 | bf16 | 461.8 | 606.0 | 583.0 | 0.762 | 0.792 | 1.039 | 0.760 |
| self b=32 ctx=1024 | bf16 | 2371.7 | 2359.9 | 2574.2 | 1.005 | 0.921 | 0.917 | 1.002 |
| cross b=32 ctx=1024 | bf16 | 2309.7 | 2303.7 | 2446.0 | 1.003 | 0.944 | 0.942 | 1.003 |
| self b=1 ctx=4096 | bf16 | 120.4 | 261.9 | 219.6 | 0.460 | 0.548 | 1.192 | 0.579 |
| cross b=1 ctx=4096 | bf16 | 237.1 | 406.9 | 380.6 | 0.583 | 0.623 | 1.069 | 0.678 |
| self b=1 ctx=256 | fp32 | 11.6 | 11.8 | 9.7 | 0.977 | 1.188 | 1.216 | 0.965 |
| cross b=1 ctx=256 | fp32 | 9.4 | 10.1 | 6.9 | 0.924 | 1.348 | 1.459 | 0.932 |
| self b=8 ctx=256 | fp32 | 105.7 | 274.0 | 241.0 | 0.386 | 0.438 | 1.137 | 0.413 |
| cross b=8 ctx=256 | fp32 | 96.1 | 249.5 | 201.0 | 0.385 | 0.478 | 1.241 | 0.413 |
| self b=8 ctx=1024 | fp32 | 1197.6 | 1192.7 | 1170.8 | 1.004 | 1.023 | 1.019 | 0.994 |
| cross b=8 ctx=1024 | fp32 | 1155.8 | 1153.3 | 1134.2 | 1.002 | 1.019 | 1.017 | 0.994 |
| self b=32 ctx=1024 | fp32 | 4694.4 | 4694.5 | 4733.5 | 1.000 | 0.992 | 0.992 | 0.995 |
| cross b=32 ctx=1024 | fp32 | 4536.3 | 4534.7 | 4598.4 | 1.000 | 0.986 | 0.986 | 1.005 |
| self b=1 ctx=4096 | fp32 | 441.0 | 589.6 | 584.4 | 0.748 | 0.755 | 1.009 | 0.732 |
| cross b=1 ctx=4096 | fp32 | 504.9 | 655.0 | 637.4 | 0.771 | 0.792 | 1.028 | 0.756 |

### Where it stands

| | rows | beats mojo | beats vllm | worst cpp/mojo | worst cpp/vllm |
|---|---:|---:|---:|---:|---:|
| dense bf16 | 11 | 8 | 10 | 1.246 (not resolvable, ctrl 1.251) | 1.073 |
| dense fp32 | 11 | 4 | 2 | 1.321 (128:8:64, ctrl 1.297) | 1.454 |
| paged bf16 | 10 | 6 | 8 | 1.011 | 1.354 |
| paged fp32 | 10 | 7 | 6 | 1.004 | 1.348 |

* **Paged is the win.** b=8 is 1.7-2.6x faster than the Mojo backend
  (0.39-0.59) and 1.4-2.3x faster than vLLM; b=1/4096 is 1.7-2.2x. The rows that
  lose are b=1/ctx=256 (see below) and b=32/1024, which is parity and
  memory-roofline bound.
* **Dense bf16 is at parity or ahead of the Mojo backend and 1.35x ahead of
  vLLM**, except `128:8:64`, whose ratio moved 1.25 → 0.88 between two runs of
  the same binary (the duplicate arm moved with it), so it is not resolvable.
* **Dense fp32 is the weak column** and the honest gap: 1024/2048:16:128 are
  1.02-1.03 of Mojo (inside the noise floor) but 1.18-1.25 of vLLM;
  `512:32:128` is 1.01 of Mojo / 1.36 of vLLM. Everything else is parity.
* **`b=1` paged loses to vLLM by 1.19-1.35x** and that is a harness asymmetry,
  not a kernel gap: the audit preallocates vLLM's output tensor outside the
  timed region (`vllm.cpu_attention_with_kv_cache(..., out, ...)`) and
  `torch.empty` alone is 1.2 µs of a 6.5-8 µs call. Measured split at
  b=1/ctx=256 bf16: 2.8 µs extension+region, 5.9 µs through
  `cross_attention`. Against the Mojo backend (which allocates too) these rows
  are 0.92-1.01.

## Falsification ledger

Each of these was measured in-process with the duplicate arm in the same table;
the losing change was reverted.

| idea | result |
|---|---|
| run the row loop to a runtime bound (`for r < nrows` over `__m512 c[RQT]`) | the array lands in memory and every FMA becomes a load+store: **2.2-2.9x slower** |
| key the causal loop bound at `seq_k` instead of the last row's limit | up to 4.8x the useful work at rq=32; bounding it took 1.4-2.6x off every long row |
| fused per-head pack (one slab per thread, no barrier) everywhere | 1.26x faster below `0.6*LLC` per-thread, **1.26x slower** at 4096:16:128 fp32 where 8 heads' K/V is 32 MB against a 16 MB L3 |
| `rq` 16 / 32 / 64 / 128 at 4096:16:128 fp32 | 100.4 / 100.3 / 99.0 / 99.6 µs — flat; the accumulator size is not the lever here |
| `kt` 128 / 256 / 512 at 4096:16:128 fp32 | 101.5 / 99.0 / 98.8 µs — flat |
| software-pipeline the K load one dimension ahead in `qk` | **1.48x slower** fp32 D=128 (4.31 vs 2.90 in the phase bench); the pointer walk and the extra live register cost more than the latency it hides |
| `rq` as a fixed 32KB accumulator (128 rows at D=64) | no better than 16KB, and the extra rows lose L1 residency |
| 1/2/4 workers for the b=1 paged rows | 19.8 / 12.9 / 8.8 µs against 7.3 at 8 — the pool always wins, so the inline rule stays at the Mojo threshold |

## Instruments

* `tests/bench_cpp.py` — the three-arm in-process A/B (cpp / mojo / vllm) with a
  duplicate-`cpp` control, rotated order, min and median, JSON out.
* `tests/bench_vec_prims.cpp` — per-phase (qk / softmax / pv) timing against a
  pre-packed slab, in FMA-equivalents per cycle. That is how the `qk` load stall
  above was found: 1.45 cycles per FMA where 1.0 is the port limit.

## What is still open

1. **fp32 dense on a non-AMX x86 box.** Q@K^T runs at 1.45 cycles/FMA against a
   1.0 port limit; the load/FMA ratio is 1.125 and Zen 4's two load ports should
   absorb it, so the residue is scheduling inside the 8-row broadcast loop. On
   the AMX boxes this row is covered by `cpu_attn_amx.hpp` instead.
2. **`128:8:64`.** 33 µs, of which the pack and the call path are a large share;
   the ratio is not stable enough on this box to tune against.
3. **b=1 paged** is at the API floor (output allocation) unless the engine is
   willing to pass a preallocated output.
