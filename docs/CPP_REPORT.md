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

| dense row | dtype | cpp | mojo | vllm | cpp/mojo | cpp/vllm | mojo/vllm |
|---|---|---:|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 31.7 | 26.1 | 30.2 | 1.216 | 1.051 | 0.864 |
| 128:8:64 | fp32 | 39.2 | 30.4 | 27.7 | 1.290 | 1.416 | 1.098 |
| 512:8:64 | bf16 | 260.6 | 265.1 | 380.6 | 0.983 | 0.685 | 0.697 |
| 512:8:64 | fp32 | 345.0 | 348.8 | 344.6 | 0.989 | 1.001 | 1.012 |
| 1024:8:64 | bf16 | 1025.7 | 1039.5 | 1454.7 | 0.987 | 0.705 | 0.715 |
| 1024:8:64 | fp32 | 1278.9 | 1341.7 | 1340.1 | 0.953 | 0.954 | 1.001 |
| 2048:8:64 | bf16 | 3981.5 | 4117.9 | 5702.6 | 0.967 | 0.698 | 0.722 |
| 2048:8:64 | fp32 | 4956.3 | 5319.5 | 5141.8 | 0.932 | 0.964 | 1.035 |
| 4096:8:64 | bf16 | 15889.5 | 16308.0 | 22736.4 | 0.974 | 0.699 | 0.717 |
| 4096:8:64 | fp32 | 20078.2 | 20627.4 | 20462.5 | 0.973 | 0.981 | 1.008 |
| 256:16:128 | bf16 | 265.4 | 336.1 | 372.1 | 0.790 | 0.713 | 0.903 |
| 256:16:128 | fp32 | 375.9 | 420.4 | 412.9 | 0.894 | 0.910 | 1.018 |
| 1024:16:128 | bf16 | 4067.1 | 5066.2 | 5618.4 | 0.803 | 0.724 | 0.902 |
| 1024:16:128 | fp32 | 5813.5 | 6190.8 | 5104.2 | 0.939 | 1.139 | 1.213 |
| 2048:16:128 | bf16 | 15663.2 | 19744.4 | 21983.5 | 0.793 | 0.713 | 0.898 |
| 2048:16:128 | fp32 | 21405.3 | 22418.9 | 19789.7 | 0.955 | 1.082 | 1.133 |
| 4096:16:128 | bf16 | 61587.8 | 84537.5 | 87667.9 | 0.729 | 0.703 | 0.964 |
| 4096:16:128 | fp32 | 82742.3 | 117346.5 | 77822.2 | 0.705 | 1.063 | 1.508 |
| 512:32:128 | bf16 | 2412.5 | 2881.9 | 3013.0 | 0.837 | 0.801 | 0.956 |
| 512:32:128 | fp32 | 3544.4 | 3717.8 | 2794.9 | 0.953 | 1.268 | 1.330 |
| 1024:8:32 | bf16 | 592.8 | 651.6 | 801.2 | 0.910 | 0.740 | 0.813 |
| 1024:8:32 | fp32 | 709.5 | 733.3 | 721.2 | 0.968 | 0.984 | 1.017 |

| paged case | dtype | cpp | mojo | vllm | cpp/mojo | cpp/vllm | mojo/vllm |
|---|---|---:|---:|---:|---:|---:|---:|
| self b=1 ctx=256 | bf16 | 8.1 | 8.0 | 6.6 | 1.011 | 1.227 | 1.214 |
| cross b=1 ctx=256 | bf16 | 6.5 | 6.4 | 4.8 | 1.009 | 1.354 | 1.341 |
| self b=8 ctx=256 | bf16 | 34.7 | 58.7 | 51.6 | 0.592 | 0.673 | 1.138 |
| cross b=8 ctx=256 | bf16 | 32.0 | 54.3 | 44.5 | 0.588 | 0.719 | 1.222 |
| self b=8 ctx=1024 | bf16 | 465.8 | 626.3 | 608.2 | 0.744 | 0.766 | 1.030 |
| cross b=8 ctx=1024 | bf16 | 461.8 | 606.0 | 583.0 | 0.762 | 0.792 | 1.039 |
| self b=32 ctx=1024 | bf16 | 2371.7 | 2359.9 | 2574.2 | 1.005 | 0.921 | 0.917 |
| cross b=32 ctx=1024 | bf16 | 2309.7 | 2303.7 | 2446.0 | 1.003 | 0.944 | 0.942 |
| self b=1 ctx=4096 | bf16 | 120.4 | 261.9 | 219.6 | 0.460 | 0.548 | 1.192 |
| cross b=1 ctx=4096 | bf16 | 237.1 | 406.9 | 380.6 | 0.583 | 0.623 | 1.069 |
| self b=1 ctx=256 | fp32 | 11.6 | 11.8 | 9.7 | 0.977 | 1.188 | 1.216 |
| cross b=1 ctx=256 | fp32 | 9.4 | 10.1 | 6.9 | 0.924 | 1.348 | 1.459 |
| self b=8 ctx=256 | fp32 | 105.7 | 274.0 | 241.0 | 0.386 | 0.438 | 1.137 |
| cross b=8 ctx=256 | fp32 | 96.1 | 249.5 | 201.0 | 0.385 | 0.478 | 1.241 |
| self b=8 ctx=1024 | fp32 | 1197.6 | 1192.7 | 1170.8 | 1.004 | 1.023 | 1.019 |
| cross b=8 ctx=1024 | fp32 | 1155.8 | 1153.3 | 1134.2 | 1.002 | 1.019 | 1.017 |
| self b=32 ctx=1024 | fp32 | 4694.4 | 4694.5 | 4733.5 | 1.000 | 0.992 | 0.992 |
| cross b=32 ctx=1024 | fp32 | 4536.3 | 4534.7 | 4598.4 | 1.000 | 0.986 | 0.986 |
| self b=1 ctx=4096 | fp32 | 441.0 | 589.6 | 584.4 | 0.748 | 0.755 | 1.009 |
| cross b=1 ctx=4096 | fp32 | 504.9 | 655.0 | 637.4 | 0.771 | 0.792 | 1.028 |

### Where it stands

| | rows | beats mojo | beats vllm | worst cpp/mojo | worst cpp/vllm |
|---|---:|---:|---:|---:|---:|
| dense bf16 | 11 | 10 | 10 | 1.216 (`128:8:64`) | 1.051 |
| dense fp32 | 11 | 10 | 5 | 1.290 (`128:8:64`) | 1.416 |
| paged bf16 | 10 | 6 | 8 | 1.011 | 1.354 |
| paged fp32 | 10 | 7 | 6 | 1.004 | 1.348 |

* **The blocked K^T was the largest single win** (see the ledger): it took
  `4096:16:128 fp32` from 93.2 to 79.2 ms against vLLM's 80.1, `4096:8:64 fp32`
  from 23.1 to 20.1 ms against 20.0, and `4096:16:128 bf16` from 65.8 to 61.6 ms
  against 88.8. Dense fp32 went from 4/11 to 10/11 against the Mojo backend.
* **Paged is the strongest column**: b=8 is 1.7-2.6x the Mojo backend and
  1.4-2.3x vLLM, b=1/4096 is 1.7-2.2x both. The b=32/1024 rows are parity
  (memory-roofline bound), and b=1/256 loses to vLLM only (see below).
* **The two dense rows that lose to Mojo are the smallest** (`128:8:64`, 33-39 µs).
  Fixed cost — pack + region + `torch.empty` + the call path — is most of them,
  and the ratio moved 0.88 → 1.22 between runs of the same binary (the duplicate
  arm moved with it), so this box cannot resolve them. `CPU_ATTENTION_RQ` did
  not move them either.
* **Dense fp32 vs vLLM** is now 5/11 with the D=128 long rows at parity
  (`4096:16:128` 1.03, `4096:8:64` 1.00, `1024:8:64` 0.95). The rows vLLM still
  wins are `512:32:128` (1.27), `128:8:64` (1.40) and the mid `16:128` ones
  (1.06-1.13). On the AMX boxes `cpu_attn_amx.hpp` covers fp32 instead, and vLLM
  has no AMX fp32 kernel at all.
* **`b=1` paged loses to vLLM by 1.19-1.35x** and that is a harness asymmetry:
  the audit preallocates vLLM's output tensor outside the timed region
  (`vllm.cpu_attention_with_kv_cache(..., out, ...)`) and `torch.empty` alone is
  1.2 µs of a 6.5-8 µs call. Measured split at b=1/ctx=256 bf16: 2.8 µs
  extension+region, 5.9 µs through `cross_attention`. Against the Mojo backend
  (which allocates too) these rows are 0.92-1.01.

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
| pack K^T as `[hd][seq_k]` (the Mojo fp32 layout) | **1.17x slower** at 4096:16:128 fp32: `seq_k*4` bytes between dimensions is a TLB miss per load. Blocked every 64 keys (`[tile][hd][64]`) is what vLLM and the Mojo tree both do |
| `rq` as a fixed 32KB accumulator (128 rows at D=64) | no better than 16KB, and the extra rows lose L1 residency |
| 1/2/4 workers for the b=1 paged rows | 19.8 / 12.9 / 8.8 µs against 7.3 at 8 — the pool always wins, so the inline rule stays at the Mojo threshold |

## Instruments

* `tests/bench_cpp.py` — the three-arm in-process A/B (cpp / mojo / vllm) with a
  duplicate-`cpp` control, rotated order, min and median, JSON out.
* `tests/bench_vec_prims.cpp` — per-phase (qk / softmax / pv) timing against a
  pre-packed slab, in FMA-equivalents per cycle. That is how the `qk` load stall
  above was found: 1.45 cycles per FMA where 1.0 is the port limit.

## What is still open

1. **fp32 dense on a non-AMX x86 box** is 5/11 against vLLM. Q@K^T still runs at
   ~1.4 cycles/FMA against a 1.0 port limit; the load/FMA ratio is 1.125 and two
   load ports should absorb it, so the residue is scheduling inside the 8-row
   broadcast loop (pipelining it by hand measured 1.48x *slower*, see the
   ledger). On the AMX boxes this row is covered by `cpu_attn_amx.hpp`.
2. **`128:8:64`** (both dtypes) is the only dense row behind, at 33-39 µs where
   the pack, the OpenMP region and the output allocation are most of the call.
   Its ratio is not stable enough on this box to tune against.
3. **b=1 paged** is at the API floor (output allocation) unless the engine is
   willing to pass a preallocated output.
4. **The three-box matrix has not been completed**: the ARM (NEON) and Xeon
   (AMX) files are separate work items and this report only claims the Ryzen.

## ARM / NEON (Graviton 5, m9g.2xlarge, 8 cores)

`cpu_attn_neon.hpp`, measured by the ARM agent with the same protocol (one
process, four arms, rotated, min of 41 dense / 15 paged with a duplicate `cpp`
control reading 0.99-1.01).

**Paged is the best column in the project**: `cpp/vllm` **0.20-0.95, 10/10
wins**; `cpp/mojo` 0.56-1.09, 7/10 (the three losses are `b=1` rows at
1.01-1.09). Highlights: b=8/256 bf16 0.23 of vLLM and 0.51 of Mojo, b=8/256
fp32 0.36/0.54, b=1/4096 bf16 0.82/1.05. Parity: paged bf16 is within
**0.25-0.5 bf16 ulp** of the Mojo oracle, fp32 2.2-3.7e-7.

**Dense is `1.02-1.57x` of Mojo and `0.94-1.49x` of vLLM** — 4 of 22 rows beat
vLLM (fp32 D=128 at ≥1024 keys, 0.97-1.00) and the fp32 D=128 long row is at
parity with Mojo (1.02). The ablation puts qk+pv at 2.17 `bfmmla`/cycle against
a 2.0 measured issue ceiling, i.e. the matmuls are at the hardware rate and the
residue is the softmax (~28% of the call at D=64) plus, at S=128, 32% fixed cost
(pack + accumulator zero + epilogue + fork-join). The single largest step was
not a kernel change: rebasing onto the current shared driver (exact per-chunk
`nk` clamp, per-chunk causal skip, LLC-aware fuse/split) took the matrix from
`1.10-2.12x` to `1.06-1.58x` of Mojo.

Falsified on ARM: 4-row bf16 softmax (+0.4%, and it introduced a NaN via
`exp2(+inf)` on a never-visible row); 16 head dims per `pv` pass (1.16x slower,
spills); KT 512/1024 (0-6% worse). The blocked K^T that was worth 1.17x on
AVX-512 is only 1.03-1.04x on NEON — its K sweep already reads 16 bytes per
(dim group, key pair), so the page walk is a smaller share.

Open on ARM: fuse the exp into the `qk` loop (one pass with the running max as
the base, falling back to two passes on the first tile). That needs a contract
hook today's driver does not have — `qk` would take the row max, or `softmax`
would be allowed to be a no-op.

## Closing ledger — the ARM bf16 dense column remains open

Four rounds went into fusing the softmax's second pass into the P@V walk on
NEON, on the theory that it was the exposed ~1.7 cycles/row-key residue. That
theory is now **falsified with a number**, and the falsification is worth more
than the change would have been:

* At `kt=1024`, `rq=512` the softmax is already ~**1.2** cycles/row-key at D=64,
  not 1.7 — the prize was ~8% of the call, not the 15-20% the older 256-key-tile
  build suggested.
* The fused form is **3-5% slower on all 11 bf16 rows** (`cpp/vllm` 1.16-1.27
  against the two-pass 1.10-1.24). P@V's first d-chunk pass already holds 16
  accumulator registers plus 4 A + 4 B operands; the exp adds 8 live logit
  vectors and ~6 polynomial temps, so pass 0 spills. Phase total is conserved
  and the spill traffic costs more than the overlap returns — the same register
  wall that killed 16-dim `pv` passes (1.16x) and the 4-row bf16 softmax.
* **The four rounds of apparent kernel failures were a harness bug.** Once
  `t_neon` gained `batch` and strided q/k/v — the two dimensions it lacked, and
  the two present in every "failing" case — the fused kernel and a *verbatim*
  copy of the verified pre-fusion kernel agreed to the last digit on all nine
  shapes (`4.941e-01` vs `4.941e-01`, ...), and both disagreed with the
  harness's own reference. All 55 contiguous batch=1 shapes were correct in
  both. **A shape grid that omits the failing axis produces four rounds of
  confident misdiagnosis.**

Two process rules earned the hard way, both now demonstrated:

1. **A fallback restored by hand must be a verbatim copy of the verified file,
   not a flag.** Restoring code by hand twice reintroduced the bug and made the
   suite read "still failing" when the fast path was fine.
2. **Scratch sized by the logical row count while the vector loop runs over the
   register tile** has now produced three separate bugs in this tree: `acc_rows`
   on AVX-512, the padding-row `m`/`l` write on NEON, and the `l`/`alpha` pointer
   recovered from the accumulator address (correct only for a work item's first
   chunk). Size scratch by what the loop touches.

Where the remaining ARM bf16 headroom actually is, per the phase ablation:
`qk`+P@V run at 2.17 `bfmmla`/cycle against a measured 2.0 issue ceiling, so the
matmuls have ~no headroom, and the rest at `S<=512` is per-work-item overhead
(pack, accumulator zero, epilogue, fork-join; 2.4% of the S=128 call is the
driver's `memset` alone).
