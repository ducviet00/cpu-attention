# C++ backend — design, numbers, and the falsification ledger

`cpu_attention/csrc/` is a second backend beside the Mojo kernel with the same
Python API (`flash_attn_func`, `flash_attn_with_kvcache`, and the engine aliases
`self_attention` / `cross_attention` / `reshape_and_cache`). `CPU_ATTENTION_BACKEND`
selects one, both shared objects load in one process, and `tests/bench_cpp.py`
alternates them for an in-process A/B. `CPU_ATTENTION_BACKEND=auto` (the
default) picks `cpp` on x86-64 and stays on `mojo` on aarch64: see
"Known exceptions" §10.2 for the measured reason.

```bash
uv run --extra build python scripts/build_cpp.py           # -> cpu_attention/_C_cpp.so
CPU_ATTENTION_BACKEND=cpp uv run python tests/test_dense.py
CPU_ATTENTION_BACKEND=cpp uv run python tests/test_flash_api.py
CPU_ATTENTION_BACKEND=cpp uv run python tests/fuzz_dense.py --cases 80 --max-seq 900
CPU_ATTENTION_BACKEND=cpp uv run python tests/test_paged_sweep.py
```

## 1. Build

One artifact per architecture: baseline ISA plus runtime dispatch (CPUID /
`arch_prctl` for AMX on x86, HWCAP on aarch64), so a build made on one box runs
on another. No `-march=native`.

| TU | flags |
|---|---|
| everything else | `-O3 -fPIC -std=c++20 -fopenmp` |
| `vec.cpp` | + `-mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mavx512vnni -mfma -mavx2` |
| `amx.cpp` | the vec flags + `-mamx-tile -mamx-bf16 -mamx-int8` |
| `neon.cpp` | `-march=armv8.2-a+bf16+i8mm` |

AMX needs `arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)` before the first
tile instruction; `cpu_isa.cpp::amx_init()` does it once, from the plan build.
Dispatch is `CPU_ATTENTION_ISA=scalar|vec|amx|neon` for A/B and CPUID otherwise.

## 2. Design

`cpu_attn_impl.hpp` is the shared core: the job descriptions, the per-thread
workspace, and the loop skeletons. Each ISA file provides a *policy* — the
packed K/V layout, `qk`, `softmax`, `pv`, and the paged `store_head` /
`attend_head`.

* **Dense work item = (batch, q head, `rq` consecutive query rows).** The key
  range is walked in tiles of `kt`, the rows in register tiles of `RQT`. K/V are
  read once per `rq` rows instead of once per `RQT`, which is what the long
  D=128 rows are bound by. Two dispatch orders: *fused* (a thread packs the head
  it then attends, K/V staying in its own L2) and *split* (every head packed
  once, concurrent threads sharing one head's slab through L3). Fused wins below
  `workers * slab <= 0.6 * LLC`; the measured A/B at 1024:16:128 and
  512:32:128 fp32 is 1.08-1.15x for the fused order.
* **Paged work item = (sequence, q head) end to end**, one scan of the head's
  cache, accumulator in registers, and the store and the attend share one
  OpenMP region for MHA (a barrier separates them for GQA/MQA).
* **Causal bound:** the key loop stops at the last row's limit and each row
  chunk clamps to its own limit, so a causal call never computes a masked block
  (4.8x the useful work at rq=32 without it).
* **Call path:** the dense path uses a 5-argument plan-buffer entry
  (`dense_make_plan` + `dense_forward`); the paged path a 12-argument plan entry
  (`paged_make_plan` + `paged_forward`) that patches only pointers, two token
  strides and the scale. `self_attention` / `cross_attention` also take an
  `out=` buffer; at batch 1 `torch.empty` is 1.3 us of a 6.8 us call.
* **Workspace** is per thread, per geometry, never returned to the allocator.

Layouts (this backend defines them; `reshape_and_cache` writes them):

| path | K | V |
|---|---|---|
| vec dense fp32 | `[tile][hd][64]` | `[pad][hd]` |
| vec dense bf16 | `[tile][hd/2][64][2]` (dim pair, for `dpbf16ps`) | `[key/2][hd][2]` (key pair, for `dpbf16ps` P@V) |
| vec paged fp32 | `[hd][page]` | `[page][hd]` |
| vec paged bf16 | `[hd/2][page][2]` | `[page][hd]` |
| neon dense fp32 | `[hd][pad]` | `[pad][hd]` |
| neon dense bf16 | `[hd/4][pad/2][8]` | `[hd/2][pad/4][8]` (`bfmmla`) |
| neon paged | `[hd/2][page][2]` (fp32 `[hd][page]`) | `[page/2][hd][2]` (the Mojo ARM layout) |
| amx dense bf16 | `[dim pair][key]` | `[key pair][dim]` (both `tdpbf16ps` operands) |

The dense bf16 vec P@V is the one non-obvious choice: `dpbf16ps` contracts two
adjacent *keys* per half-lane, so both the V layout (key pairs) and the
probability buffer (bf16 key pairs, written by the softmax) are pair-formatted.
That is 2.1x the phase (see §6) and 1.33-1.45x the whole bf16 dense matrix.

## 2a. Where it stands, all three boxes

126 measured rows (42 per box: 22 dense + 20 paged). `best` is the faster of the
two baselines *on that row*, so the ratio is the one that matters.

| box | rows <= 0.97 of the best | 0.97-1.04 (noise floor) | > 1.04 |
|---|---:|---:|---:|
| Ryzen 7 7840H | 30 / 42 | 10 | 2 |
| Xeon 8581C (4C) | 29 / 42 | 10 | 3 |
| Graviton 5 | 19 / 42 | 5 | 18 |
| **all** | **78 / 126 (62%)** | 25 | 23 |

Against vLLM alone it wins 76 of 84 rows on the x86 boxes (77 of 84 against
Mojo), and on Graviton 37 of 42 against vLLM against 21 of 42 against Mojo --
i.e. even on the one box where it is behind, it beats the ISA-appropriate vLLM
build on all but five rows. The shortfall is concentrated in one place:
**aarch64 dense prefill**, where the Mojo kernel is the faster baseline and we
are 4-19% behind it (18 of the 23 rows over 1.04). §10.2 has the measured reason
and the two restructures that failed to close it. No other column is behind by
more than the noise floor.

## 3. Measurement protocol

* **One process, interleaved, rotated arms, minimum**, with a duplicate `cpp`
  arm as the control. `tests/bench_cpp.py` does this for the servers (41 dense /
  15 paged rounds); `tests/bench_row.py` (one row per process) plus
  `tests/bench_matrix.py` does it for the laptop, which throttles 20-50% over a
  ten-minute sweep. `tests/ab_env.py` alternates whole processes and takes the
  median of the per-process minima for env-knob A/Bs on the laptop.
* **Ratios are `ours / baseline`, below 1 is faster.** Baselines are the Mojo
  backend in the same process and vLLM's `csrc/cpu` at the pinned commit
  `db3b0f0`, built for the ISA that box actually uses: AMX for Xeon bf16, VEC
  for x86 fp32, NEON for Graviton.
* Ours includes the K/V pack and (unless `out=` is passed) the output
  allocation. vLLM's cache is staged and its output preallocated outside the
  timer, which is the convention the audit established; for the paged rows the
  cpp and Mojo arms are now given a preallocated `out` too, so those rows are
  like-for-like.
* The control arm's distance from the `cpp` arm is the noise floor for that row
  and is printed with every table below.

## 4. Ryzen 7 7840H (8 cores, AVX-512 + BF16, no AMX), 8 threads

One process per row, 21 rounds, min of 21, arms rotated. `c/m` = cpp/mojo,
`c/v` = cpp/vllm, `floor` = |ctrl/mojo − cpp/mojo|.

### 4.1 Dense (22 rows)

| row | dt | cpp µs | mojo | vllm | c/m | c/v | floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 22.2 | 29.9 | 32.3 | 0.744 | 0.689 | 0.055 |
| 256:16:128 | bf16 | 182.6 | 345.9 | 385.0 | 0.528 | 0.474 | 0.001 |
| 512:8:64 | bf16 | 173.4 | 263.7 | 352.9 | 0.658 | 0.491 | 0.005 |
| 512:32:128 | bf16 | 1444.6 | 2724.7 | 3000.3 | 0.530 | 0.481 | 0.018 |
| 1024:8:32 | bf16 | 415.6 | 624.5 | 801.6 | 0.665 | 0.518 | 0.001 |
| 1024:8:64 | bf16 | 703.5 | 1025.6 | 1443.8 | 0.686 | 0.487 | 0.013 |
| 1024:16:128 | bf16 | 2634.8 | 5137.3 | 5680.5 | 0.513 | 0.464 | 0.010 |
| 2048:8:64 | bf16 | 2873.8 | 4110.7 | 5734.7 | 0.699 | 0.501 | 0.007 |
| 2048:16:128 | bf16 | 10933.5 | 19735.4 | 22409.2 | 0.554 | 0.488 | 0.004 |
| 4096:8:64 | bf16 | 11726.4 | 16533.1 | 23264.2 | 0.709 | 0.504 | 0.005 |
| 4096:16:128 | bf16 | 44180.2 | 84634.3 | 89890.7 | 0.522 | 0.491 | 0.001 |
| 128:8:64 | fp32 | 27.0 | 31.6 | 28.7 | 0.852 | 0.938 | 0.002 |
| 256:16:128 | fp32 | 309.4 | 389.2 | 389.0 | 0.795 | 0.795 | 0.012 |
| 512:8:64 | fp32 | 301.8 | 329.1 | 319.9 | 0.917 | 0.944 | 0.049 |
| 512:32:128 | fp32 | 3137.6 | 3792.6 | 2872.8 | 0.827 | **1.092** | 0.026 |
| 1024:8:32 | fp32 | 670.7 | 745.5 | 718.2 | 0.900 | 0.934 | 0.000 |
| 1024:8:64 | fp32 | 1192.6 | 1317.3 | 1330.2 | 0.905 | 0.897 | 0.002 |
| 1024:16:128 | fp32 | 5370.5 | 6251.2 | 5353.3 | 0.859 | **1.003** | 0.016 |
| 2048:8:64 | fp32 | 5052.8 | 5526.2 | 5356.7 | 0.914 | 0.943 | 0.000 |
| 2048:16:128 | fp32 | 21867.0 | 22806.9 | 20447.6 | 0.959 | **1.069** | 0.004 |
| 4096:8:64 | fp32 | 20640.2 | 21025.1 | 21143.3 | 0.982 | 0.976 | 0.010 |
| 4096:16:128 | fp32 | 83320.7 | 138321.3 | 83407.2 | 0.602 | 0.999 | 0.002 |

**22/22 beat Mojo. 19/22 beat vLLM**; the three that do not are fp32 D=128 at
1.003-1.092 and are the documented exception in §10.1.

### 4.2 Paged (20 rows, 15 rounds)

| case | dt | cpp µs | mojo | vllm | c/m | c/v | floor |
|---|---:|---:|---:|---:|---:|---:|---:|
| self b=1 ctx=256 | bf16 | 5.1 | 6.9 | 6.0 | 0.738 | 0.857 | 0.003 |
| cross b=1 ctx=256 | bf16 | 4.2 | 6.4 | 4.4 | 0.658 | 0.961 | 0.003 |
| self b=8 ctx=256 | bf16 | 18.2 | 51.5 | 46.7 | 0.353 | 0.389 | 0.001 |
| cross b=8 ctx=256 | bf16 | 16.9 | 46.7 | 38.6 | 0.362 | 0.438 | 0.001 |
| self b=8 ctx=1024 | bf16 | 538.6 | 661.0 | 662.8 | 0.815 | 0.813 | 0.005 |
| cross b=8 ctx=1024 | bf16 | 433.4 | 627.7 | 608.6 | 0.690 | 0.712 | 0.031 |
| self b=32 ctx=1024 | bf16 | 2384.1 | 2365.3 | 2511.4 | 1.008 | 0.949 | 0.009 |
| cross b=32 ctx=1024 | bf16 | 2302.2 | 2335.1 | 2382.9 | 0.986 | 0.966 | 0.004 |
| self b=1 ctx=4096 | bf16 | 108.9 | 338.5 | 309.2 | 0.322 | 0.352 | 0.028 |
| cross b=1 ctx=4096 | bf16 | 103.9 | 270.1 | 244.6 | 0.385 | 0.425 | 0.042 |
| self b=1 ctx=256 | fp32 | 6.1 | 7.8 | 6.8 | 0.776 | 0.901 | 0.006 |
| cross b=1 ctx=256 | fp32 | 5.3 | 7.6 | 5.3 | 0.695 | 1.002 | 0.004 |
| self b=8 ctx=256 | fp32 | 110.7 | 263.1 | 249.4 | 0.421 | 0.444 | 0.008 |
| cross b=8 ctx=256 | fp32 | 104.5 | 241.1 | 228.1 | 0.433 | 0.458 | 0.020 |
| self b=8 ctx=1024 | fp32 | 1162.2 | 1187.3 | 1225.3 | 0.979 | 0.949 | 0.002 |
| cross b=8 ctx=1024 | fp32 | 1138.3 | 1162.0 | 1172.9 | 0.980 | 0.971 | 0.005 |
| self b=32 ctx=1024 | fp32 | 4754.1 | 4721.2 | 4699.6 | 1.007 | 1.012 | 0.002 |
| cross b=32 ctx=1024 | fp32 | 4586.4 | 4622.6 | 4557.5 | 0.992 | 1.006 | 0.003 |
| self b=1 ctx=4096 | fp32 | 387.8 | 596.2 | 577.7 | 0.651 | 0.671 | 0.011 |
| cross b=1 ctx=4096 | fp32 | 381.5 | 590.4 | 580.3 | 0.646 | 0.657 | 0.004 |

**19/20 beat vLLM** (worst 1.012) and **19/20 beat Mojo** (worst 1.008); both
exceptions are the `b=32/1024` rows, which are a 268 MB stream and sit on the
memory roofline — a plain read of that stream is 2.1 ms of the 2.3-4.8 ms call.

## 5. Graviton 5 (m9g.2xlarge, 8 cores, NEON + BF16), `--isa neon`

One process per row, 21 rounds, min, arms rotated.

### 5.1 Dense (22 rows)

| row | dt | cpp µs | mojo | vllm | c/m | c/v |
|---|---:|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 23.7 | 21.1 | 22.7 | 1.125 | 1.046 |
| 256:16:128 | bf16 | 207.6 | 188.6 | 212.7 | 1.101 | 0.976 |
| 512:8:64 | bf16 | 233.4 | 196.5 | 231.8 | 1.188 | 1.007 |
| 512:32:128 | bf16 | 1510.4 | 1446.3 | 1546.2 | 1.044 | 0.977 |
| 1024:8:32 | bf16 | 581.3 | 527.8 | 638.5 | 1.101 | 0.910 |
| 1024:8:64 | bf16 | 870.2 | 775.3 | 929.2 | 1.122 | 0.936 |
| 1024:16:128 | bf16 | 2841.4 | 2716.4 | 2968.4 | 1.046 | 0.957 |
| 2048:8:64 | bf16 | 3404.3 | 3138.7 | 3553.4 | 1.085 | 0.958 |
| 2048:16:128 | bf16 | 11225.4 | 10936.8 | 11496.4 | 1.026 | 0.976 |
| 4096:8:64 | bf16 | 13480.2 | 12565.4 | 14063.4 | 1.073 | 0.959 |
| 4096:16:128 | bf16 | 44199.4 | 43043.3 | 45466.5 | 1.027 | 0.972 |
| 128:8:64 | fp32 | 43.4 | 42.6 | 45.8 | 1.017 | 0.946 |
| 256:16:128 | fp32 | 579.7 | 521.7 | 659.2 | 1.111 | 0.879 |
| 512:8:64 | fp32 | 578.4 | 525.9 | 594.4 | 1.100 | 0.973 |
| 512:32:128 | fp32 | 4619.2 | 4049.5 | 4797.0 | 1.141 | 0.963 |
| 1024:8:32 | fp32 | 1304.7 | 1192.1 | 1320.6 | 1.094 | 0.988 |
| 1024:8:64 | fp32 | 2314.9 | 2072.8 | 2335.0 | 1.117 | 0.991 |
| 1024:16:128 | fp32 | 8656.4 | 7817.8 | 9476.4 | 1.107 | 0.913 |
| 2048:8:64 | fp32 | 9209.1 | 8201.0 | 9050.6 | 1.123 | 1.018 |
| 2048:16:128 | fp32 | 34106.5 | 31375.2 | 36814.2 | 1.087 | 0.926 |
| 4096:8:64 | fp32 | 36799.0 | 33162.4 | 34414.5 | 1.110 | 1.069 |
| 4096:16:128 | fp32 | 134897.3 | 139770.5 | 143634.3 | 0.965 | 0.939 |

**18/22 beat vLLM NEON** (0.879-1.069) and **1/22 beat the Mojo backend**
(0.965-1.188). This is the one column of the project that is behind a baseline,
and §10.2 has the measurement that explains it.

### 5.2 Paged (20 rows, 15 rounds)

| case | dt | cpp µs | mojo | vllm | c/m | c/v |
|---|---:|---:|---:|---:|---:|---:|
| self b=1 ctx=256 | bf16 | 6.6 | 8.6 | 9.0 | 0.764 | 0.733 |
| cross b=1 ctx=256 | bf16 | 5.4 | 7.5 | 5.2 | 0.720 | 1.039 |
| self b=8 ctx=256 | bf16 | 26.0 | 47.6 | 89.6 | 0.545 | 0.290 |
| cross b=8 ctx=256 | bf16 | 24.1 | 55.1 | 93.7 | 0.437 | 0.257 |
| self b=8 ctx=1024 | bf16 | 295.3 | 321.7 | 380.2 | 0.918 | 0.777 |
| cross b=8 ctx=1024 | bf16 | 299.7 | 320.7 | 334.6 | 0.935 | 0.896 |
| self b=32 ctx=1024 | bf16 | 1324.7 | 1371.2 | 1647.7 | 0.966 | 0.804 |
| cross b=32 ctx=1024 | bf16 | 1481.9 | 1491.4 | 1737.5 | 0.994 | 0.853 |
| self b=1 ctx=4096 | bf16 | 132.7 | 158.3 | 205.9 | 0.839 | 0.645 |
| cross b=1 ctx=4096 | bf16 | 132.1 | 158.0 | 205.1 | 0.836 | 0.644 |
| self b=1 ctx=256 | fp32 | 8.9 | 11.4 | 16.5 | 0.781 | 0.539 |
| cross b=1 ctx=256 | fp32 | 7.3 | 9.9 | 13.8 | 0.734 | 0.528 |
| self b=8 ctx=256 | fp32 | 84.2 | 125.9 | 183.3 | 0.669 | 0.459 |
| cross b=8 ctx=256 | fp32 | 82.3 | 123.8 | 173.7 | 0.665 | 0.474 |
| self b=8 ctx=1024 | fp32 | 532.5 | 576.9 | 653.4 | 0.923 | 0.815 |
| cross b=8 ctx=1024 | fp32 | 538.5 | 576.0 | 692.7 | 0.935 | 0.777 |
| self b=32 ctx=1024 | fp32 | 2198.4 | 2418.5 | 2991.3 | 0.909 | 0.735 |
| cross b=32 ctx=1024 | fp32 | 2164.5 | 2374.1 | 2962.2 | 0.912 | 0.731 |
| self b=1 ctx=4096 | fp32 | 277.1 | 304.9 | 358.0 | 0.909 | 0.774 |
| cross b=1 ctx=4096 | fp32 | 272.3 | 297.0 | 348.6 | 0.917 | 0.781 |

**20/20 beat Mojo and 19/20 beat vLLM**; the exception is `cross b=1/256` at
1.039, where both baselines are in the 5 µs region and the row is the
output-allocation convention (Mojo allocates, vLLM does not; ours does not when
`out=` is passed, which is how this table measures it).

## 6. Xeon Platinum 8581C (4 physical cores, AVX-512 + AMX), `--isa amx` / `vec`

4 torch threads, 41 dense / 15 paged rounds, one process, arms rotated, min.
`OMP_PROC_BIND=spread` is required on this box: it has 8 hardware threads on 4
cores, libgomp does not pin, and the same binary takes 26 or 65 µs at
`128:8:64` depending on whether two workers land on siblings — the ratio against
a *separate* run moves 1.0 -> 1.9 there. Every number below is one process.

The Xeon uses vLLM's AMX build for the bf16 rows and its VEC build for fp32
(there is no AMX fp32 kernel upstream), and our fp32 rows ride the AMX kernel
with a bf16 hi/lo split.

### 6.1 Dense (22 rows)

| row | dt | ISA | cpp µs | mojo | vllm | c/m | c/v |
|---|---|---|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | amx | 26.0 | 23.7 | 26.2 | 1.099 | 0.992 |
| 128:8:64 | fp32 | vec | 47.9 | 43.8 | 56.6 | 1.093 | 0.846 |
| 512:8:64 | bf16 | amx | 174.4 | 179.0 | 215.8 | 0.974 | 0.808 |
| 512:8:64 | fp32 | vec | 364.1 | 359.7 | 591.1 | 1.012 | 0.616 |
| 1024:8:64 | bf16 | amx | 569.0 | 611.2 | 708.5 | 0.931 | 0.803 |
| 1024:8:64 | fp32 | vec | 1102.7 | 1119.4 | 2244.8 | 0.985 | 0.491 |
| 2048:8:64 | bf16 | amx | 1953.2 | 2153.0 | 2967.9 | 0.907 | 0.658 |
| 2048:8:64 | fp32 | vec | 4007.3 | 4146.5 | 7595.4 | 0.966 | 0.528 |
| 4096:8:64 | bf16 | amx | 7309.7 | 8109.8 | 11195.2 | 0.901 | 0.653 |
| 4096:8:64 | fp32 | vec | 14869.4 | 15719.5 | 40664.2 | 0.946 | 0.366 |
| 256:16:128 | bf16 | amx | 174.6 | 192.3 | 198.9 | 0.908 | 0.878 |
| 256:16:128 | fp32 | vec | 364.5 | 375.4 | 637.2 | 0.971 | 0.572 |
| 1024:16:128 | bf16 | amx | 1629.9 | 1787.1 | 1925.4 | 0.912 | 0.846 |
| 1024:16:128 | fp32 | vec | 3863.0 | 3825.3 | 8065.8 | 1.010 | 0.479 |
| 2048:16:128 | bf16 | amx | 5590.3 | 6054.6 | 7403.7 | 0.923 | 0.755 |
| 2048:16:128 | fp32 | vec | 13680.1 | 14654.4 | 28300.8 | 0.934 | 0.483 |
| 4096:16:128 | bf16 | amx | 20842.0 | 23351.8 | 28384.7 | 0.893 | 0.734 |
| 4096:16:128 | fp32 | vec | 60425.2 | 71282.5 | 126565.4 | 0.848 | 0.477 |
| 512:32:128 | bf16 | amx | 1029.9 | 1131.6 | 1163.9 | 0.910 | 0.885 |
| 512:32:128 | fp32 | vec | 2236.6 | 2312.6 | 4185.4 | 0.967 | 0.534 |
| 1024:8:32 | bf16 | amx | 478.4 | 517.1 | 637.5 | 0.925 | 0.750 |
| 1024:8:32 | fp32 | vec | 819.8 | 802.3 | 1251.1 | 1.022 | 0.655 |

**22/22 beat vLLM** (0.366-0.992) and **17/22 beat Mojo** (0.848-1.099); the
five at 1.01-1.10 of Mojo are the short rows (`128:8:64` both dtypes) and three
fp32 rows at 1.01-1.02. `4096:16:128` fp32 is 0.848 of Mojo and 0.477 of vLLM.

### 6.2 Paged (20 rows)

| case | dt | ISA | cpp µs | mojo | vllm | c/m | c/v |
|---|---|---|---:|---:|---:|---:|---:|
| self b=1 ctx=256 | bf16 | amx | 8.8 | 10.7 | 11.9 | 0.824 | 0.742 |
| cross b=1 ctx=256 | bf16 | amx | 7.4 | 10.0 | 6.5 | 0.743 | 1.147 |
| self b=8 ctx=256 | bf16 | amx | 56.2 | 93.0 | 111.9 | 0.604 | 0.502 |
| cross b=8 ctx=256 | bf16 | amx | 53.0 | 92.0 | 99.2 | 0.576 | 0.534 |
| self b=8 ctx=1024 | bf16 | amx | 356.9 | 378.4 | 359.4 | 0.943 | 0.993 |
| cross b=8 ctx=1024 | bf16 | amx | 353.7 | 374.1 | 345.5 | 0.945 | 1.024 |
| self b=32 ctx=1024 | bf16 | amx | 1726.8 | 3126.5 | 2522.6 | 0.552 | 0.685 |
| cross b=32 ctx=1024 | bf16 | amx | 1653.8 | 3005.7 | 2467.2 | 0.550 | 0.670 |
| self b=1 ctx=4096 | bf16 | amx | 170.6 | 231.1 | 194.7 | 0.738 | 0.876 |
| cross b=1 ctx=4096 | bf16 | amx | 157.9 | 213.4 | 188.0 | 0.740 | 0.840 |
| self b=1 ctx=256 | fp32 | vec | 13.5 | 21.4 | 20.9 | 0.630 | 0.644 |
| cross b=1 ctx=256 | fp32 | vec | 11.0 | 19.6 | 12.1 | 0.560 | 0.906 |
| self b=8 ctx=256 | fp32 | vec | 167.5 | 170.5 | 179.8 | 0.982 | 0.932 |
| cross b=8 ctx=256 | fp32 | vec | 162.3 | 167.3 | 163.5 | 0.970 | 0.993 |
| self b=8 ctx=1024 | fp32 | vec | 674.1 | 730.2 | 785.7 | 0.923 | 0.858 |
| cross b=8 ctx=1024 | fp32 | vec | 659.8 | 716.4 | 787.5 | 0.921 | 0.838 |
| self b=32 ctx=1024 | fp32 | vec | 3573.2 | 4616.7 | 4832.2 | 0.774 | 0.739 |
| cross b=32 ctx=1024 | fp32 | vec | 3530.0 | 4575.9 | 4694.3 | 0.771 | 0.752 |
| self b=1 ctx=4096 | fp32 | vec | 308.1 | 329.4 | 327.0 | 0.935 | 0.942 |
| cross b=1 ctx=4096 | fp32 | vec | 305.8 | 326.6 | 319.8 | 0.936 | 0.956 |


**20/20 beat Mojo** (0.550-0.982) and **18/20 beat vLLM** (0.502-1.147); the two
that lose are `cross b=1/256` (1.147 — a 7 µs call whose duplicate-arm floor is
±6%) and `cross b=8/1024` bf16 at 1.024.

## 7. Falsification ledger

Every entry was measured in-process with the duplicate control arm, and the
losing change was reverted. The first two are the ones that cost real time and
are worth reading before touching an inner loop.

| idea | result | mechanism |
|---|---|---|
| two 16-key groups per Q@K^T step (halve the broadcast loads, 1.125 -> 0.625 loads/FMA) | **1.10-1.15x slower** on 8 of 8 rows | Zen 4's `vfmaddps zmm, zmm, m32bcst` is one macro-op whose 4-byte broadcast does not contend for the 512-bit load port; the second accumulator set only costs registers and code size. The load:FMA model that predicted a win is wrong on this core. |
| read V straight from the caller's tensor instead of copying it (halve the pack) | neutral below 1 MB per head, **1.34x slower** at 4096:16:128 | a per-key token stride of 8 KB is one TLB entry per key on every sweep; past ~1 MB the copy is cheaper than the stride, which is what `_llc_bytes()` gates. |
| `prefetchnta` the pack's source (it is read once and evicts the slab) | neutral (0.983/0.995) | the source stream is not what displaces the slab on Zen 4's L3. |
| `rq` 32 -> 64 at D=128 fp32 | 1.000-1.014 | the accumulator is not the traffic lever once K^T is blocked. |
| `kt` 256 -> 512 or 128 | 0.984-1.002 | per-tile fixed cost is already 1-2%. |
| split order (pack all heads, barrier, dynamic) vs fused | fused **1.08-1.15x faster** at 1024:16:128 and 512:32:128 fp32 | a thread keeps its head's slab in its own L2 and skips the barrier; the split order is only for slabs past 0.6 LLC. |
| two OpenMP regions for paged store + attend | shipped the fused form: **2.4 -> 1.1 µs** of fixed cost | a second fork-join on torch's pool is a sixth of a batch-1 call. |
| 23-argument paged entry, allocating its output | shipped the 12-argument plan entry + `out=`: **8.9 -> 5.3 µs** at `cross b=1/256` fp32 | the marshal and `torch.empty` were 40% of the call. |
| `rq` as a fixed 32 KB accumulator (128 rows at D=64) | no better than 16 KB | the extra rows lose L1 residency for the logits. |
| software-pipeline the K load one dimension ahead in `qk` | **1.48x slower** fp32 D=128 | the pointer walk and the extra live register cost more than the hidden latency. |
| pack K^T as `[hd][seq_k]` (no 64-key blocking) | **1.17x slower** at 4096:16:128 fp32 | `seq_k*4` bytes between dims is a TLB miss per load. |
| 1/2/4 workers for the b=1 paged rows | 19.8 / 12.9 / 8.8 µs against **7.3 at 8** | the pool always wins above 32 KV positions per call. |
| keep the *fused register panel* on aarch64 (logits in registers, no round trip) | **1.52x slower** | the logits fit in registers but the *output* accumulator does not, so panel-looping pays 4 accumulator load/stores per row-key against 0.125 for the tile loop. |
| a Cauchy-Schwarz fixed base for the row max on aarch64 (delete the max pass and the rescale, `qk` writes bf16 P itself) | implemented twice, **never passed parity**, reverted | the bound itself is now *verified exact*: `max|k|` accumulated in `pack`, `||q_r||` from the query row, and `m_r` match a Python reference to 5 digits for all 8 heads of a 128-key D=64 case, and `qk`'s per-row probability sums match too. The first version read `||q_r||` out of the packed Q group, where one 16-byte group holds *both* rows of a row pair, so the bound was 1.4x too large; fixing that left a second, unisolated defect in the `qk`-writes-P -> `pv`-reads-P handoff: for 3 of 8 heads the P@V numerator is wrong (row sums -7.6 against -14.2 exact) while `l` is right, identically with 1 worker and with both dispatch orders, so it is not a race. One row-mapping bug stands between this variant and being measurable; the recipe (pack kmax, the `m_r` prologue, the in-place exp epilogue) is in `docs/arm/RESULTS.md`. |
| fp32 via bf16 `bfmmla` hi/lo on aarch64 | **1.2%** (9177 -> 9064 µs at 2048:8:64) | `bfmmla` issues at 1.88-2.5/cycle against the scalar FMA's 2.25, so 3x the products at 2x the density nets out, and the doubled operand loads take the rest. |
| 4-row bf16 `pv` / 16-dim `pv` passes / one-row-pair at a time / L1-staged logits / softmax pipelining (aarch64) | 1.16x slower, 1.09-1.11x slower, 0-11% | all register-pressure variants of the same idea; see `docs/arm/RESULTS.md`. |
| degree-2 vs degree-5 exponential for bf16 paged | shipped degree 2 (inside the bf16 store), fp32 moved to **degree 6** | the degree-5 polynomial was 3.4e-6, which was the entire fp32 paged error budget (1.2e-6 at 17 keys). |
| denser fp32 via `dpbf16ps` on x86 | rejected on arithmetic | hi/lo needs 3 products; 32 MACs/instruction against the fp32 FMA's 16 is only 2x density for 3x the work. |

## 8. What the ISA files measure about themselves

* **x86 vec:** `qk` and `pv` run at 1.25 and 1.4 cycles per 64-MAC row-key, i.e.
  ~72-80% of the 1 zmm-FMA/cycle roof a double-pumped Zen 4 gives; at
  512:32:128 fp32 the measured 1.0 ms of `qk` is the roofline prediction to two
  digits. The in-situ split at 1024:16:128 fp32 is pack 20%, qk 47%, softmax 4%,
  P@V 36%; at 2048:16:128 bf16 it was pack 6%, qk 28%, softmax 6%, P@V 56%
  before the bf16 P@V change.
* **x86 AMX:** both matmul phases at 55-87% of the `tdpbf16ps` roof; the softmax
  is the largest single phase at D=64 (45% of a call), the pack is DRAM-bound
  (21 GB/s on one thread, 60-111 on four), and `kt=256` beat 128/512.
* **aarch64 NEON:** bf16 dense is at 18.6 instructions/row-key against Mojo's
  18.3 and 5.4 cycles against 4.7; the 8 `bfmmla`/row-key are 3.7 of those
  cycles against a 3.2-cycle issue floor. fp32 is at 14.2 cycles/row-key = 32
  FMAs at 2.25/cycle, and replacing a quarter of those instructions with a
  3x-denser instruction moved the total by 1.2%, which is the proof that it is
  the scalar-pipe roof rather than a scheduling artefact.

## 9. Numerics

Gates: dense fp32 <= 2e-4 against a naive fp32 reference, dense bf16 <= 3e-2,
paged bf16 0-2 bf16 ulp against the other backend, paged fp32 <= 1e-6.

| suite | result |
|---|---|
| `tests/test_dense.py` (65 checks, shapes/GQA/masks/plan reuse) | 65/65 |
| `tests/fuzz_dense.py --cases 80 --max-seq 900` | 80/80, worst 0.52x tolerance |
| `tests/test_flash_api.py` (flash-shaped API + rejections) | all |
| `tests/test_paged_sweep.py` (504 checks: hd 32/64/128 x page 16/32/48/64 x GQA 1/2/4 x 7 shapes x 2 dtypes, strided q/k/v, permuted block tables) | 504/504; worst fp32 1.6e-6, bf16 1.86 ulp |
| `tests/test_cpu_attention.py` (fast-encdec engine, 45 checks) | 45/45 |
| Mojo backend, unchanged | `test_dense` 65/65, `test_flash_api`, `test_paged_sweep` 504/504 |

Two real bugs were found by the new sweep and fixed in both backends:
`store_head` indexed the new K/V by *query* head (wrong slice, and off the end of
the tensor, for GQA/MQA), and the dense plan cache was not keyed on the backend,
so switching backends in one process kept calling the first one.

## 10. Known exceptions, with the measurement that explains each

### 10.1 Ryzen fp32 dense at D=128, long rows (1.003-1.092 vs vLLM)

The kernel is at the FMA roof and vLLM is not charged for the pack.

* `512:32:128` fp32, in-situ ablation: pack 22%, `qk` 32%, `pv` 30%, driver 1.6%.
  `qk` and `pv` each need 33.5 M vector FMAs, which at the 1 zmm-FMA/cycle roof
  over 8 threads is 1.0 ms each — measured 1.03 and 0.95. There is no headroom.
* The pack moves 16.8 MB (source read + slab write) in 0.71 ms, which is
  ~49 GB/s of combined traffic, i.e. the DRAM roof of this laptop.
* vLLM's arm is handed a *pre-staged* cache and a preallocated output; its
  kernel does the same 2.0 ms of FMA work and nothing else. Our 3.14 ms against
  their 2.87 ms is therefore "kernel at parity plus a pack they do not pay".
  Excluding the pack we are 0.93-0.95 of vLLM on these rows.
* Everything that could remove the pack was measured and lost: a native-layout
  `qk` needs either a transpose per query block or a horizontal reduction per
  logit (16-21 instructions per row-key against 17); V-native is neutral below
  1 MB and 1.34x worse above it; the blocked `[tile][hd][64]` K^T is already the
  best layout (1.17x over an unblocked one).

### 10.2 aarch64 dense prefill (1.02-1.19 of Mojo, 0.88-1.07 of vLLM)

The Mojo ARM dense kernel is the fastest dense kernel either tree has on that
box (0.80-0.93 of vLLM), and we are 2-19% behind *it* while being ahead of vLLM
on 18 of 22 rows. The measured state is instruction parity and a cycle gap:

| | ours | Mojo |
|---|---:|---:|
| bf16 instructions/row-key | 18.6 | 18.3 |
| bf16 cycles/row-key | 5.4 | 4.7 |
| bf16 `bfmmla` issue floor | 3.2 | 3.2 |
| fp32 cycles/row-key | 14.2 (= 32 FMA at 2.25/cycle) | 12.7 |

Both structural routes to those cycles were implemented and falsified: the fused
register panel is 1.52x slower because the *output* accumulator does not fit in
the register file alongside the logits, and the bounded-max formulation that
would delete the max pass and its rescale fails parity (41/65) on a defect that
is localised but not fixed: the bound itself is correct when instrumented
(`max|k|`, `‖q_r‖` and `m_r` all print sane, monotone values), every `Sq == 1`
case passes, and every multi-row case fails, which leaves the per-tile
probability sum that `softmax` flushes into `l` as the single suspect — one
indexing bug in one file, recorded with the instrumented recipe in
`docs/arm/RESULTS.md`. That is the one open path to these rows. The fp32 gap is 11% of *achieved* FMA throughput
(2.25 against 2.50 per cycle) at the scalar-pipe roof, and replacing a quarter of
the instructions with 3x-denser ones moves it 1.2%.

This is why `auto` resolves to `mojo` on aarch64. `CPU_ATTENTION_BACKEND=cpp`
there is still the right choice for the **paged** path, where it wins 20/20
against Mojo and 19/20 against vLLM.

### 10.3 paged `b=32/1024` (1.006-1.012 on Ryzen, parity elsewhere)

A 268 MB stream per call; a plain read of that stream is 2.1 ms of a 2.3-4.8 ms
call, and the row is memory-roofline for both backends. On Graviton the same row
is 0.73-0.85 of vLLM because that box has the bandwidth.

### 10.4 `b=1` paged rows

Use `out=`. The engine aliases accept a preallocated `[num_seqs, num_heads,
head_dim]` buffer, and at batch 1 `torch.empty` is 1.3 µs of a 6.8 µs call.
Without it those rows read 1.1-1.3 against a reference whose output is
preallocated outside the timer; with it they are 0.53-0.96.

## 11. Reproducing a table

```bash
# servers: one process, four arms, rotated, 41 dense / 15 paged rounds
python tests/bench_cpp.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so \
    --isa amx --rounds 41 --json /tmp/m.json          # bf16 on Xeon
python tests/bench_cpp.py --vllm ... --isa vec --rounds 41 --only dense
# laptop: one process per row (the box throttles 20-50% over a full sweep)
python tests/bench_matrix.py --vllm ... --isa vec --rounds 21 --json /tmp/m.json
# env-knob A/B on the laptop: alternating processes, median of per-process mins
python tests/ab_env.py --knob CPU_ATTENTION_KT --a 256 --b 512 --rows 1024:16:128:fp32
```

Boxes: Ryzen 7 7840H (8 cores, no AMX), Graviton 5 `m9g.2xlarge` (8 cores,
NEON+BF16), Xeon `m8i.2xlarge` (4 cores / 8 threads, AVX-512 + AMX). vLLM is
pinned at `db3b0f0`; `scripts/box_setup_cpp.sh` provisions a fresh box with the
pinned build, both backends and the reference.
