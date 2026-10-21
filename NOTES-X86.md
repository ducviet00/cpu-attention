# NOTES-X86 — the AMX campaign (x86 kernel agent)

Ratios are `ours/baseline` (below 1 = we win). Box: `ubuntu@3.27.45.233`,
m8i.xlarge = **4 vCPU / 2 physical cores**, Xeon 6975P-C (Granite Rapids),
`avx512_bf16` + `amx_bf16` + `amx_tile` + `amx_int8` (no `amx_fp16` flag),
480 MiB L3, torch 2.14.1+cpu, mojo 1.1.0, gcc 15.2.
Unless stated: `tests/bench_vs_refs.py --rounds 5 --threads 2`, causal, medians
of 5, all variants interleaved in one process.

## 0. Session baseline (reproduced, `--isa vec`, before any of my changes)

```
shape              bf16 o/sdpa o/vllm     fp32 o/sdpa o/vllm
S=128  H=8  D=64        1.66   1.72         0.68   1.69
S=512  H=8  D=64        1.81   1.38         0.59   1.51
S=1024 H=8  D=64        2.58   1.32         1.19   2.05
S=2048 H=8  D=64        3.14   1.43         1.40   1.94
S=4096 H=8  D=64        5.11   1.88         1.60   1.97
S=256  H=16 D=128       2.78   1.68         0.96   1.79
S=1024 H=16 D=128       4.88   1.77         1.39   2.01
S=2048 H=16 D=128       6.25   1.81         1.75   2.11
S=4096 H=16 D=128       6.77   1.84         1.76   2.01
S=512  H=32 D=128       3.25   1.69         0.95   2.05
S=1024 H=8  D=32        1.67   1.30         0.71   1.19
```
Matches the integrator's table within drift. Parity green (49/49 + flash API).

## 1. The decisive measurement: **vLLM's AMX build**

I built `vllm_cpu_attn_bench_amx.so` (`-DCPU_CAPABILITY_AMXBF16 -mamx-tile
-mamx-bf16 -mamx-int8`, same sources, same recipe as `scripts/setup_box.sh`) and
ran the same table with `--isa amx`. **Their AMX path is bf16-only**
(`TileGemm224` is specialised for `q=BFloat16`; fp32 throws "Unsupported kv cache
type"), so this is the bf16 ceiling reference:

```
shape                 ours    vllm-amx   vllm-vec   sdpa    ours/amx
S=128  H=8  D=64       181         46         97     100      3.95
S=512  H=8  D=64      1567        397       1131     867      3.95
S=1024 H=8  D=64      5508       1291       4169    2122      4.27
S=2048 H=8  D=64     22686       5061      16054    6955      4.48
S=4096 H=8  D=64    115207      20443      63092   24727      5.64
S=256  H=16 D=128     1963        356       1158     697      5.51
S=1024 H=16 D=128   27629       3241      15045    6235      8.52
S=2048 H=16 D=128  107173      12175      59788   22117      8.80
S=4096 H=16 D=128  434804      45372     237194   71855      9.58
S=512  H=32 D=128   13984       2002       8118    3991      6.99
S=1024 H=8  D=32     3203       1181       2475    1913      2.71
```

**AMX is worth 4-9.6x over the old AVX-512 dot kernel and 2.2x over SDPA.**
Their AMX kernel runs at **1.33 TFLOP/s** (2 threads) at S=1024:16:128 — 0.66
TFLOP/s/core against oneDNN's 1.45 TFLOP/s/core bf16 GEMM ceiling measured in
the same process (`scripts/prof/x86/amx_probe.py`: bf16 matmul 1119-1497 GFLOP/s
1 thread, fp32 218, SDPA bf16 attention 350-464).

**The target is their AMX number, not their VEC number.** The integrator has
since made this the campaign bar (`--isa amx` on x86, `neon` on Graviton), since
that is what a user gets; fp32 is the exception because their AMX path does not
exist for fp32, so VEC is their honest fp32 ceiling.

## 2. Where the time goes once you have AMX

Per (query row, key) "logit", attention needs `2*D` MACs. Measured ceilings:
- AVX-512 fp32 FMA: **249.6 GFLOP/s/core** (`fma fp32`, amxpeak) = 64 flops/cycle.
- AVX-512 `vdpbf16ps`: **119 GFLOP/s/core** = 32 flops/cycle — *no better than
  fp32 FMA* on this part. This is why our bf16 was never faster than our fp32:
  it costs 16 cycles/logit at D=128, and the measured 25.6 was that plus softmax.
- AMX bf16 (oneDNN GEMM): **1.45 TFLOP/s/core** = 372 flops/cycle, 11.7x the
  `dpbf16ps` rate.
- vLLM's *attention* at 1.33 TFLOP/s = 341 flops/cycle → 3.0 cycles/logit at
  D=128 (2 cores, 8.4 M logits, 3241 us). Matmul alone at 372 flops/cycle is
  **1.38 cycles/logit**, so **their softmax+bookkeeping is ~1.6 cycles/logit**.
  At D=64 the matmul term halves (0.69) and they run at ~2.4 c/logit.

So an AMX kernel is **softmax-bound (~1.5-2 c/logit)**, and the design has to be
organised around the softmax and around keeping the tiles fed.

Measured phase split of *my* kernel (`tests/bench_amx_ablate.py`, bit-ablation,
one process), share of the attention part:

| shape | Q@K^T | softmax | P@V |
|---|---:|---:|---:|
| 128:8:64 | 25% | 51% | 29% |
| 1024:8:64 | 26% | 58% | 20% |
| 4096:8:64 | 24% | 63% | 20% |
| 1024:16:128 | 33% | 39% | 28% |
| 4096:16:128 | 34% | 43% | 32% |
| 1024:8:32 | 15% | 72% | 13% |

(The `noqk` arm is confounded — garbage logits make the max move every tile and
force a 16 KB accumulator rescale — so the QK column is a lower bound. Read
QK+PV as "the matmul", which is what the D=128 split shows: 62% matmul, 39%
softmax, and the matmul is within 20-25% of the oneDNN-derived floor.)

## 3. AMX mechanics established on this box (build traps)

- `arch_prctl(ARCH_REQ_XCOMP_PERM /*0x1023*/, XFEATURE_XTILEDATA /*18*/)` must
  be called before any tile instruction, or you get SIGILL. This is also the
  fallback gate: if it fails, the .so reports "no AMX" and the Mojo kernel runs.
- gcc 15's AMX intrinsics (`_tile_loadd`, `_tile_dpbf16ps`, ...) take **literal
  tile numbers** (an `"i"` constraint), not a `__tile` type.
- `_tilecfg` layout that matters: palette at byte 0, **`colsb[8]` are `uint16_t`
  at offset 16**, `rows[8]` are `uint8_t` at offset 48. Writing colsb as bytes
  gives colsb=0x4040 and `ldtilecfg` faults.
- **`tdpbf16ps`'s B operand is 16x16 *32-bit* units**: element (tile row r, index
  t) holds `B[2r + (t&1)][t>>1]`, i.e. the contraction pair is interleaved at
  2-byte granularity inside a 4-byte unit and the N index is the unit. Recovered
  by single-bit probes (`dense_amx_debug_qk_raw` + a 512-position sweep: every
  position gave a clean (k,n)). A 16-bit-granularity transpose is wrong; vLLM's
  `reshape_and_cache` says the same thing ("head elements packed as quadwords").
  This cost me two hours and is now pinned by `dense_amx_pack_selftest`.
- The A operand *is* natural (element t = contraction t), confirmed by the QK
  matching `q @ k^T` to 4e-6.
- `_mm512_alignr_epi32` needs a literal immediate, so a computed rotate has to
  be `permutexvar_epi32` with an index vector (gcc refuses the runtime offset
  at -O1 and silently accepts it at -O3 — do not rely on that).
- The CPU does **not** advertise `amx_fp16` (GNR part, flag masked in this VM),
  so `tdpfp16ps` is out. Only `amx_bf16` + `amx_int8`.
- ASan works on this C kernel (`-fsanitize=address` + a standalone driver,
  `scripts/prof/x86/amx_drv*.c`) and found the two real layout bugs faster than
  guessing did. The analytic drivers (probe with V=one-hot / V[j]=j and compare
  against closed-form expectations) are the cheapest correctness test there is.
- ctypes releases the GIL around the call, so the shared pack scratch needs a
  mutex; verified with 4 threads x 8 calls → bit-identical to single-threaded.

## 4. The kernel (`cpu_attention/_csrc/dense_amx.c`)

One work item = 32 query rows of one (batch, head) swept over the key range.
32 rows because `tdpbf16ps` gives a 16x16 fp32 tile and 4 accumulators (2 row
groups x 2 key groups) is what fits beside the operand tiles in the 8-tile file.

```
tiles 0,1  A   (QK: Q rows 0-15, 16-31; PV: P rows 0-15, 16-31)
tiles 2,3  B   (QK: keys 0-15, 16-31; PV: dims 0-15, 16-31)
tiles 4-7  C   (row group x {B0,B1})
```

Two-level key blocking: Q@K^T in 32-key chunks into an fp32 logits buffer, then
one online-softmax pass over the whole `kt`-key super-tile, then P@V over the
super-tile. That amortises the output-accumulator load/store over `kt` keys
instead of 32 (`8*head_dim/kt` bytes per logit).

Layouts, and who builds them:
- **Q: nobody packs it.** The A operand is 16 rows x 32 head dims, which is
  exactly `[row][dim]` — `_tile_loadd(stride = q_stride_t*2)` straight from the
  caller's tensor, so any q/k/v strides work and a partial last row block is a
  zero-padded stack copy.
- **K: packed** by `pack_k_block` = 16x16 **32-bit** transpose, one 1 KB tile per
  (16 keys, 32 dims), zero-padded past `seq_k`.
- **V: packed** by `pack_v_block` = per dim `u`, the pair (V[2r][u], V[2r+1][u]) as
  a 32-bit unit (two 32-byte loads + four shuffles per tile row).
- Both packs run in the *same* `#pragma omp parallel` region as the attention
  (one fork-join), one task per (batch, kv head, key block) covering all dim
  chunks, so the source read is linear.

Softmax: per-row max over the super-tile, FA4's `-8` rescale threshold (the
accumulator is rescaled almost never), degree-4 `exp2` polynomial with an
integer exponent insert (masked lanes become exactly 0), probabilities rounded to
bf16 once, fp32 accumulate in P@V, output divided by the row sum.

## 5. What I tried and what it cost (do not redo)

| change | result |
|---|---|
| softmax over only the lanes QK wrote (`nkp`) + 256-key super-tiles | **1.05-1.10x** — ported |
| pack task = key block x all dim chunks (linear reads) | 1.2-1.5x on the pack, S=128 25→17 us — ported |
| zero the PV accumulator tiles at t0=0 instead of `memset` | 1.02-1.10x (S=256: the 16 KB memset was ~20%) — ported |
| two rows in flight in the exp loop | **-10..-40%** (four exp chains + four sums spill) |
| `vscalefps` instead of the integer exponent insert | neutral to slightly worse |
| kt = 384 / 512 keys per super-tile | 0.93-1.03x vs 256 (working set leaves 48 KB L1) |
| `_mm512_alignr_epi32` with a computed offset | won't compile below -O3 |

## 6. Final bf16 table (this kernel, 2 threads, causal, medians of 5)

See the commit message of the "bf16 AMX" commit and `tests/bench_vs_refs.py`.
Latest run vs vLLM's AMX build: o/sdpa 0.45-0.73, o/vllm 0.78-1.37; vs their VEC
build o/vllm 0.19-0.63. 875-1541 GFLOP/s; oneDNN's bf16 GEMM ceiling on this box
is 1450, so the long shapes are at ~1.05x of a pure GEMM.
Still behind their AMX at: S=128 (1.37), 256:16:128 (1.25), 1024:16:128 (1.13),
512:32:128 (1.16) — all short-S, where ~20-35% of our call is the K/V pack that
their timed region does not include (`bench_amx_phases.py`).

## 7. x86 qt sweep for fp32 (the integrator's fp32 dispatch, Xeon authority)

`tests/bench_qt.py --dtype fp32 --qts 4,6,8,10`, interleaved, medians of 5,
2 threads (qt=10/12 arms added to `dense.mojo`; additive, no ARM path selects
them):

| shape | qt=4 | qt=6 | qt=8 | qt=10 | best |
|---|---:|---:|---:|---:|---|
| 512:8:64 | 1167 | **1095** | 1278 | 1341 | 6 |
| 1024:8:64 | 5410 | **4579** | 5447 | 5461 | 6 |
| 2048:8:64 | 32202 | **24170** | 26927 | 26158 | 6 |
| 4096:8:64 | 154894 | **111017** | 117523 | 112779 | 6 |
| 1024:16:128 | 37074 | 29196 | 26342 | **25013** | 10 |
| 2048:16:128 | 150372 | 116515 | 104594 | **96845** | 10 |
| 4096:16:128 | 634259 | 488886 | 435250 | **401802** | 10 |
| 512:32:128 | 13690 | 12310 | **11440** | 11536 | 8 (~10) |

So on the Xeon: **fp32 head_dim 64 → qt=6** (1.19-1.40x over 4, and 8 is 5-18%
worse than 6), **head_dim 128 → qt=10** (1.20-1.48x over qt=8, 1.48-1.58x over
qt=4), with qt=8 the pick at 512:32:128. Every arm parity-checked (d ≤ 2.5e-6).
The old `_qt_for` returned 8 for x86 fp32 at S≥1024, which is 1.10x off at
head_dim 64 and 1.2-1.5x off at head_dim 128.
