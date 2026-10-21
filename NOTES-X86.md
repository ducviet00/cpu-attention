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

---

# Round 2 — fp32 on AMX (3-term bf16 hi/lo)

Branch `x86-amx-fp32` off `main` (f10f05e). fp32 has no AMX instruction and never
will, so each operand is split into two bf16 halves and every product keeps
three terms:

    x = hi + lo,  hi = bf16(x),  lo = bf16(x - hi)      (|lo| <= 2^-8 |x|)
    x*y ~= hi_x*hi_y + hi_x*lo_y + lo_x*hi_y             (drops lo*lo, 2^-16)

That is 3x the tile ops for fp32-accurate arithmetic (products exact, fp32
accumulate) against AMX's ~11x instruction advantage over AVX-512 fp32 FMA.

Implementation notes worth keeping:
- one entry point (`dense_amx_run`) with an `fp32` flag; bf16 and fp32 are the
  same kernel with two extra tile ops per k-block. `dense_amx_bf16` /
  `dense_amx_fp32` are thin wrappers, so the ctypes signature never changed.
- the packs emit an (hi, lo) slab **pair** per (key block, dim chunk), so packed
  K/V bytes equal the fp32 source bytes and the stride math is one constant
  (`kstride = fp32 ? 1024 : 512` bf16 per pair).
- Q needs no transpose (the A operand is `[row][dim]`), so splitting it is a
  straight conversion into a per-work-item hi/lo panel. That is 16 KB of traffic
  per work item at head_dim 128 -- visible, but 1/8 of the K traffic.
- the exp loop also writes P lo (`bf16(e - hi)`), which is what makes P@V
  accurate; without it P's 4e-3 relative error dominates at 1e-2.
- **one bug cost an hour**: the QK key-block base offset still used the bf16
  slab stride (512) instead of the pair stride, which reads the wrong K block.
  Everything passed parity except fp32, at ~1.0 max|diff| (a layout bug, not a
  precision bug). If fp32 parity ever shows O(1) errors, check strides before
  arithmetic.

Measured error (real kernel, 3 runs x 8 shapes x both causal modes, worst of
each run): **5.5e-5, 5.7e-5, 6.6e-5** against the 2e-4 gate; the single worst
observation over all runs is 7.6e-5 (512:32:128 causal). Grows with seq and
head_dim, flat in the seed, and non-causal is 5x better than causal (masked
logits at -3e38 push more of the weight into the exponent tail).

## Round 2 table (fp32, 2 threads, causal, medians of 5, interleaved)

| shape | before (o/sdpa, o/vllm) | after | o/sdpa | o/vllm (VEC) | GFLOP/s |
|---|---|---|---|---|---|
| 128:8:64 | 168us 0.68 / 1.69 | **121** | 0.49 | 1.22 | 138 |
| 512:8:64 | 1546us 0.59 / 1.51 | **768** | 0.30 | 0.75 | 350 |
| 1024:8:64 | 7473us 1.19 / 2.05 | **2236** | 0.39 | 0.62 | 480 |
| 2048:8:64 | 26700us 1.40 / 1.94 | **7673** | 0.38 | 0.56 | 560 |
| 4096:8:64 | 107451us 1.60 / 1.97 | **28691** | 0.41 | 0.52 | 599 |
| 256:16:128 | 1974us 0.96 / 1.79 | **979** | 0.46 | 0.88 | 274 |
| 1024:16:128 | 27104us 1.39 / 2.01 | **7701** | 0.39 | 0.57 | 558 |
| 2048:16:128 | 112937us 1.75 / 2.11 | **26860** | 0.41 | 0.50 | 640 |
| 4096:16:128 | 413541us 1.76 / 2.01 | **114024** | 0.47 | 0.55 | 603 |
| 512:32:128 | 15315us 0.95 / 2.05 | **5219** | 0.32 | 0.70 | 411 |
| 1024:8:32 | 2582us 0.71 / 1.19 | **1588** | 0.44 | 0.72 | 338 |

274-640 GFLOP/s against VLLM VEC fp32's 150-290 (its own best fp32). 10 of 11
rows beat vLLM's fp32 kernel; the exception is 128:8:64 at 1.22x, the same
short-S pack/marginal-cost row bf16 loses.

## The pack is at the memory ceiling on this box, not 5x off it

The integrator's estimate (13 GB/s, "should be 4-5 us") was measured against a
60+ GB/s Mojo pack on the Ryzen (8 cores). On this box with 2 threads:
- a warm 32 MB memcpy is **38.7 GB/s**; a warm 1 MB copy is 139 GB/s
  (`/tmp/bw.c`, 2 threads, second rep);
- our pack moves 16 MB of traffic in 475 us at 1024:16:128 = **34 GB/s**, i.e.
  88% of the 2-thread memcpy rate;
- at S=128 it is **10 us marginal** for 512 KB = 51 GB/s.

Two corrections to how it is measured, both now fixed in
`tests/bench_amx_phases.py`:
- the pack-only arm pays the fork-join + barrier with no attention work behind
  it, so it reads 2x high (20 us vs 10 us marginal at S=128). Report
  `full - kernel`;
- a round-robin task assignment was measured *slower* in that arm (20 us) than
  1 thread (14 us); I tried a blocked range instead and the marginal cost got
  worse (full 53 -> 53 with pack 24 -> and 3697 vs 3552 at 1024:16:128), so the
  round-robin is back. The apparent "no scaling" was the barrier, not the pack.

A 16x16 transpose rewrite (4 stages of `vpermt2d`, 64 permutes, 16 live
registers -- dense.mojo's formulation, replacing my unpack/permute2x128 version
with two stack arrays) changed nothing measurable: the pack is memory-bound, not
instruction-bound. It stays because it is smaller and has no stack traffic, and
`dense_amx_pack_selftest` now verifies both formulations agree.

---

# Round 3 — the AMX kernel in Mojo (the C shim is gone)

Branch `x86-amx-mojo` off `main` (23df21e). The campaign's finish line is one
language and one build artifact, so the kernel now lives in
`cpu_attention/_mojo/src/dense_amx.mojo` (`amx.mojo` underneath) and
`cpu_attention/_csrc/`, `scripts/build.py`'s gcc step, the ctypes plumbing and
`_csrc_kernels.so` are deleted.

**The port had never executed.** It compiled for `--target-cpu sapphirerapids`
and the emitted instructions were checked, but every parity run before this one
was served by the C shim: `main`'s Python gate sent fp32 to C, and for bf16
`_mojo_amx()` gated on `dense_has_amx()` — which was False — so the "c vs mojo"
A/B I ran earlier compared the C kernel with itself (1.00x everywhere, which
should have been the tell). Lesson: a dispatch experiment needs an assertion
that the *other* path is the one running.

## Three bugs, all only findable on hardware

1. **Element offsets used as byte addresses.** Mojo has no typed pointers, so
   every offset must be scaled by the element size; half the file added element
   offsets to a byte address. That is correct only at offset 0 (batch 0, first
   head, first key block), so the symptom was "NaN / O(1) errors for multi-head
   or multi-block, fine for the first one". `_ea(base, elem, esize)` now makes
   the convention explicit at every site and is the first place to look.
2. **The V pack wrote hi and lo to the same slab.** A shared `dst` plus
   `512 + r*32` in one store and `r*32` in another: the lo slab came out equal
   to the hi slab, i.e. a silent **2x in P@V** (V=1 in, 2.0 out). Separate
   `hi_p`/`lo_p` pointers. Found by dumping the packed workspace and comparing
   it against the source, not by reading the kernel.
3. **The QK key-block pitch was hardcoded 512 bf16** while fp32 packs an
   (hi, lo) pair per (key block, dim chunk), pitch 1024. Invisible for key
   block 0, wrong for every later one: it only showed at `seq_k > 32`, and only
   in fp32. Exactly the bug the C kernel had.

Also: `_C.dense_amx_init()` now exposes the `arch_prctl(ARCH_REQ_XCOMP_PERM)`
probe, so `_kernel.amx_available()` asks for tile-data permission itself rather
than relying on torch's oneDNN having done it.

## Performance: C vs Mojo, interleaved (`tests/ab_two_trees.py`)

Ratio is **C/Mojo**, so <1 means the Mojo kernel is slower. Same box, alternating
processes, medians of the per-round medians.

| shape | bf16 first run | bf16 final | fp32 final |
|---|---|---|---|
| 128:8:64 | 0.84 | 0.88 | 1.01 |
| 1024:8:64 | 0.83 | 0.97 | 0.94 |
| 4096:8:64 | 0.78 | 0.96 | 0.95 |
| 1024:16:128 | 0.84 | 0.97 | 0.96 |
| 4096:16:128 | 0.82 | 0.97 | 0.95 |

Two things closed almost all of the 16-22% gap:

* **`stack_allocation` aligns to the element size**, so the logits/prob/acc/qpad
  buffers were 4- or 2-byte aligned while every access to them is 64 bytes — each
  one split a cache line. `alignment=64` (the C version got it free from
  `posix_memalign`) is worth **~15%** and is now commented as load-bearing in the
  source. This is the single most portable Mojo lesson from the port.
* Nothing else did. Removing the `~{memory}` clobber from `tdpbf16ps`/`tilezero`
  (register-only instructions; `has_side_effect=True` still pins them in
  program order) was **neutral** — 0.86/0.82/0.77 vs 0.88/0.83/0.78 — so the
  "the clobber costs pipelining" theory that the gap suggested is falsified. It
  is kept anyway because it describes the instruction correctly. `perf stat` on
  1024:16:128 bf16 agrees that instruction count is not the残 residue: +4%
  instructions, +8% cycles.
* The remaining 3-5% on the long shapes and 12% at 128:8:64 is unexplained and
  small; 128:8:64 is a 60 µs call dominated by per-call fixed costs.

## Test lesson that cost the most time

The `amx.mojo` smoke test compared a 16x16x32 GEMM against a scalar reference
built from `a[i] = (i%7)*0.25`, `b[i] = (i%5)*-0.5` and reported 0.0 against a
**wrong** B-operand model, because that data is regular enough to fit two
different layouts. **Layout tests need irregular data.** With `(i*37)%101` the
same test separates them immediately, and it also caught that
`tdpbf16ps`'s AT&T operand order puts the destination tile *last*.

---

# Round 4 — the short-S rows: the per-call intercept, measured

Branch `x86-small-s`. Question: why do the short/medium bf16 rows lose to
vLLM-AMX, and is `128:8:64 fp32` (1.18 vs VEC) the same problem.

## The answer, in one table

`/tmp/intercept.py` style sweep, H=8 D=64 bf16 causal, 2 threads, medians:

| S | ours us | vLLM-AMX us | ratio |
|---:|---:|---:|---:|
| 8 | 19 | 4 | 4.37 |
| 16 | 22 | 7 | 3.11 |
| 32 | 18 | 11 | 1.70 |
| 64 | 29 | 26 | 1.14 |
| 128 | 68 | 59 | 1.15 |
| 256 | 200 | 179 | 1.12 |

Fit the two lines: **our slope is 0.38 us/key, vLLM's is 0.43.** We are *faster
per key than their AMX kernel*. The whole deficit is the intercept: **ours ~18 us,
theirs ~4 us**, and it is flat in S, so it is exactly what dominates the seven
losing rows and it is the same thing that loses `128:8:64 fp32`.

## What the intercept is made of (`tests/bench_amx_phases.py`)

| shape | total | pack | attention | fixed (region + marshalling) |
|---|---:|---:|---:|---:|
| 128:8:64 bf16 | 52 | 16 (31%) | 31 (60%) | 7.3 |
| 256:16:128 bf16 | 431 | 116 (27%) | 338 | 13 |
| 512:8:64 bf16 | 417 | 96 (23%) | 346 | 10 |
| 128:8:64 fp32 | 97 | 37 (38%) | 61 | 7.9 |

and inside the attention (`tests/bench_amx_ablate.py`, marginal cost of each
phase as a share of the attention part): **Q@K^T 33-40%, softmax 45-55%, P@V
0-10%** at these shapes — the softmax is the largest single piece, and P@V is
nearly free once the accumulator is L1-resident.

Independent confirmation that it is a *fixed* cost, not work: a heads sweep at
S=128 gives +4.8 us per 4 items, i.e. **~1.2 us per work item on top of its
matmul**, and our per-item overhead is about the same as vLLM's. Their advantage
is entirely that their timed region starts with K/V already staged.

## What I changed and what it was worth

* **16x16 transpose in place, two temporaries.** The first version built a new
  16-register array per stage, so 32 zmm were live on a 32-register machine.
  K pack 19 -> 14 us, fp32 pack 44 -> 37. Ported.
* **Profiling knobs** (`CPU_ATTENTION_AMX_MODE` bit table in the kernel header)
  and `tests/bench_amx_phases.py` / `tests/bench_amx_ablate.py` rewritten for
  them. These are what produced the tables above; keep them.

## Falsified this round

* **Hoisting the bounds check out of the pack's unrolled load loop.** The theory
  was that 16 conditional loads per tile put every load in its own basic block
  (IPC 0.16). Measured: neutral (pack 16-17 vs 14-16, inside noise). Reverted.
* **Cost-ordered work-item dispatch.** A causal item's cost is proportional to
  its key range, so plain round-robin gives one thread blocks 0+2 and the other
  1+3 (128 vs 192 units at S=128) and the call scaled only 1.08x on two cores.
  Ordering items by descending cost and round-robining over that should balance
  it exactly. Measured: **no change at S=128 and 3% worse at 1024:16:128 and
  4096:16:128**. Reverted. So the imbalance is not what limits the S=128 call.
* **More threads.** 3T and 4T are within 1% of 2T at every shape
  (2T/1T 1.83x, 4T/1T 1.83x at 1024:16:128), so this box has ~2 cores' worth of
  throughput and `_physical_cores()` capping at 2 is right. Worth recording:
  `/sys/.../thread_siblings_list` says cpu0 and cpu2 are siblings, but
  `taskset -c 0,2` is *faster* than `taskset -c 0-1` (59 vs 65 us at S=128), i.e.
  the VM's topology labels do not describe the host's real cores. Do not trust
  them for affinity decisions.

## The open item, with the numbers that bracket it

The pack moves 512 KB (read+write) in 16 us at S=128 = **32 GB/s**, against a
measured 139 GB/s for an L2-resident copy of that size. Its instruction budget is
~96 per 1 KB tile (16 loads of 64 B, 64 `vpermt2d`, 16 stores) = ~25 K
instructions per thread = **~2 us at 1.2 IPC**. Both estimates say 2-4 us; it
takes 16. There are **no zmm spills** in either worker (`objdump` on
`_gomp_worker_amx_bf16`: 0 `vmovdqu64 %zmm,(%rsp)`, 12 static `tileloadd`,
2352 instructions total), so it is not register pressure. Something is stalling
inside the pack loop that neither the instruction count nor the traffic explains,
and that is the one thing left that would close these rows: at 3x the pack rate
S=128 would be 52 -> 41 us against vLLM's 46-59, i.e. every remaining bf16 row
and the fp32 row would flip.

---

# Round 5 — the pack, isolated. It cannot go below ~10 us, and here is why

Branch `x86-small-s`, on `main` after the ARM merge (which routes the Mojo
AVX-512/ARM paths through a fused pack+attention region; the AMX kernel already
had one). Re-baselined first, as asked: parity green, and S=128 pack 14 us,
total 51 us — the round-4 numbers hold.

## The ladder (`cpu_attention/_mojo/tests/pack_bench.mojo`)

Same 16 loads of 64 B / transpose / 16 stores of 64 B per tile, 512 tiles per
rep, 8 reps, variants interleaved, single-threaded, medians of 9:

| variant | ns/tile | GB/s |
|---|---:|---:|
| full | **22** | 89 |
| no loads (transpose + stores only) | **16** | 124 |
| no transpose (loads + stores only) | **22** | 89 |

Read it twice, because it kills the obvious theory:

* **The transpose is free.** Removing it changes nothing (22 = 22): the network
  is entirely hidden behind the memory ops. The integrator's prior — 4 dependent
  stages x 16 shuffles = ~190 cycles per tile — does not survive; with 16
  independent outputs per stage the chain is 4 x 3 cycles, not 64 x 3.
* **The loads are the cost**, and even so, loads+stores alone run at **89 GB/s**
  for 2 KB of traffic per tile, which is the same class as the L2-resident copy
  ceiling (139 GB/s).

So the *tile* work is not the problem. The problem is that the same code inside
the kernel runs at ~109 ns/tile/thread (14 us for 256 tiles over 2 threads) —
5x the standalone rate — and **nothing I changed from inside the pack moved the
whole-call number**:

| change | A/B result (new/old, interleaved, `ab_two_trees.py`, 4 shapes x 2 dtypes) |
|---|---|
| in-place transpose, 2 temps (16+2 registers live instead of 32) | 0.99-1.03x, i.e. neutral |
| transpose split into 4 distinct stage functions (`_tr_s0..s3`, matching `dense.mojo`) so LLVM cannot reroll the stages into a runtime loop with per-stage index-vector loads | 1.00-1.02x bf16, 0.99-1.03x fp32 |
| pack task = 8 key blocks instead of 1 (decode ~16x instead of ~100x, removing ~2.6 us of `idiv` from a 14 us pass) | neutral |
| `PackJob` fields hoisted to locals (no aliasing info in Mojo, so the compiler may reload `k_stride_t` after every store) | neutral |
| bounds-check hoisting in the unrolled load loop (round 4) | neutral |

`objdump` agrees that it is not register pressure (0 `vmovdqu64 %zmm,(%rsp)` in
either worker) and not call overhead (the pack is inlined: the worker's only
`call`s are `omp_get_thread_num`, `omp_get_num_threads`, `GOMP_barrier`,
`memset`). It also shows the transpose *does* get rerolled into a loop with the
index vectors loaded from `.rodata` per iteration — which is why the split-stage
rewrite looked so promising — but fixing it changed nothing measurable, so the
reroll is not what costs either.

## Verdict, with the numbers

At S=128 (the worst row) the pack is 10-16 us of a 49-55 us call, and a
*free* pack would leave us at ~40 us against vLLM's 46-59 — i.e. the row is
reachable only by making the pack ~free, not by making it 2x faster. Everything
I can reach from inside the pack is neutral, the tile loop is already at memcpy
class standalone, and the residual (kernel 109 ns/tile vs standalone 22) is not
register pressure, call overhead, instruction count, the transpose network, the
task decode, or aliasing. I could not find it, and I would now stop rather than
grind: the honest statement is **the pack cannot go below ~10 us by any change I
have been able to test**, and the remaining bf16 rows plus `128:8:64 fp32` are
not reachable through this path.

The one structural option left is the one you raised and I rejected twice:
the pack is a pure function of K/V, and every work item's *first* super-tile
needs only keys [0, q0+32). Overlapping it with the attention needs a
producer/consumer handshake (per-key-block ready flags) or a three-phase split
(K pack | QK+softmax while packing V | P@V), which costs an extra barrier and a
global P buffer per work item (512 KB at S=128/H=8). Estimated ceiling: V is
half the pack, so at best it saves ~half of it — 5-8 us — and the three-phase
version also has to beat the extra barrier and P traffic. That is a redesign of
the driver, not a pack optimisation, and I would want your go-ahead before
building it.

`pack_bench.mojo` stays: it is the instrument that says where to look (loads,
not the network) and where not to (the transpose).

## Final table on the merged tree (`scripts/verify.sh ... r5`)

| | beats SDPA | beats VLLM-VEC | beats VLLM-AMX |
|---|---|---|---|
| fp32 (11 rows) | 11/11 | **11/11** | n/a (their AMX has no fp32) |
| bf16 (11 rows) | 11/11 | 11/11 | 6/11 |

**All 22 rows now beat vLLM-VEC**, including `128:8:64 fp32`, which was 1.18-1.21
across the last four verify runs and measured **0.98** here (o/sdpa 0.38 against
0.55 in the previous run — the merge with `main` moved it, and the ratio moved
more than the drift band, so it is not noise). The remaining bf16 losses against
vLLM's AMX build are 128:8:64 (1.49), 512:32:128 (1.13), 256:16:128 (1.14),
512:8:64 (1.07), 1024:16:128 (1.00) — every one of them the pack+fixed
intercept measured in round 4, not the kernel.

## Round 5b — the pack question is CLOSED

Four more probes after the ladder, all negative. Written down so nobody
re-derives them.

**1. It is not TLB and not page faults.** `perf stat` on 400 pack-only calls vs
400 no-pack calls (same script, same shape): `dTLB-load-misses` 2.27 M of 1.19 G
loads = **0.19%** (the standalone loop: 0.04%), `page-faults` identical (41 800,
all torch startup). The counters also give the pack's own cost directly: the
difference between the two runs is **41 M instructions and 17 M cycles over 400
calls = 102 K instructions and 42 K cycles per call**, i.e. 11 us — the same
number the phase bench reports. The ARM agent's 28 GB/s and our 32 GB/s are not
a TLB effect.

**2. The pack retires ~400 instructions per 1 KB tile; the same tile standalone
retires ~96.** 102 K instructions per call at S=128 is 256 tiles' worth, and the
tile is 16 loads + 64 shuffles + 16 stores. So the in-call pack executes ~4x the
instructions of the identical tile loop, at IPC 2.4 (so it is not stalled — it is
doing more work). That is the whole "5x slower than standalone" mystery, and it
is an *instruction* effect, not a memory one.

**3. Three source formulations compile to the same code, and none of them change
it.** (a) the stages in one function with `comptime for k`; (b) each stage a
separate `@inline(.always)` function (matching `dense.mojo`'s `_tr16_s0..s3`, so
LLVM cannot reroll across them); (c) sixteen *named* locals with all 64
`shuffle` calls written out and no array indexing at all, so no memory form is
expressible. All three: identical `perf` instruction count (4.564 / 4.569 /
4.574 G over the same 400 calls, within 0.2%) and identical wall time
(0.99-1.03x in `ab_two_trees`). Mojo/LLVM normalises them, and the 4x is what
this expression lowers to on this toolchain. Not fixable from the source side.

**4. The three-phase split is decisively out, measured not modelled.** Deferring
P@V forces every work item's P (probabilities) to survive until the V pack
finishes, so the per-thread P buffer grows from one item's worth to all of that
thread's items. Writing + reading that buffer, warm, single-threaded:

| shape | P per thread | traffic | cost | (V pack it would hide) |
|---|---:|---:|---:|---:|
| 128:8:64 | 96 KB | 192 KB | 25.8 us | ~7 us |
| 256:16:128 | 640 KB | 1280 KB | 426 us | ~52 us |
| 512:32:128 | 4.6 MB | 9.2 MB | 13.3 ms | ~200 us |

The buffer costs **2-8x the phase it would overlap**, at every target shape, and
it is 640 KB-4.6 MB per thread, i.e. out of L2. The "K pack | QK+softmax |
V pack + P@V" idea is dead on its own traffic, so I did not build it.

**Status: closed.** At S=128 the pack is 10-16 us of a 49-55 us call; a *free*
pack would leave us at ~40 us against vLLM's 46-59; the pack cannot be made
faster by any source change this toolchain responds to, and it cannot be hidden
behind the attention without paying more than it costs. `128:8:64` bf16 (1.49)
needs the *kernel* ~25% faster at 60 us, which is a different project and not
started.

## Round 6 — correction on `128:8:64 fp32`, and where the per-item cost really is

**The `0.98` on `128:8:64 fp32` was a contaminated run.** Re-run three times in a
clean environment (no `CPU_ATTENTION_*` set, fresh shell, after a clean rebuild):

```
ours 129 / 123 / 165 us    vllm 107 / 108 / 169    o/vllm 1.21 / 1.14 / 0.98
sdpa 241 / 254 / 450                                 o/sdpa 0.54 / 0.49 / 0.37
```

The third run is the one that produced 0.98, and look at what moved with it:
**vLLM's own time went 108 -> 169 us and SDPA's 254 -> 450**, i.e. that whole
process ran ~60-80% slow. Ours moved only 28%, so the *ratio* dropped below 1
without us getting faster. `bench_vs_refs.py` interleaves within a process, so a
uniform slowdown preserves ratios — this one did not, which means the
interference was bursty enough to hit some variants' samples harder than others.
**Conclusion: the row is 1.14-1.21, the tree has 21/22 against VEC, and I should
have flagged a single-row run whose SDPA number had moved 77%.** Single-shape
runs on this box have +-30% error; only the full matrix in one process is worth
quoting.

## The pack number is not an ablation artifact — but the prologue/epilogue is real

New profiling bits: **128 = skip the per-item Q prologue** (the zero-padded panel
and the Q tile loads), **256 = skip the output epilogue** (the normalise-and-
store). Medians of 11, all modes round-robin in one process, `128:8:64` bf16:

| | us |
|---|---:|
| full | 63.9 |
| Q prologue | **3.1** |
| output epilogue | **3.9** |
| both | 4.2 |
| pack, marginal (`full - noPack`) | **14.5** |
| pack, isolated (`packonly - nothing`) | **14.4** |
| nothing (region + marshalling) | 9.2 |

The two pack numbers agree to 1%, so the pack measurement is clean — and it has
to be, because bit 1 skips the *whole* item loop (`task = state.items`), so
pack-only never runs a prologue, an epilogue or a single tile instruction. The
hypothesis that the ablation was charging the pack for per-item work is
falsified.

What the ablation *did* find is a legitimate target the integrator predicted for
a different reason: **the Q prologue plus the output epilogue are 7 us of a
64 us call at S=128 (11%)**, i.e. ~0.22 us per work item, on top of the ~1.2 us
per item of everything else. Both are worth attacking:
* the Q prologue is 2 tile loads per (key chunk, dim chunk) and Q does not change
  across key chunks — 16 KB of redundant L1 re-reads per item at nk=128, for
  4 KB of Q;
* the epilogue is 12 instructions per 16 output elements and measures ~5x that,
  the same unexplained factor as the pack (see round 5).

Caveat for whoever uses bit 128: with the Q prologue skipped the kernel reads an
uninitialised panel, and **fp32 then segfaults** (garbage logits reach a path that
indexes past a buffer). bf16 survives. Use bit 128 for bf16 timings only.

## Round 6b — the clock is NOT downclocked, and my "4x instructions" claim is retracted

**Clock, measured with `perf stat -e cycles` over a 20-iteration run minus the
same script at 0 iterations (single-threaded, 2048:8:64):**

| workload | cycles/second |
|---|---:|
| full AMX attention (bf16) | **3.80 GHz** |
| AVX-512 Mojo attention | 3.67 GHz |
| scalar burn loop | 3.90 GHz |

**There is no AMX downclock on this box.** So the round-6 caveat — "half of every
5x gap I quoted may be frequency" — is **wrong and retracted**: the clock is the
same under AMX, AVX-512 and scalar code, and the gaps are real work.

**And the "~400 instructions per tile against ~96 standalone" figure does not
survive.** It came from a `perf stat -e instructions` difference between
pack-only and no-pack runs, and that difference is **0.9-3% of a 4.5 G-instruction
process**, i.e. inside the process-to-process startup variance of torch itself.
Trying to split it further (K-only vs V-only vs both, 400 calls each) gives
incoherent numbers — K alone +140 M, V alone +145 M, both +152 M — which is the
signature of noise, not of an accounting. **Retracted.** What is solid:

* the pack costs **10-16 us** at S=128 by two independent methods that agree to
  1% (marginal `full - noPack` = 14.5 us, isolated `packonly` = 14.4 us, and
  pack-only skips the whole item loop so it cannot be contaminated);
* the same tile work standalone runs at **22 ns/tile, 89 GB/s**, with the
  transpose free (round 5);
* the debut of the 5x: unexplained and now *not* frequency and *not* evidenced
  by instruction count. Anyone re-opening this needs the two-point perf method
  (100 vs 900 iterations, subtract) rather than the single-run difference.

**Epilogue is not a divide artefact.** `objdump`: the bf16 worker contains
exactly one `vdiv*` (one scalar reciprocal per row, as written) and the whole
`.so` has zero `vdivps`. So the 3.9 us epilogue is not a vector divide.

**The Q prologue cannot be hoisted with 8 tiles.** The A operand needs 4 tiles
for D=64 (2 row groups x 2 dim chunks), the B operand 2 and the C tiles 4: that
is 10. Hoisting Q out of the key-chunk loop would need the whole Q panel live in
tiles across chunks, which does not fit; the alternatives are 16-key chunks
(2 C tiles, worse softmax) or D=32 only. Recorded so it is not attempted again.
