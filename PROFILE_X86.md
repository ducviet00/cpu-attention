# PROFILE_X86 — where the x86 dense kernel's cycles actually go

> **Historical profile (pre-AMX dense kernel, third campaign).** The counters
> and instruction counts below are for the vector kernel; the current AMX
> numbers are in `docs/AUDIT.md` and `docs/STATUS.md`.


Box: `m8i.2xlarge`, Xeon 6975P-C, 8 vCPU = **4 physical cores**, AVX-512 + `avx512_bf16`
+ `amx_bf16`, 2 MB L2/core, ~3.8 GHz. Ubuntu 26.04, torch 2.14.1+cpu, mojo 1.1.0.
Protocol: **one thread (`taskset -c 3`, `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`),
medians of 9, all variants interleaved in one process.** No kernel source was
modified; everything here is measurement.

Read this as: *numbers → mechanism → falsification test → confidence*. Where my
data contradicts a claim someone else made, the contradiction is stated with the
counts that produce it.

---

## 0. TL;DR

1. **Our instruction count is not the problem at D=64; our IPC is.** At
   S=2048/H=8/D=64 fp32 we retire 320 M instructions per call and vLLM retires
   313 M — a dead heat — but we take 260 M cycles and they take 109 M
   (IPC 1.23 vs 2.87). Same story at D=128: 398 M vs 296 M instructions but 323 M
   vs 101 M cycles.
2. **What costs is data supply, and it is measurable:** 42 % of our loads fill
   from L2-or-further (vLLM: 21 %), **6.0 % of our loads miss L2 (vLLM: 0.07 %)**,
   and the backend is bound on loads for **48 % of all cycles (vLLM: 15 %)**.
   Neither spills, the `probs` round trip, nor the dependency chain is
   responsible: all three were measured and are ≤1–2 %.
3. **The single biggest lever is rows per K/V sweep.** `CPU_ATTENTION_QT=8` — a
   dispatch knob, no kernel change — is **1.20× faster at D=64/S=2048, 1.25× at
   D=128/S=1024, 1.34× at D=128/S=4096**, and it halves the L2 demand-miss rate
   (6.34 → 2.62 %, 5.33 → 4.17 %). That is the mechanism, and it says the
   direction to build into the kernel is *more query rows per K/V read*, not a
   bigger softmax tile.
4. **The "big key tile" claim is demoted, not confirmed.** Per-tile fixed work is
   **23 % of our instructions at D=128** and **8.4 % at D=64**; removing 100 % of
   it caps the win at 1.18× / 1.06×, and a 480-key tile cannot remove 100 %
   because `exp` scales with keys. The parts of it that *are* real — loop-overhead
   amortisation and a `kt=32`→`64` port-ratio change — are worth ~15 % at D=128
   and are better had by fixing the row count and the K layout.
5. **At short S our kernel is already faster than vLLM's kernel; we lose on the
   pass before it.** S=128: ours 187 µs of attention + 65 µs K-transpose.
   vLLM: 210 µs of attention + **124 µs** of `cpu_attn_reshape_and_cache`, which
   is the same transpose and runs at 4 GB/s against our 8 GB/s. Charged fairly we
   are **1.32× faster end-to-end at S=128**, and 2.2× slower at S=1024/D=128.
   The remaining short-S overhead that is genuinely ours: `flash_attn_func`'s
   validation (10 % of the call at S=128) and two `torch.empty`s (1.4 %).
6. **bf16 at D=128 is not at parity on this box** — 1.91× behind vLLM
   single-threaded, and 5× behind SDPA, which is AMX-backed here. The AMX ceiling
   is real and no `vdpbf16ps`-shaped kernel will reach it (see §7).

---

## 1. Tooling: what worked, what did not

| tool | status | note |
|---|---|---|
| `perf stat` | **works** (needs `sudo`; `perf_event_paranoid=4`) | 8 general counters per group, no multiplexing if you stay ≤8; `LLC-load-misses` is `<not supported>`, use `l2_rqsts.demand_data_rd_miss` |
| `perf record` | **fails**: `PMU Hardware doesn't support sampling/overflow-interrupts` | so no per-instruction cycle attribution; per-phase cycles are `instruction share × measured cycles` (§4) |
| `valgrind --tool=callgrind` | **fails**: `SIGILL` on our AVX-512 kernel after 3.6 G Ir | no line-level Ir/cycle attribution either; replaced by static `objdump --line-numbers` on a `-g1` build, validated against `perf`'s `instructions` counter (§4) |
| `llvm-mca` (21) | works, on the extracted inner loops | gives the port floor (§4) |
| `objdump -d` | works | static instruction accounting, `scripts/prof/x86/asm_count.py` |

Two consequences to keep in mind when reading the tables: (a) every counter
includes ~3.2 G cycles / 4.6 G instructions of Python+torch start-up, so per-call
numbers are `(total − baseline)/(warm 3 + reps)` with a baseline process that
does the same setup and zero calls (`perf_baseline` mode of `perf_matrix.sh`);
(b) per-phase *cycles* assume uniform IPC across phases, which is an
approximation — the per-phase *instruction* counts are exact.

---

## 2. Headline table (single thread, medians of 9, one process)

`us` per call; `GFLOP/s` in brackets; causal, scale `1/sqrt(D)`, b=1.

| S : H : D | dtype | **ours** | **vLLM** | SDPA | ours/vLLM | ours/SDPA |
|---|---|---:|---:|---:|---:|---:|
| 2048:8:64 | fp32 | **68 492** (63) | 30 461 (141) | 39 436 (109) | 2.25× | 1.74× |
| 2048:8:64 | bf16 | **48 095** (89) | 33 225 (129) | 13 520 (318) | 1.45× | 3.56× |
| 1024:16:128 | fp32 | **78 900** (54) | 31 585 (136) | 52 375 (82) | 2.50× | 1.51× |
| 1024:16:128 | bf16 | **60 919** (71) | 31 900 (135) | 12 224 (351) | 1.91× | 4.98× |
| 512:8:64 | fp32 | **2 707** (99) | 2 113 (127) | 4 425 (61) | 1.28× | 0.61× |
| 512:8:64 | bf16 | **3 070** (87) | 2 291 (117) | 1 627 (165) | 1.34× | 1.89× |
| 128:8:64 | fp32 | **278** (60) | 188 (89) | 324 (52) | 1.48× | 0.86× |
| 128:8:64 | bf16 | **292** (57) | 184 (91) | 159 (105) | 1.59× | 1.83× |

Parity was checked in the same process for every row (`max|d|` ≤ 1.6e-2 bf16,
≤ 4.8e-7 fp32 against SDPA; vLLM identical to us to the same tolerance), so no
row is a timing of skipped work.

All three variants are slower here than the 8-thread numbers in the repo README.
That is the point of the single-threaded protocol — one core, no SMT, no pool —
and it is the only way to compare instruction streams.

Repeatability: the whole table was re-run later in a fresh process (raw output in
`scripts/prof/x86/data/bench_1t_repeat.txt`) and every row agrees within 1–6 %
(worst: 1024:16:128 fp32, 74 506 vs 78 900 µs — box drift, which is why every
cross-variant claim below is a ratio measured *inside* one run).

---

## 3. Counters per call (what the mechanism is)

`Mcyc`/`Minsn` = per call after baseline subtraction; `in/pair` = instructions per
(query row, key) MAC pair, which is the blocking-independent efficiency measure;
`D1fill%` = `l1d.replacement / loads`, `L2miss%` = `l2_rqsts.demand_data_rd_miss /
loads`, `Ldbnd%` = `exe_activity.bound_on_loads / cycles`.

| variant | shape | dtype | Mcyc | Minsn | IPC | FMA/cyc | in/pair | D1fill% | L2miss% | Ldbnd% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ours | 2048:8:64 | fp32 | 260.3 | 319.9 | **1.23** | 0.52 | 19.07 | 42.4 | **5.97** | **48.3** |
| vllm | 2048:8:64 | fp32 | 109.3 | 313.2 | **2.87** | 1.23 | 18.67 | 20.6 | **0.07** | **14.6** |
| sdpa | 2048:8:64 | fp32 | 139.9 | 368.1 | 2.63 | 0.96 | 21.94 | 17.4 | 0.16 | 13.4 |
| ours | 1024:16:128 | fp32 | 322.8 | 397.7 | **1.23** | 0.42 | 47.41 | 29.2 | **5.21** | **35.7** |
| vllm | 1024:16:128 | fp32 | 100.7 | 296.1 | **2.94** | 1.33 | 35.30 | 21.5 | **0.19** | **13.0** |
| ours | 512:8:64 | fp32 | 13.4 | 27.2 | 2.03 | 0.63 | 25.94 | 36.8 | 0.42 | 21.3 |
| vllm | 512:8:64 | fp32 | 7.5 | 21.2 | 2.85 | 1.12 | 20.26 | 20.4 | −0.00 | 22.6 |
| ours | 1024:16:128 | bf16 | 246.8 | 345.8 | 1.40 | 0.54 | 41.22 | 26.1 | 0.98 | 22.1 |
| vllm | 1024:16:128 | bf16 | 116.5 | 311.3 | 2.67 | 1.15 | 37.11 | 11.5 | 0.03 | 7.0 |
| sdpa | 1024:16:128 | bf16 | 49.6 | 57.7 | 1.16 | **2.71** | 6.88 | 65.0 | 2.09 | 37.3 |
| ours | 2048:8:64 | bf16 | 210.6 | 269.7 | 1.28 | 0.64 | 16.08 | 31.6 | 0.93 | 20.6 |
| vllm | 2048:8:64 | bf16 | 123.7 | 328.6 | 2.66 | 1.08 | 19.59 | 11.4 | −0.00 | 7.5 |

Three facts to take away:

* **`in/pair` is at parity for fp32 at D=64** (19.07 vs 18.67) and only +34 % at
  D=128 (47.41 vs 35.30), while IPC is 2.3–2.4× worse everywhere. Instruction
  count cannot explain a 2.2–2.5× timeout.
* **`FMA/cyc` for bf16 SDPA is 2.71 with `in/FMA` 0.43** — that is not a
  16-lane formulation; that is AMX (256 MACs per instruction). It is the ceiling
  reference, not a competitor.
* The L2 demand-miss rate is the only counter that separates us from vLLM by
  more than the runtime ratio, in the same direction, at every long-S shape.

---

## 4. Where the instructions go, per phase

Method: build the same sources with `mojo build -g1` into `/tmp/C_dbg.so`
(no source change), then
`objdump -d --line-numbers C_dbg.so | scripts/prof/x86/asm_count.py --sym 'dtype=f32,head_dim=128,qt=4' --bucket ...`.
Dynamic counts = static count × trip count from the source; validated against
`perf stat`'s `instructions`, which they match to 84 % (D=128) / 90 % (D=64), the
residual being prologue/epilogue bookkeeping the buckets do not cover.

**D=128 fp32 (`_dense_rows_mem`, `qt=4`, `kt=32`) — 3 678 instructions per 32-key
tile, 28.7 per (row,key) pair:**

| phase | instr/tile | share | what it is |
|---|---:|---:|---|
| Q@K^T | 2 176 | **44 %** | 17 instructions per head-dim step: 2×`vmovups` (K), 4×`vbroadcastss` (q, one per row), 8×`vfmadd231ps`, 3 loop/addressing |
| P@V | ~1 920 | **39 %** | 4 head-dim chunks × (16 acc loads + 16 acc stores + 32 keys × (2 V loads + 4 prob broadcasts + 8 FMA)) |
| softmax | 643 | 13 % | 2×`reduce_max`/row, `_vexp` (378 = 59 % of it), `reduce_add`, rescale, `probs` store |
| mask + tile head | 237 | 6 % | `iota`+`lt`+`select` per row (boundary tiles only), loop re-entry, phi materialisation |

**D=64 fp32 (`_dense_rows`, `qt=4`, `kt=64`) — 5 240 instructions per 64-key tile,
20.5 per pair:** P@V **54 %**, Q@K^T 38 %, softmax 6 %, tile head 2.5 %.

**llvm-mca on the D=128 Q@K^T loop** (17 instructions, SAPPHIRERAPIDS):
`Block RThroughput: 8.0`, IPC 2.11 → **8 cycles per head-dim step = 1.0 FMA/cycle**,
because 6 zmm loads (2 K + 4 broadcast) occupy the load ports for 8 cycles while
the 8 FMAs need only 4. At `qt=8, kt=32` the same loop is 12 cycles per 16 FMAs =
1.33; at `qt=8, kt=64` it would be 16 per 32 = **2.0 (peak)**. So there is a
*structural* ceiling of 1.0 FMA/cycle on the current D=128 configuration that we
are not even reaching (measured 0.42) — memory binds first, ports second.

---

## 5. Ranked hypotheses (mine, from these numbers)

Each: **claim → evidence → mechanism → expected payoff → falsification test → confidence**.

### #1 More query rows per K/V sweep (`qt` 4 → 8, eventually 16+). Payoff **1.20–1.34×, measured**. Confidence **high**.

*Evidence.* `CPU_ATTENTION_QT` is a dispatch knob, so this is an experiment, not a
port. Single thread, one process each, `perf`-counted:

| shape | qt | µs | Mcyc | IPC | in/pair | D1fill% | L2miss% | Ldbnd% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024:16:128 fp32 | 2 | 127 203 | 478.7 | 0.77 | 43.68 | 64.3 | 16.68 | 63.0 |
| | **4** | 76 537 | 316.9 | 1.25 | 47.40 | 29.1 | 5.33 | 37.5 |
| | **8** | **61 399** | 248.2 | 1.41 | 41.83 | 17.9 | 4.17 | 35.6 |
| 2048:8:64 fp32 | 2 | 109 374 | 418.2 | 0.91 | 22.70 | 61.4 | 10.02 | 58.5 |
| | **4** | 69 679 | 269.0 | 1.19 | 19.02 | 42.3 | 6.34 | 46.7 |
| | **8** | **58 031** | 232.1 | 1.54 | 21.33 | 16.5 | 2.62 | 33.9 |
| 4096:8:128 fp32 | 4 | 760 273 | — | — | — | — | — | — |
| | **8** | **569 118** | — | — | — | — | — | — |

  (45.2 → 60.4 GFLOP/s at S=4096; vLLM at that shape: 219 ms, 156 GFLOP/s.)
*Mechanism.* K/V traffic per output row is ∝ 1/qt. Doubling rows halves it: D1 fills
42 → 16.5 %, L2 demand misses 6.34 → 2.62 % at D=64 (a 2.4× fall for a 2× traffic
cut), and load-bound cycles 47 → 34 %. It is *not* an instruction win (in/pair gets
*worse* at D=64: 19.0 → 21.3) — which is exactly why it is evidence for the traffic
mechanism rather than for better code generation.
*Payoff.* Free 1.2–1.34× on the shapes that matter, and it is the foundation the
rest of the plan needs (a memory-resident accumulator is what lets qt go to 16).
*Falsification.* If the win were ILP/latency instead of traffic, the L2-miss rate
would not fall with qt — it falls by ~2.4×. If it were noise/drift, an interleaved
in-process A/B (two `flash_attn_func` closures, one per `CPU_ATTENTION_QT`) would
not reproduce the ordering; that A/B is the remaining check (see §9).
*Caveat that matters.* `AGENTS.md` §9.10 records "qt = 2/4/6/8 sits within 1 %" and
§4a "8 was 20–40 % slower than 4" — both from the laptop at 8 threads. On this box,
single-threaded, **qt=8 is faster at every shape I measured** (D=128 and D=64,
S=128…4096). Whatever the default becomes, it has to be re-derived per box; the
current hard-coded `_QT = 4` is leaving 20–34 % on the floor here.

### #2 Make the K read a *stream*: block-major K tiling instead of one `[head_dim, seq]` array. Payoff **1.3–1.5× at long S** (estimate). Confidence **medium**.

*Evidence.* L2 demand-miss rate 5.97 % vs vLLM's 0.07 % (D=64/S=2048), 5.21 % vs
0.19 % (D=128), load-bound cycles 48 % vs 15 %. It is **shape-dependent in exactly
the way a cache-residency story predicts**: at S=512, where one head's transposed K
is 128 KB and stays resident, our L2 miss rate drops to 0.42 % and the runtime
ratio falls from 2.25× to 1.28×.
*Mechanism.* Our K^T is one `[head_dim, seq_pad]` array per (batch, head), so a
32-key tile reads `head_dim` separate 128 B segments spread `seq_k*4` bytes apart —
64 streams at S=2048 (8 KB stride), 128 streams at D=128 (16 KB stride). That
defeats the L2 streamer; vLLM's cache is `[head_dim, block_size]` **per 128-token
block** (`cpu_attn_vec.hpp:204-223`), so the same strided read is contained in a
32 KB block and the *blocks* are walked sequentially (`cpu_attn_impl.hpp:1228-1276`)
— every line is used 8 times (once per q row) while hot and the next block is
prefetched.
*Payoff.* 1.3–1.5× at long S, additive with #1 (they reduce different halves of
the same traffic).
*Falsification (cheap, ~30 min).* Keep the layout, change the *loop order* only:
hoist the Q@K^T `d` loop so a 256-key super-tile's `head_dim` rows are touched
once each, i.e. `for super_tile: for d: for sub_tile:` with the same arithmetic.
If `l2_rqsts.demand_data_rd_miss / loads` does not fall below ~1 %, the stream
hypothesis is wrong and the residue is pure volume — in which case #1/#3 are the
only levers and #2 should be dropped. **Do this before writing any layout code.**

### #3 Memory-resident accumulator so `qt` can reach 16–32 (vLLM's structure). Payoff **1.2–1.4×** on top of #1 at long S. Confidence **medium**.

*Evidence.* Traffic ∝ 1/qt and #1 shows the traffic term is what binds (qt=8 still
leaves us 2.6× behind vLLM at S=4096/D=128 with L2miss 2.6 % vs 0.07 %). The
register file is what stops qt: the accumulator alone is `qt*head_dim/16` zmm
(32 at qt=4/D=128), which is why `_dense_rows_mem` exists at all. vLLM keeps the
accumulator in `partial_q_buffer` and never zeroes it (`accum_c = !is_first_iter`,
`cpu_attn_impl.hpp:1337`), and the logits in `logits_buffer` with the probabilities
aliasing it in place (`:1318-1330`), which is what makes 8 rows × 512 keys fit.
*Mechanism.* Not "the accumulator in L1 is slow" — it is that the row count is
pinned by registers, and the row count is the arithmetic intensity.
*Payoff.* ~1.2-1.4x, only after #2 (otherwise the extra rows just make a bigger
strided sweep).
*Falsification.* We already ran it in the cheap direction: at qt=8 the D=64 path
switches to `_dense_rows_mem` (accumulator in L1) and gets *faster*, not slower —
so "accumulator in memory" is not itself a penalty. If a qt=16 prototype shows no
further L2-miss reduction, the traffic model is exhausted and this item is dead.

### #4 (Claim A, demoted) Bigger key tile / fewer per-tile softmax reductions. Payoff **≤1.18× at D=128, ≤1.06× at D=64** — i.e. bounded, and mostly *unreachable*. Confidence **high that it is not the main term**.

*Evidence.* Per-tile fixed work = tile head (186) + softmax (643) = **829 of 3 678
instructions per 32-key tile at D=128 (23 %)**; at D=64 it is 440 of 5 240 (8.4 %).
Even at zero, the win is `1/(1−0.23)` = 1.18× at D=128. And it cannot go to zero:
of the 643, `_vexp` is 378 and it scales with the number of keys; a 480-key tile
keeps the `exp` and removes only the per-tile reductions, rescale, mask and
re-entry — so the realistic payoff is ~10 % at D=128 and ~4 % at D=64.
*Sub-claims, measured individually:*
- **mask-by-range**: 51 instructions per boundary tile, ~1 boundary tile per
  q-block → ~1 % of the call. Real, trivial, not worth a recommendation on its own.
- **−8 rescale threshold** (`cpu_attn_impl.hpp:1468`): our rescale is 121+32
  instructions per tile (line 552/549 of `dense.mojo`) and already skipped when
  `alpha == 1`; at D=128 that is ~4 % of the call. Also real, also small.
- **`kt` 32 → 64** (the part that *is* worth something): the Q@K^T loop is
  2.6 instr/FMA at `kt=32` and 1.94 at `kt=64` because the 3 loop/addressing
  instructions amortise over twice the FMAs; Q@K^T is 44 % of our instructions at
  D=128, so this is ~15 % of the call. This is a *tile-shape* argument, and it is
  the one piece of the big-tile thesis that survives.
*Falsification (already run).* If per-tile fixed cost were the main term, then
`in/pair` should fall sharply when the *effective* tile grows. The qt sweep grows
rows per tile, not keys, and it *raised* in/pair at D=64 (19.0 → 21.3) while
getting 1.2× faster — the per-tile-cost term is not what moves the runtime.

### #5 (short S) The transpose pass, then Python. Payoff **1.2× at S=128**. Confidence **high**.

Numbers in §6.

---

## 6. Short S: the fixed-cost breakdown (Claim C), and it was mis-framed

S=128/H=8/D=64 **fp32**, single thread, medians of 15, all rows interleaved:

| component | µs | % of our call |
|---|---:|---:|
| dense kernel call (27 args, buffers preallocated) | 187.4 | 67.2 |
| K^T transpose pass | **65.0** | **23.3** |
| `flash_attn_func` validation (full − `_dense`) | **28.8** | **10.3** |
| `torch.empty(kt)` + `torch.empty_like(out)` | 3.8 | 1.4 |
| 27-arg marshalling (`batch=0` call) | 3.3 | 1.2 |
| **total** | **278.9** | |

And the same decomposition at the other protocol shapes:

| shape | total µs | dense | transpose | wrapper | allocs | marshal |
|---|---:|---:|---:|---:|---:|---:|
| 128:8:64 | 278.9 | 67.2 % | **23.3 %** | 10.3 % | 1.4 % | 1.2 % |
| 256:8:64 | 845.9 | 79.0 % | 15.1 % | 3.1 % | 1.1 % | 1.0 % |
| 512:8:64 | 2 659.7 | 88.6 % | 9.5 % | 1.8 % | 0.5 % | 0.4 % |
| 1024:16:128 | 86 222.7 | 90.0 % | 3.3 % | 1.8 % | 0.05 % | 0.02 % |

**The two `GOMP_parallel` regions are not a cost, they are the win.** S=128 with
4 threads on 4 physical cores: `CPU_ATTENTION_WORKERS=1` (inline, no region)
255/256 µs, `=4` (one region, 4 workers) 106/103 µs — **2.45×**, and the region
itself is inside the 2.4 µs/2.7 µs cold/warm figure already in `AGENTS.md` §7.
There is no "GOMP overhead" to recover; it is 1 % of a 100 µs call for 2.45×.

**The fair framing matters.** vLLM has no transpose inside
`cpu_attention_with_kv_cache`, but it pays for the same K column-majorisation in
`cpu_attn_reshape_and_cache`, and pays *more*:

| shape | our transpose | vLLM `reshape_and_cache` | ratio |
|---|---:|---:|---:|
| 128:8:64 | 65.0 µs (8.1 GB/s) | **123.7 µs (4.2 GB/s)** | 1.9× |
| 512:8:64 | 253.5 µs (8.3 GB/s) | **548.8 µs (3.8 GB/s)** | 2.2× |
| 1024:16:128 | 2 845 µs (5.9 GB/s) | **4 761 µs (3.5 GB/s)** | 1.7× |

So charged end-to-end (attention + staging), **S=128: ours 252 µs, vLLM 333 µs —
we are 1.32× faster**, and vLLM's kernel is the slower half of its own call. At
S=1024/D=128: ours 80.4 ms, vLLM 36.6 ms — vLLM 2.2× faster, staging included.
The "we are 2–3× slower at short S" premise is true of the *attention call* and
false of the *layer*.

What is genuinely ours and fixable: `flash_attn_func`'s validation is 29 µs (10 %)
at S=128 — the engine already uses the thin `self_attention` alias to dodge it
(`AGENTS.md` §8), and the two allocations are 3.8 µs. The transpose at 8 GB/s is
the one real target (a 4-thread run does it at 36 µs, so it parallelises).

---

## 7. The four questions, answered

### Q1. fp32 D=128, our worst path: spill traffic, the `probs` round trip, a dependency chain, or issue bandwidth?

**None of the four. It is data supply, and the candidate micro-causes are each
measurably small:**

| candidate | measurement | share of 260–323 M cycles |
|---|---|---|
| spill traffic | the Q@K^T loop (44 % of instructions) has **zero** stack accesses except the tile-invariant `q` broadcasts; the P@V loop keeps `accv` in zmm. Static stack-frame instructions are 16 % of the *function* but live in the tile head and softmax | ~6 % (the 237 instruction tile  head + part of the 643) |
| `probs` round trip | `ld_blocks.store_forward` = 0.005–0.08 per (row,key) → ≤1 store-forward stall per 12 keys, ~5 cycles each | ~1 % |
| dependency chain | Q@K^T has 8 independent accumulator chains (4 rows × 2 zmm) at 4 cycles FMA latency = 2 FMA/cycle available; P@V has 8 | 0 |
| issue bandwidth | IPC 1.23 of a 6-wide dispatch | 0 |
| **data supply** | **D1 fill 29 %, L2 demand miss 5.2 %, backend bound on loads 36 % (vLLM: 21 / 0.19 / 13)** | **dominant** |

The instruction-level deltas versus vLLM, largest first (D=128 fp32, per (row,key)
pair): **(1)** +12.1 in/pair overall (47.41 vs 35.30), of which **(2)** ~4 in/pair
is `kt=32` loop/addressing overhead in Q@K^T (2.6 vs 1.94 instr/FMA), **(3)**
~3 in/pair is the `_dense_rows_mem` head-dim chunking (load+store the accumulator
to L1 four times per key tile, 32 instructions per tile per chunk), **(4)**
~1.5 in/pair is `_vexp`'s constant rematerialisation (7 `vbroadcastss` per vector
rather than registers held across the loop — deliberate regalloc, cheap per uop),
**(5)** ~1 in/pair is the per-tile mask/reduce/rescale. Items 2–5 together are the
whole +34 %, and none of them is the 2.4× IPC gap.

### Q2. fp32 D=32/64 (1.2–1.3× behind vLLM, faster than SDPA)

At D=64 fp32 we have **the same instruction count as vLLM** (19.07 vs 18.67
in/pair) and 2.3× worse IPC (1.23 vs 2.87). Per phase, P@V is the instruction
leader (54 %), then Q@K^T (38 %), then softmax (6 %). The gap is again traffic
(L2 miss 6.0 % vs 0.07 %, load-bound 48 % vs 15 %), and it is quadratic in S: at
S=512 we are 1.28× behind and our L2 miss rate is 0.42 %, at S=2048 we are 2.25×
behind at 5.97 %. `qt=8` recovers 1.20× of it (§5 #1) without touching the kernel.

### Q3. Short S: which of (a) transpose, (b) attention, (c) Python/allocation, (d) two GOMP regions dominates?

**(b) attention dominates from S=256 up; at S=128 the transpose (23 %) and
Python (10 %) together are a third of the call; (d) is a 2.45× win, not a cost.**
Full table in §6. The one correction to the premise: vLLM pays for its staging
too, at half our throughput, so the layer-level comparison at S=128 favours us
1.32×.

### Q4. bf16 at D=128: is there headroom, and where?

**On this box we are not at parity — 1.91× behind vLLM, 5× behind SDPA
single-threaded.** bf16 shows the same mechanism as fp32 (D1 fill 26 % vs 11 %,
L2 miss 0.98 % vs 0.03 %, load-bound 22 % vs 7 %, in/pair 41.2 vs 37.1), so #1–#3
apply unchanged. Beyond that the headroom is architectural: SDPA's 49.6 M cycles
for this shape is `FMA/cyc 2.71 at in/FMA 0.43`, which is AMX
(`amx_bf16`+`amx_tile` are both present in `/proc/cpuinfo`). A kernel built on
`vdpbf16ps` (32 MACs/instr) cannot approach 256 MACs/instr, so **the bf16 ceiling
on an AMX box is not reachable by tuning the dot path** — it needs the 16-row tile
+ tile-loads (`tile_stream_loadd`), i.e. a different kernel, and the honest target
for a `vdpbf16ps` formulation is vLLM's VEC number (135 GFLOP/s at D=128), not
SDPA's 351.

---

## 8. What vLLM actually does — verified against the source, with the claims I could kill

Their VEC decode/prefill kernel is `AttentionImpl<ISA::VEC>` +
`AttentionMainLoop` (`~/vllm_cpu_build/csrc/cpu/cpu_attn_vec.hpp`,
`cpu_attn_impl.hpp`). The things that matter, in the order they matter to us:

1. **Rows, not tiles.** `MaxQHeadNumPerIteration = 8` (`cpu_attn_vec.hpp:156`) and
   `TileGemm82` is a **8-row × 32-position** micro-tile with 16 accumulator
   registers (`cpu_attn_vec.hpp:36-133`). Their prefill packs 8 query *tokens* into
   one M (`cpu_attn_impl.hpp:1799-1803`, `curr_max_q_token_num_per_iter =
   max_q_head_num_per_iter / q_heads_per_kv`), so one K/V read serves 8 rows. Ours
   serves 4 (`qt=4`) — and that is worth the measured 1.2–1.34× (§5 #1).
   Their inner loop is FMA-contracted (`vfmadd231ps`, confirmed in the built `.so`:
   2 loads + 8 `vbroadcastss` + 16 FMA per head-dim step = 1.75 instr/FMA).
2. **The key tile is L2-sized — but the payoff is not where the claim said.**
   `calcu_tile_size_with_constant_q` (`cpu_attn_impl.hpp:989`) divides
   `get_available_l2_size()` (= `l2_cache_size >> 1`, `utils.hpp:61`) by the
   per-token footprint, giving ~480 keys at D=128 fp32 / ~992 at D=64. One
   `tile_gemm<QK>` writes the whole tile, `apply_softmax` (`:1280-1311`) runs
   **once per tile**, then `tile_gemm<PV>` consumes it. **But** I measured what
   that per-tile work actually costs us: 23 % of instructions at D=128 and 8.4 % at
   D=64, of which only the reductions/rescale/mask/re-entry are tile-fixed (the
   `exp` scales with keys) → **≤1.18× / ≤1.06× even if it went to zero** (§5 #4).
   The tile size is still worth having, for the *loop-overhead amortisation* and
   *K-locality* reasons, not for the softmax reductions.
3. **Mask by range instead of per lane** (`apply_mask`, `cpu_attn_impl.hpp:1360-1426`):
   the invalid region is contiguous per row, so they write `-inf` into
   `[0,left_invalid)` and `[right_invalid,n)` with scalar stores. We do
   `iota`+`lt`+`select` on every lane of the boundary tile. **Measured: 51
   instructions per boundary tile → ~1 % of a call.** True, cheap, not the story.
4. **Rescale threshold −8** (`:1468`, `need_rescale = (init_max − new_max) < −8.0`,
   "same rescale threshold as FA4"); when it does not fire they do not even move
   the running max. We rescale whenever the max increases. **Measured: the rescale
   is 121+32 instructions per tile ≈ 4 % at D=128, and we already skip it when
   `alpha == 1`.** True, small.
5. **Probabilities alias the logits buffer** (`:1318-1330`), read back through
   `ProbabilityTokenStore` whose `TokenStride` is 1 for VEC (`:1117-1129`), i.e. no
   packing on this path. We write `probs` to a stack buffer and re-read it as
   4-byte scalars — which *sounds* like a store-forwarding stall per key (and is
   what `AGENTS.md` §4 blames in the C++ kernel). **Measured:
   `ld_blocks.store_forward` = 0.005–0.016 per (row,key) at D=128, i.e. ~1 % — it
   is not a problem, because a 4-byte load contained in a just-written 64-byte
   store forwards fine.** Hypothesis dead; do not spend time on it.
6. **KV split** (`enable_kv_split`, `kv_split_pos_start/end`, `partial_output_buffer_t`,
   `reduce_splits` with the `exp(m_local − m_global)` rescale,
   `cpu_attn_impl.hpp:664-700`, `:2131-2215`) and work items packed by
   **KV length** rather than round-robin (`kv_len_per_thread`, `:539-700`).
   Ours has neither. See §8.1 for whether that is worth anything to us.
7. `vec16` (`cpu_attn_vec16.hpp`) is a **16-row × 16-position** tile with
   `MaxQHeadNumPerIteration = 16` — the "more rows" direction again, taken to 16.
   It exists for prefill-shaped work; the generator
   (`generate_cpu_attn_dispatch.py`, `cpu_attn_dispatch_generated.h`) emits head_dim
   32/64/80/112/128 × {vec, vec16, amx, …} specialisations, so their 8-row VEC is a
   deliberate compromise, not a limit.

### 8.1 Would a KV split help our long-S rows (S=4096/D=128 fp32)? **No — not for parallelism, and it is the wrong tool for locality.**

*Parallelism:* at S=4096/H=8/D=128 with `qt=4` we have 1·8·1024 = 8 192 independent
tasks for 4–8 threads, and round-robin over a monotone work function is already
balanced (`sum over g ≡ i mod T of 4g ≈ total/T`). There is nothing to fix.
*What a split would change:* each thread's sweep would be 4096/8 = 512 keys instead
of 4096, so a thread's K/V working set would drop from 4 MB to 512 KB and stay in
L2 — i.e. **it buys locality, at the price of a combine pass** (per split: 1 row ×
128 dims of output, plus `exp(m_local − m_global)` — trivial in flops).
*But there is a cheaper way to buy the same locality:* tile the K sweep **inside**
a task (read a 512-key super-tile of K/V once, finish it, move on). That gets the
same residency with **no combine, no extra buffer, no exp correction**. So: the
vLLM KV split is a *load-balancing* device for their schedule (few work items per
thread, one long q-tile can exceed a thread's share); it is not the answer to our
long-S gap, and implementing it would be strictly dominated by #2.

---

## 9. The one experiment I would run next (if I had the box another hour)

**An interleaved in-process A/B of `qt=4` vs `qt=8` vs `qt=8` + super-tiled K
loop, 4 threads, on 1024:16:128 and 4096:8:128 fp32, with `l2_rqsts` counted for
each arm.** Three questions fall out of it at once: (a) does #1 survive
multithreading (its 1.2–1.34× is single-threaded; with 4 threads the L2 pressure
is shared, so the win could shrink or grow); (b) does #2's loop reordering recover
the L2-miss rate without changing the layout (the falsification test for #2, and it
is ~30 lines in the Q@K^T loop); (c) whether #1 and #2 are additive. Everything
needed is already committed: `scripts/prof/x86/bench_shapes.py`,
`run_one.py --qt`, `perf_matrix.sh`, `analyze.py`.

Cheap follow-ups if that is inconclusive, in order: mask-by-range + `kt=64` for the
D=128 path (bounded ≤15 %, low risk), then the memory-resident accumulator (#3).

---

## 10. Files

```
scripts/prof/x86/_common.py        shapes, vLLM loader, paged-cache construction, interleaved timer
scripts/prof/x86/bench_shapes.py   headline table (ours / vLLM / SDPA), parity check, --json
scripts/prof/x86/run_one.py        one (variant, shape, dtype) N times: the perf/callgrind unit
scripts/prof/x86/components.py     short-S fixed-cost split: transpose / allocs / python / marshal / call
scripts/prof/x86/perf_matrix.sh    the counter matrix (2 <=8-event groups, per-variant processes)
scripts/prof/x86/perf_baseline.sh  the reps=0 half of it: process start-up cost to subtract
scripts/prof/x86/analyze.py        CSVs -> the tables in §3 (baseline subtraction, per-call, per-pair)
scripts/prof/x86/asm_count.py      objdump --line-numbers -> per-source-line instruction classes + buckets
scripts/prof/x86/data/perf_1t.csv          the raw counters behind §3
scripts/prof/x86/data/base_1t.csv          the matching reps=0 baselines
scripts/prof/x86/data/qt_sweep_fp32.csv    the raw counters behind §5 #1
scripts/prof/x86/data/bench_1t_repeat.txt  the wall-clock repeat of §2
```

Reproduce (the box needs `sudo apt-get install -y valgrind llvm`):

```bash
cd ~/cpu-attention
P=$PWD/.venv/bin/python
taskset -c 3 env OMP_NUM_THREADS=1 $P scripts/prof/x86/bench_shapes.py --dtype both --with-sdpa
sudo bash scripts/prof/x86/perf_matrix.sh > /tmp/perf.csv      # + the reps=0 baseline pass
$P scripts/prof/x86/analyze.py /tmp/perf.csv /tmp/base.csv
taskset -c 3 env OMP_NUM_THREADS=1 $P scripts/prof/x86/components.py --dtype fp32
# static per-phase counts (no source change: -g1 into /tmp)
mojo build cpu_attention/_mojo/src/_C.mojo --emit shared-lib -o /tmp/C_dbg.so -g1 \
     -Xlinker -l:libgomp.so.1 -Xlinker -L$(dirname $(python -c 'import torch;print(torch.__file__)'))/lib
objdump -d --line-numbers /tmp/C_dbg.so | $P scripts/prof/x86/asm_count.py \
     --sym 'dtype=f32,head_dim=128,qt=4' --bucket 497-517:QK --bucket 519-530:mask \
     --bucket 532-558:softmax --bucket 560-574:PV
```

## 11. Confidence summary

| finding | confidence | why |
|---|---|---|
| IPC, not instruction count, is the D=64 gap | **high** | in/pair at parity, IPC 2.3× apart, both measured |
| data supply (L1/L2) is the binding constraint | **high** | L2 miss 80×, load-bound 3.3×, both track the S-dependence |
| qt=8 is worth 1.20–1.34× today | **high** | direct measurement at 3 shapes, mechanism visible in the counters |
| per-tile softmax cost ≤1.18× (D=128), ≤1.06× (D=64) | **high** | instruction counts per tile, arithmetic |
| block-major K tiling worth 1.3–1.5× | **medium** | inferred from miss rates + vLLM's layout; the loop-order test has not been run |
| store-forwarding / spills / dependency chains are <2 % | **high** | each counted |
| bf16 on this box cannot reach AMX-backed SDPA | **high** | 2.71 FMA/cyc at 0.43 in/FMA vs a 32-MAC instruction |
| KV split would not fix S=4096 | **medium-high** | parallelism is already 8 192 tasks; the locality argument favours in-task tiling |

---

# Round 2 — the confound closed, two hypotheses killed, one retraction

New box (`i-013035a8618efeb56`, same type: 4 physical cores, 2 MB L2/core, ~3.8 GHz).
Same protocol unless stated. Nothing above has been edited; the claims this round
falsified are marked **RETRACTED** here rather than rewritten.

## R2.1 The `qt` confound: **the rows carry the win, the kernel swap is a 6 % loss**

`CPU_ATTENTION_QT=8` at head_dim 64 does not only add rows: the dispatch rule
`qt * head_dim > 384` routes it from `_dense_rows` (register accumulator) to
`_dense_rows_mem` (L1 accumulator). To separate the two factors I built two
private `.so`s from copies of the tree (sources untouched in the repo):

* `/tmp/src_mem/dense.mojo` — threshold `> 96`  → **mem kernel at any qt**
* `/tmp/src_reg/dense.mojo` — threshold `> 1000000` → **register kernel at any qt**

and ran the 2×2 at **2048:8:64 fp32**, one process per cell, `CPU_ATTENTION_LIB`
pointing at the cell's build:

| | qt=4 | qt=8 | row effect |
|---|---:|---:|---:|
| register kernel (`_dense_rows`) | 66 057 µs | **55 456 µs** | **1.19×** |
| mem kernel (`_dense_rows_mem`) | 69 929 µs | 59 086 µs | **1.18×** |
| kernel effect | 0.94× | 0.94× | |

**Verdict: rows are worth 1.18–1.19× on their own, and the L1-accumulator kernel
is consistently 6 % *slower* when the register kernel is also available.** So the
recommendation stands as "raise `qt`", not "port the mem path to qt=4", and the
default build's `reg@qt8` (55.5 ms here, `std` 57.6 ms in the sweep) is the best
cell. Cross-check: `std@qt4` = 66.1 in the sweep vs `reg@qt4` = 66.06 here —
the two builds are the same binary for that cell, so the harness is not the
variable.

Consequence worth recording: `AGENTS.md` §4a says the no-spill mem path is
"within 1 %" of the register version. That is only true where the register
version is unusable (qt≥4 at D=128); where both are legal the register kernel
wins by 6 %.

## R2.2 Two address-pattern probes: the traffic/miss hypothesis is **RETRACTED**

I claimed (§3, §5 #2, §5 #3) that the gap was data supply, with the L2 demand-miss
rate as the causal counter, and proposed block-major K as the fix. I probed both
halves of the traffic directly, by overriding **only the strides** the kernel is
handed — same instructions, same load count, same tile count, wrong numbers:

| arm | 2048:8:64 fp32 | L2miss% | K panel | V panel |
|---|---:|---:|---:|---:|
| qt=4, real layout | 66 087 µs | 6.78 | 512 KB | 512 KB |
| qt=4, `--kt-stride 512` | 65 993 µs | 6.75 | 128 KB | 512 KB |
| qt=4, `--kt-stride 64` (L1-resident) | 65 850 µs | **6.69** | **32 KB** | 512 KB |
| qt=4, `--v-stride 8` (V in 64 KB) | 65 319 µs | — | 512 KB | **64 KB** |
| qt=4, **both** starved | 64 406 µs | **6.5** | **24 KB** | **64 KB** |
| qt=8, real | 57 595 µs | 2.66 | 512 KB | 512 KB |
| qt=8, both starved | 56 564 µs | 2.9 | 24 KB | 64 KB |

**Making the entire K+V working set 88 KB — L1-resident several times over —
changes the runtime by 1–2 % and leaves the "L2 miss rate" at 6.5 %.** Therefore:

1. **block-major K is dead as a recommendation.** It cannot pay, because removing
   50× of the K footprint pays 0.4 %. Do not touch the transpose layout (which is
   good news: it also protects the paged kernel's page layout that already beats
   vLLM).
2. **the `l2_rqsts.demand_data_rd_miss` column in §3 is an artefact of this VM's
   vPMU, not a miss count.** It reports the same *absolute* count (5.7-5.8 M/call)
   for an 88 KB working set as for a 1 MB one, and scales only with kernel shape.
   Strike it; it was the weakest link in the round-1 argument and it was wrong.
   `l1d.replacement` behaves sanely (42 % → 16.4 % of loads from qt=4 to qt=8) but
   is not causal either.
3. the **`exe_activity.bound_on_loads`** counter is still meaningful (it counts
   cycles waiting on load-*related* resources, including L1-hit latency): 42.2 %
   → 33.0 % at D=64 with qt=8. It is a symptom, not a mechanism.

## R2.3 What the mechanism actually is: independent work per tile

With traffic eliminated as an explanation, the two head dims show *different*
reasons for the same lever — both measured with the kernel held fixed:

| shape | arm | µs | insn/call | IPC | in/pair | why it wins |
|---|---|---:|---:|---:|---:|---|
| 1024:16:128 fp32 | qt=4 | 73 947 | 312 M | 1.41 | 37.2 | — |
| | qt=8 | **65 084** | 276 M (−12 %) | 1.41 | 32.9 | **fewer instructions** (Q@K^T 17 per 8 FMAs → 29 per 16) |
| 2048:8:64 fp32 | reg qt=4 | 66 057 | 252 M | 1.31 | 15.0 | — |
| | reg qt=8 | **55 456** | 281 M (+12 %) | **1.57** | 16.8 | **IPC**, at an instruction *penalty* |
| 1024:16:128 bf16 | qt=4 | 68 071 | 272 M | 1.28 | 32.4 | — |
| | qt=8 | **57 663** | 318 M (+17 %) | **1.78** | 37.9 | **IPC** (+39 %) beats +17 % insn |
| 2048:8:64 bf16 | qt=4 | 50 268 | 209 M | 1.31 | 12.5 | — |
| | qt=8 | 51 392 | 257 M (+23 %) | 1.74 | 15.3 | **wash** (0.98×) |
| 1024:8:32 fp32 | qt=4 | 5 947 | — | 1.82 | 8.7 | — |
| | qt=8 | 6 356 | — | 1.89 | 8.8 | **0.94×**, nothing to hide |

Read this as: the lever is **independent work per tile** — more chains for the OOO
window and better amortisation of the per-`d` loads/addressing — and it pays
*only while the accumulator still fits*. It pays twice at D=128 (fewer
instructions, and IPC is stable), pays via IPC at D=64 fp32, and is cancelled by
spill instructions at D=64 bf16 (acc = 8×64 floats = 32 zmm) and at D=32 fp32
(nothing is stalling: `in/pair` is already 8.7 and load-bound is 25 %).

**So the per-box recommendation is not "qt=8" but a table:** fp32 qt=8 at D=64 and
D=128 (1.19× / 1.14×), keep qt=4 at D=32 (0.94×); bf16 qt=8 at D=128 (1.18×),
keep qt=4 at D=64 (0.98×). The repo's `_qt_for` hook (added for ARM this round)
is the right place for it.

## R2.4 Does it survive the engine's thread count? Yes, and it grows slightly

`taskset -c 0-3` (4 physical cores) / `-c 0-7` (4 cores + SMT), `OMP_NUM_THREADS`
matched, medians of 5:

| shape | qt | 1 thread | 4 threads | 8 threads (SMT) | vLLM @4T |
|---|---:|---:|---:|---:|---:|
| 2048:8:64 fp32 | 4 | 66 057 | 18 201 | 18 952 | — |
| | **8** | 57 595 | **15 225** | **14 859** | 7 920 |
| 1024:16:128 fp32 | 4 | 73 947 | 22 197 | 21 926 | — |
| | **8** | 65 084 | **16 749** | **16 667** | 7 717 |

* the `qt=8` win is **1.20× at 4 and 8 threads** (D=64) and **1.33×** (D=128) —
  it does not wash out under parallelism, as a pure-bandwidth effect would have.
* scaling is 3.8–3.9× on 4 physical cores for both arms (94–97 % efficiency), and
  SMT adds 0–2 %: this kernel has no spare SMT slots.
* **the gap to vLLM is unchanged at the engine's thread count: 1.92× (D=64) and
  2.17× (D=128) against vLLM's 7.92 / 7.72 ms.** `qt=8` is worth taking but it is
  not the answer to the whole deficit.

## R2.5 "With no traffic penalty, what would our time be?" — their IPC is real work

Our best D=128 arm (qt=8, single thread) is **65.1 ms at IPC 1.41 and `in/pair`
32.9**; vLLM is **31.6 ms at IPC 2.94 and `in/pair` 35.3**. We already retire
*fewer* instructions per pair than they do and are still 2.06× slower, so the
answer to "is their 2.87-2.94 IPC doing real work" is **yes, definitively** — and
since R2.2 removes misses as the explanation for our IPC, what is left is latency
inside the tile: a serial `reduce_max → _vexp → reduce_add → rescale` chain every
64 (or 32) keys, which drains the pipeline because a 64-key tile is ~5 000
instructions and the reorder buffer cannot span two tiles.

Note this does **not** contradict my §5 #4 bound; it *reinterprets* it. That bound
(≤1.18× at D=128, ≤1.06× at D=64) is on the **instruction count** of the per-tile
fixed work. A pipeline drain costs **cycles, not instructions**, and is outside
that bound. The ARM worker reaches the same place independently ("the residual
3.76 cycles/row-key are spill and latency inside the 23 % softmax block").

## R2.6 Cross-examination of the ARM findings

**Agreed, on different evidence.** The L2-sized tile is dead: they get 5.7 % from a
1024-key tile, I get an instruction-count bound of ≤1.18× / ≤1.06× that a bigger
tile cannot reach because `_vexp` scales with keys. Two different methods, same
verdict — treat it as settled.

**"85 % of the ARM gap is P@V on fp32 FMA instead of `bfmmla`" — I would not sign
it, and their own follow-up already contradicts it.** On x86 the same form of
argument is demonstrably invalid: P@V is **54 % of our instructions** at D=64
(§4) and we retire the **same instructions per row-key as vLLM** (19.07 vs 18.67),
yet lose 2.3×. So "phase X is N % of the instructions" does not imply "N % of the
gap". Their later statement — 100 % of the 2.36-cycle gap is *non*-matmul cycles —
is the consistent one; the 85 % framing is not. **Test they can run in ten
minutes:** fp32 vs bf16 at one fixed shape on ARM. If P@V's matmul were 85 % of the
gap, bf16 (which routes P@V through `bfmmla`) must be ~6× faster than fp32; on x86
the same A/B is **1.31×** (322.8 → 246.8 Mcyc at 1024:16:128) and on ARM bf16 vs
fp32 is a similar order of magnitude, not 6×.

**Where our mechanisms genuinely differ, and which I think is fundamental.** They
say "fp32 is load-bound, not pipe-bound — 24.2 vs 13.4 cycles/row-key at an
identical instruction set, because `qt=2` gives FMA-per-load 2 where vLLM gets 8".
I agree with the lever and disagree with the causal chain, on measurement:

* **FMA-per-load is not causal by itself.** It is a *structural* quantity that
  sets the port floor — `llvm-mca` on our D=128 Q@K^T loop says 6 zmm loads need
  8 cycles while 8 FMAs need 4, so the floor is **1.0 FMA/cycle** and we run 0.42
  (§4). That floor is real and independent of memory: *a loop whose loads all hit
  L1 still cannot exceed 1.0 FMA/cycle in this shape.* So of the two deficits —
  2.0 → 1.0 (structure) and 1.0 → 0.42 (stalls) — the first is the load:FMA
  ratio, exactly as they say.
* **But the second deficit is not bytes.** R2.2 shrinks the working set 50× and
  gains 0.4 %. So "load-bound" here means *load-latency-bound* (L1 hit latency on
  the operand feeds, plus the serial softmax chain), not *bandwidth*- or
  *footprint*-bound. That matters for what to build: their framing points at
  reuse-across-rows to cut traffic; mine says the traffic cut is a side effect and
  the thing to build is **more independent work per tile plus fewer operand loads
  per FMA**, which is why the same lever shows up as an *instruction* win at D=128
  and an *IPC* win at D=64.
* **`exe_activity.bound_on_loads` sits above both.** It counts cycles where the
  backend is blocked on a load-related resource — misses *and* L1 latency *and*
  port conflicts — so 48 % vs 15 % is a true symptom and a useless discriminator.
  Rank the three: **FMA-per-load (floor) > independent work per tile (how much of
  the floor you reach) > miss rate (not causal here at all)**.

## R2.7 The one experiment I would run with two more hours

**Hoist the softmax out of the inner loop at `qt=8`, keeping our layouts.**
Concretely: `_dense_rows_mem` at D=64 fp32 with `kt = 512` and a `probs` buffer
(the vLLM structure), measured against `reg@qt8` on the same four counters
(`instructions`, IPC, `in/pair`, `bound_on_loads`). This is the only untested
hypothesis that survives R2.2, and it is the one both profiles converge on.

*Prediction:* IPC rises from 1.57 to ≥2.0 while `in/pair` rises ≤15 % (logits and
probs become memory traffic), for a net 1.2–1.3×.

*Abandon criterion, stated before the run:* if IPC stays below ~1.8, or if
`in/pair` rises more than 15 %, the drain hypothesis is dead — the residual is
vLLM's 8–16-row × memory-logits structure that we cannot reach without a rewrite,
and the right move is to ship the `qt` table from R2.3 and stop optimising the
x86 dense kernel. **Do not start a third round on micro-tuning if that fires.**

## R2.8 Round-2 data

`scripts/prof/x86/data/r2_*.csv` — the `qt` A/B matrix, the K/V stride probes,
the 2×2 cells, the thread sweep and the vLLM thread reference. The 2×2 needs the
two patched builds; the patch is one line each:

```bash
cp -r cpu_attention/_mojo/src /tmp/src_mem && sed -i 's/qt \* head_dim > 384/qt * head_dim > 96/' /tmp/src_mem/dense.mojo
cp -r cpu_attention/_mojo/src /tmp/src_reg && sed -i 's/qt \* head_dim > 384/qt * head_dim > 1000000/' /tmp/src_reg/dense.mojo
mojo build /tmp/src_mem/_C.mojo -I /tmp/src_mem --emit shared-lib -o /tmp/C_mem.so -Xlinker -l:libgomp.so.1 -Xlinker -L$(dirname $(python -c 'import torch;print(torch.__file__)'))/lib
CPU_ATTENTION_LIB=/tmp/C_mem.so python scripts/prof/x86/run_one.py --variant ours --shape 2048:8:64 --dtype fp32 --qt 4
```

## R2.9 Confidence after round 2

| finding | confidence | change from round 1 |
|---|---|---|
| rows per tile is the lever; 1.18–1.19× measured with the kernel held fixed | **high** | was medium (confounded) → now isolated by 2×2 |
| the L1-accumulator kernel is *not* the thing to port; it is 6 % slower when both are legal | **high** | new |
| traffic / block-major K / the L2-miss story | **retracted** | two probes, 50× footprint change, 0.4 % effect |
| the residual is latency + instruction structure inside the tile, not bytes | **medium-high** | replaces the miss story; consistent with both profiles |
| `qt` recommendation is per (dtype, head_dim): fp32 8/8/4, bf16 4/8 at D=64/128 | **high** (measured), **medium** on generality | new |
| vLLM's 2.9 IPC is real work, not a counter artefact | **high** | new, from `in/pair` 32.9 vs 35.3 |
