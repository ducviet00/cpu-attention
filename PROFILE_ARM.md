# ARM profile + vLLM `csrc/cpu` source read

Box: `m9g.2xlarge` (Graviton 5 / Neoverse-V3, 8 cores, ASIMD + `bf16`, L1d 512 KiB,
**L2 2 MiB/core**, L3 48 MiB). Clock **3.28 GHz** (inferred below), single core,
`taskset -c 3`, `OMP_NUM_THREADS=1`.

Baseline commit for our tree: `c87b476`. vLLM sources on the box are byte-identical
to upstream `main` at `e9cc6244a` (md5-checked, six files), i.e. **post-#56045**, the
Sept 2026 Arm paged-attention refactor. `vllm_cpu_attn_bench.so` was rebuilt from
those sources (9436 `bfmmla`); the provisioner had omitted
`cpu_attn_neon_bfmmla.hpp`, so the reference build did not exist before this round.

**Bottom line (corrected — see §0): the moderator's tile claim is refuted as the
explanation of the 1.45x** (per-tile fixed cost is ~6% of runtime). But so was my own
first answer: **P@V is already `bfmmla` on both phases**, so the "P@V on the fp32
pipe" attribution below is **retracted**. With both phases on the matmul pipe, both
kernels need the same 4 cycles/row-key of `bfmmla` at D=64, and the entire 2.36-cycle
gap is **non-matmul overhead — IPC, not instruction mix**, which is exactly what the
x86 worker independently measured (19.07 vs 18.67 instructions/row-key, IPC 1.23 vs
2.87). The surviving ARM lever is **traffic per row-key ∝ 1/`qt`** (§10).

## 0. RETRACTION (added after the x86 worker's report)

**The claim "85% of the bf16 gap is P@V on fp32 FMA; only Q@K^T uses `bfmmla`" is
withdrawn.** I quoted the wrong function. The proof it is wrong:

| evidence | fact |
|---|---|
| `dense.mojo:1045` | `accv[rp*dcp+d] = _bfmmla(accv[rp*dcp+d], ap[rp], vv)` — inside `_dense_rows_bfmmla`'s P@V super-tile |
| `dense.mojo:809-826` (docstring) | "``_dense_rows`` **with `bfmmla` on both phases** (bf16 on NEON)" |
| `dense.mojo:819` | V is packed `[head_dim/2][seq_k/4][8]` "because P@V contracts over keys" |
| `dense.mojo:74` | `comptime BFMMLA = NEON_DOT` — true on this box (`+bf16`) |
| `dense.mojo:1410,1416` | dispatch: `_dense_rows_bfmmla[head_dim, qt, BKS, 8, 16]` / `[..., 4, 8]` |
| asm, `_dense_task[dtype=bf16,head_dim=64,qt=4]` | `bfmmla 80, bfdot 0` (the x86 worker counted the same 80) |

What I actually cited (`dense.mojo:723-730`, `acc[r] += probs[...] * vrow`) is in
**`_dense_rows_dot` (595-748)** — the `bfdot` fallback used when `BFMMLA` is false
(pre-`bf16` ARM, i.e. a build whose `--target-cpu` lacks `+bf16`). It is not
`_dense_rows_mem` either (396-593). The error was reading lines 600-800 and
attributing them to the next function that started after them.

Two consequences:

1. **My fit could not have caught it.** `cycles = b/2 + f/4 + s` with `s` as the
   residual has one free parameter per shape, so "residual 0.00" was guaranteed. The
   moderator is right that this is not evidence. I will not rest anything on a
   residual-with-free-parameter again; §4 is rewritten to use only measured totals
   and a recovered *shared* term.
2. **The x86 worker's cross-check is a second, independent refutation.** An
   instruction-mix asymmetry of that size cannot be the dominant term when the x86
   box has *no* such asymmetry and is **further** behind vLLM (2.25x at D=64 fp32 vs
   our 1.45x bf16). Both boxes point the same way: **same arithmetic, worse IPC.**

So: the moderator's tile claim is dead (§4.4), my P@V claim is dead (§0), and what
survives is the *fp32/q-t* line in §5.1, now measured properly in §10.

- **No PMU.** `perf_event_paranoid=4` confirmed; `sudo perf stat -e cycles true`
  returns user time only. This is a hard blocker for counter-based work.
- **callgrind is unusable for *both* kernels on this box.** `valgrind 3.26.0` dies
  with `Illegal instruction (core dumped)` inside the first kernel call — the VEX
  frontend cannot decode what the aarch64 Mojo/LLVM build emits. Symptom that
  exposed it: `--calls 21` and `--calls 1` both report **3.613 G Ir** (only torch
  import; the kernel is never reached). `--toggle-collect` cannot bind either,
  because `_C.so` exports no kernel symbol (only `PyInit__C`; the kernels are
  reached through function pointers). So there are **no exact Ir/D1/LL/branch
  counts** in this report, and I did not fabricate any. `llvm-mca` is not
  installable here (no package; `llvm-mca*` absent after `apt install linux-tools`).
- What the decomposition rests on instead:
  1. wall-clock cycles/row-key, **interleaved inside one process**, medians of 9
     (`scripts/prof/bench_shapes.py --isa neon --with-sdpa`, log kept as
     `bench_arm.log`);
  2. the core's measured `bfmmla` peak and latency, in-kernel
     (`cpu_attention/_mojo/tests/bfmmla_peak.mojo`);
  3. **static** instruction counts per key tile from
     `mojo build --emit asm --target-triple aarch64-unknown-linux-gnu
     --target-cpu neoverse-v3` (`scripts/prof/arm/asm_loop.py`);
  4. a cycle model that has no free parameter after the two above are fixed, and
     which reproduces **all four** bf16 measurements within 5-10% (§4.3).
- `perf`/PMU, callgrind and `llvm-mca` all failing is why this report leans on
  static counts + a validated model rather than counters. Anyone re-running this
  should either use a bare-metal box or accept the same ceiling.

### 1.1 Clock and the `bfmmla` ceiling (re-measured, confirms prior numbers)

`bfmmla_peak.mojo`, `taskset -c 3`:

| quantity | measured | implies |
|---|---|---|
| `bfmmla` dependent chain | 1.219 ns/instr | 4-cycle latency at 3.28 GHz |
| `bfmmla`, 8 independent chains | **0.1520 ns/instr** | **2.00/cycle**, 3.28 GHz |
| single-core ceiling | 210 GFLOP/s | 32 MACs/instr × 2/cycle × 3.28 G |

So the previously recorded "2.0/cycle, 0.152 ns" **stands**, and the clock is
3.28 GHz. Hereto every cycle number below is `ns/row-key × 3.28`.

## 2. Headline: ours vs vLLM vs SDPA (identical shapes to the x86 side)

bf16, causal, single core, µs, medians of 9, all three variants interleaved in one
process. `<1` in the ratio column means faster than the reference.

| shape | dtype | ours | vLLM | SDPA | ours/SDPA | vLLM/SDPA |
|---|---|---:|---:|---:|---:|---:|
| S=2048 H=8 D=64 | bf16 | 39693 | 27608 | 36842 | 1.08 | 0.75 |
| S=1024 H=16 D=128 | bf16 | 33833 | 21772 | 36577 | 0.92 | 0.60 |
| S=512 H=8 D=64 | bf16 | 2954 | 1818 | 4111 | 0.72 | 0.44 |
| S=128 H=8 D=64 | bf16 | 268 | 146 | 310 | 0.86 | 0.47 |
| S=2048 H=8 D=64 | fp32 | 123754 | 68765 | 104771 | 1.18 | 0.66 |
| S=1024 H=16 D=128 | fp32 | 121806 | 72912 | 96554 | 1.26 | 0.76 |
| S=512 H=8 D=64 | fp32 | 7954 | 4531 | 11196 | 0.71 | 0.40 |
| S=128 H=8 D=64 | fp32 | 540 | 326 | 777 | 0.69 | 0.42 |

Parity was checked before every timing: bf16 `max|d|` ≤ 1.6e-2 (rel ≤3.6e-3) for
both ours and vLLM against SDPA; fp32 ≤2.1e-6. Neither variant is silently wrong.

These are **1-thread** numbers, so ~4x the 8-thread figures in `AGENTS.md` §4b;
only the ratios are comparable across boxes. The two ratios the previous worker
quoted are reproduced: **bf16 D=64/S=2048 1.44x behind vLLM**, **fp32 1.80x**.

### 2.1 The same table in cycles per (query-row × key)

`row-keys = H·S·S/2`; cycles = µs / row-keys × 3280.

| shape | dtype | ours | vLLM | SDPA |
|---|---|---:|---:|---:|
| S=2048 D=64 | bf16 | **7.76** | **5.40** | 7.20 |
| S=1024 D=128 | bf16 | **13.2** | **8.5** | 14.3 |
| S=2048 D=64 | fp32 | **24.2** | **13.4** | 20.5 |
| S=1024 D=128 | fp32 | **47.6** | **28.5** | 37.7 |

The bf16 gaps are **2.36 cycles/row-key** (D=64) and **4.7** (D=128). Everything
below is an attempt to account for exactly those two numbers.

## 3. What our bfmmla kernel actually emits (static, per 16-key tile)

From `dense.mojo` structure (D=64, `qt=4`, `KT_TILE=4·LANES=16`, `NV=4`) plus the
asm histogram. The inner Q@K^T loop is genuinely tight — **16 `bfmmla` + 9 support
instructions** (`asm_loop.py`, loop at line 631898 of `arm.s`: `bfmmla=16 ldp=4
ldur=1 ldr=1 add=1 subs=1 b.ne=1`). Q@K^T is not our problem.

Per **row per tile**, the non-matmul work is:

| item | instructions | notes |
|---|---:|---|
| `reduce_max` over 16 lanes | ~4 | 3 `fmaxnm` + `fmaxv` |
| scalar `m_new`, compare, branch, `_vexp_f(alpha)` | ~10 | serial on the max chain |
| `_exp2_fast` over 16 lanes | ~44 | 4 × (deg-4 poly + magic round + exponent insert) |
| `reduce_add` + `row_sum` update | ~6 | |
| **accumulator rescale `acc[r] *= alpha`** | **16** | head_dim/4, *unconditional* |
| probs store | 4 | 16 floats = 4 `vst` |
| **P@V, per key** | **16 `fma`** + V load/cvt ÷ 4 rows | head_dim/4 per row |

Two structural facts that decide everything:

1. **Both phases are `bfmmla`** (this bullet previously said P@V was fp32; withdrawn —
   see §0). `dense.mojo:1045`. Q@K^T and P@V are each 16 MACs/instruction, so at
   D=64 the kernel needs 8 `bfmmla` per row-key and 4 cycles of matmul at the 2/cycle
   peak; everything else it does is overhead (§4.4.1). The dead giveaway that this
   section was mis-written: `_dense_rows_dot`, the function I quoted, emits **zero**
   `bfmmla` by construction.
2. **The mask is boundary-only.** `dense.mojo:729` guards it with
   `if boundary:`, and `boundary` is true only on the last ~1-2 tiles per row
   block. So the per-lane `iota`/compare/`select` runs on <2% of tiles.

## 4. The intercept decomposition (the deliverable)

### 4.1 The model

On this core the measured peak is 2 `bfmmla`/cycle and the four 128-bit FP pipes
sustain ~4 FP ops/cycle. Both kernels' non-matmul work lands on those same pipes,
so, with `b` = `bfmmla`/row-key, `f` = fp32 FMA/row-key, `s` = softmax:

```
cycles/row-key  ≈  b/2  +  f/4  +  s
```

`b` and `f` are fixed by the algorithm, not fitted:

| kernel | Q@K^T | P@V | b | f |
|---|---|---|---|---|
| ours bf16 | `bfmmla`, D/16 | **`bfmmla`, D/16** | **D/8** | 0 |
| vLLM bf16 | `bfmmla`, D/16 | **`bfmmla`, D/16** | **D/8** | 0 |

Both kernels are already on the matmul pipe for **both** phases (§0). So the two
matmul terms are *equal*, and every cycle of the gap is by construction
non-`bfmmla`: `b/2` is 4 cycles/row-key for both at D=64 and 8 at D=128. That is the
one useful thing about this framing — it makes the gap an *overhead* measurement.

### 4.2 The overhead, by subtraction

| case | b/2 (matmul floor) | measured | ⇒ overhead |
|---|---:|---:|---:|
| ours, D=64 | 4.0 | 7.76 | **3.76** |
| ours, D=128 | 8.0 | 13.2 | **5.2** |
| vLLM, D=64 | 4.0 | 5.40 | **1.40** |
| vLLM, D=128 | 8.0 | 8.5 | **0.5** |

No fitted parameter: `b` is fixed by the algorithm (D/8, verified against the source
loop counts — Q@K^T 4/row-key + P@V 4/row-key at D=64) and the matmul floor is the
measured 2/cycle peak (§1.1). The result: **we spend 2.7x more non-matmul cycles per
row-key than vLLM at D=64, and ~10x at D=128.**

One model, no tuning, four measurements within 5-10%. The softmax term is
0.5-1.8 cycles/row-key in both kernels.

### 4.3 The gap, attributed

| case | gap (cycles/row-key) | matmul | non-matmul overhead (ours − vLLM) |
|---|---:|---:|---|
| D=64 bf16 | 2.36 | 0.00 (identical 4.0) | **2.36** = 3.76 − 1.40 |
| D=128 bf16 | 4.70 | 0.00 (identical 8.0) | **4.70** = 5.2 − 0.5 |

**100% of the bf16 deficit is overhead cycles, at both head dims.** This is the same
conclusion the x86 worker reached from counters on the other box — 19.07 vs 18.67
instructions per row-key (a wash) but IPC 1.23 vs 2.87 — and it is why instruction
*mix* (my retracted §7.1) could not have been the answer. What the overhead is made
of is §4.5 and §10; the short version is K/V **traffic per row-key**, which scales as
1/`qt`, and `qt=2→4` on this box is worth 26% (§10).

### 4.4.1 Instruction accounting for the overhead (source-derived, no PMU)

Per row-key, D=64, qt=4 (super-tile `ks=64`, softmax step `sub=16`, `RP=2`):

| phase | bfmmla/row-key | other instructions/row-key |
|---|---:|---|
| Q@K^T | 4 | ~2.5 (K panels 128 + Q panels 32 per 16-key step) |
| softmax (max, exp, sum, bf16 prob pack) | 0 | ~3.75 |
| P@V | 4 | ~2.5 (V 512 + A panels 128 per super-tile) + acc spill 1 |
| prob rescale round-trip (bf16→f32→mul→bf16) | 0 | ~1 |
| **total** | **8** | **~10** |

So ~18 instructions/row-key of which 8 are `bfmmla` → 4 cycles of matmul plus ~10
instructions of overhead needing 3.76 cycles ⇒ overhead IPC ≈ 2.7. Meanwhile the asm
shows **21-26% of the super-tile loop body is `[sp, ...]` stack traffic**, i.e. spill
and remat is a real fraction of that 10. vLLM's equivalent is ~1.40 cycles at D=64.

### 4.4.2 Does the operand-load port bind before the 2/cycle `bfmmla` ceiling?

No — but only just, and it is the wrong question. Ours is **0.63 loads per
`bfmmla`** (≈2.5 loads per row-key per phase, 8-byte/16-byte `vld1q` operand pairs),
and the core sustains 2 loads/cycle against 2 `bfmmla`/cycle, so the *port* floor is
`max(8/2, 5/2) = 4` cycles — the matmul floor, i.e. loads do not bind issue. vLLM is
0.5-0.63 loads/`bfmmla` for the same reason (`gemm_8x8`: 4 A + 4 B loads per 16
`bfmmla`). **The binding constraint is not port throughput but bytes moved per
row-key**, because their K/V panel is reused across `MaxRows = 8` (and
`MaxQHeadNumPerIteration = 16`) rows while ours is reused across `RP*2 = 4`: **2-4x
more KV bytes per row-key at equal instruction count.** A register-blocked operand
(reuse one P panel across dim chunks — already done: `ap[rp]` is loaded once per
`kg` and used by all `dcp` dims; reuse V across row pairs — already done) would buy
nothing further; the missing reuse is across *rows*, which is `qt`.

### 4.4 Splitting our softmax term: per-tile fixed (a) vs per-key (b)

Using §3's counts, per row, and `cycles ≈ instructions / 4`:

| | instructions per row-tile | per row-key (/16) | cycles/row-key |
|---|---:|---:|---:|
| **(a) per-tile fixed**: scalar max/branch/alpha (10) + acc rescale (16) + probs store (4) | 30 | 1.9 | **~0.47** |
| **(b) per-key**: exp (44) + `reduce_max` (4) + `reduce_add` (6) | 54 | 3.4 | **~0.85** |
| remainder of `s` (dependency stalls on the serial max chain, tile re-entry) | | | ~0.44 |

**(a) = ~0.47 cycles/row-key = 6.1% of 7.76.** A 256-key tile amortises (a) 16x
away, i.e. **a 5.7% speedup at best**, and it cannot touch the 2.0-cycle P@V term
that makes up 85% of the gap. At S=2048 the tile count is `S/16 = 128` per row
block, so (a) is paid 128 × 30 = 3840 instructions per row — against 7.76 × 2048 =
15,900 cycles for the same row. It is a sixth of the budget, and only a sixth of
*that* is removable by enlarging the tile.

**Verdict on the moderator's claim: the mechanism is real and correctly identified
(vLLM does use 1024-key tiles, we use 16), but the magnitude is 5.7%, not 45%. It
is a 6% item, ranked 5th below, not the headline.** My ranking disagrees with the
moderator's and §4.3 is why: the term a bigger tile would remove is smaller than
the accumulator rescale it must still do, and 4x smaller than the P@V pipe choice.

A 256-512-key tile would additionally *hurt* the current kernel: with P@V in fp32
and `probs` materialised per tile, a 256-key tile is a 4 KiB (D=64) logits/prob
buffer per row-pair — 16 row-pairs at `qt=8` is 64 KiB, past L1 — so the fp32 P@V
would pay a fresh L1 store/reload per key that it currently gets from a 64-float
stack slot. **Do the P@V change first; a bigger tile only becomes attractive once
P@V reads bf16 probabilities in bfmmla order (§7.1), which is exactly the layout
that makes a big tile cheap.** These two are not independent, and the ordering
matters.

## 5. The other three claims, with numbers

| claim | what our kernel actually does | cost | verdict |
|---|---|---|---|
| **mask by range** (`apply_mask`, `cpu_attn_impl.hpp:1376`) | already boundary-only (`dense.mojo:729`), so it runs on <2% of tiles | **<0.05 cycles/row-key (<1%)** | refuted |
| **−8 rescale threshold** (`:1479`, and the max is *not* updated when it does not fire) | `acc[r] *= alpha` is unconditional, 16 `fmul` per row-tile = 1 instr/row-key; making it conditional saves at most the rescale **and** the max update | **~0.25 cycles/row-key (~3%)** | real but small; cheap to take |
| **probs alias the logits buffer** (`:1329-1331`) | separate `probs` stack buffer, 4 `vst` + element-indexed FMA (LLVM folds the reload into `fmla` by element) | **~0.25 cycles/row-key (~3%)**; and the earlier L1-staged-logits experiment measured *exactly* neutral, which is consistent | real but small; subsumed by §7.1 |

So the four "cheap things" total ~6%, the tile adds ~6% — together ~12% against a
45% gap. They are worth taking, but none of them is the story.

### 5.1 fp32 is a different failure (load-bound, not pipe-bound)

fp32 D=64: ours **24.2** vs vLLM **13.4** cycles/row-key (1.80x). Both use the same
4-lane FMA (vLLM's fp32 NEON is not `bfmmla`), so instruction *sets* are identical;
the model predicts `f = D/2 = 32` → 8 cycles if FMA-bound. We are **3x** off the
FMA bound, which means fp32 is not compute-bound at all.

Cause, from `_kernel.py:323-326`: on ARM with `_lanes() < 16` the fp32 path takes
`qt = 2`. Each V vector load therefore feeds only **2** FMAs (FMA-per-load ratio 2),
where vLLM's fp32 path uses `MaxQHeadNumPerIteration = 8` (ratio 8). That is the
`AGENTS.md` §9.10 `qt` = "FMA-per-load ratio" argument, except on AVX-512 it was
falsified because 16-lane vectors gave ratio 4*qt; on NEON with 4-lane vectors the
ratio is 4x worse and the load ports bind.

Preliminary, **not** drift-controlled (ran without `--isa neon`, so vLLM was on its
portable VEC path; the box also drifted ~1.6x during that window), so it is evidence
of direction only: fp32 `CPU_ATTENTION_QT` 2/4/8 → 16785 / 12740 / 17697 µs. `qt=4`
1.32x better than `qt=2`, and `qt=8` worse — which is the register wall
(`head_dim/4 = 16` accumulator registers per row × 8 rows = 128 regs). This needs a
proper interleaved re-run before it is trusted.

## 6. vLLM `csrc/cpu`: what they do (file:line, verified)

### 6.1 The six things that matter

1. **P@V is `bfmmla`.** `cpu_attn_neon_bfmmla.hpp:304-313` — the `else` branch of
   the phase test calls `BfmmlaGemm::gemm(...)` for `PV`, with `b_n_group_stride`
   built for the V layout. Both phases ride the matmul pipe at 16 MACs/instr. This
   is the single biggest divergence from us and the bulk of the 1.45x (§4.3).
2. **Probabilities are bf16, packed in bfmmla reduction order.**
   `prob_buffer_t = c10::BFloat16` (`:330`) and
   `ProbabilityTokenStore::store_probabilities` (`:371-390`) with
   `TokenStride = 2`, `row_in_pair = row & 1`, stores at
   `probability + row_in_pair*(KTile - row_stride)` and `+2/4/6*KTile`, converting
   with `vcvt_bf16_f32`. That is not merely "probs alias the logits buffer": it is
   a **layout deliberately built so `bfmmla` consumes it as the `a` operand with no
   repack**, at half the bytes of fp32. This is what makes (1) cheap.
3. **A big L2-sized key tile.** `calcu_default_tile_size`
   (`cpu_attn_impl.hpp:962-993`):
   `tile = cache / (head_dim*(q_buf + 2*elem + out_buf) + max_q*logits)`, capped at
   `MaxQTileIterNum(128) * max_q_head_num_per_iter` (`:429`, `:984`), rounded to 32.
   With **L2 = 2 MiB** here the raw value is 2621 (bf16 D=64) / 1337 (bf16 D=128)
   and 1986 (fp32 D=64), so every shape lands on **the cap, i.e. ≥1024 keys** —
   the claim holds a fortiori on this box. Our tile is 16 (D=64) / 8 (D=128).
   Worth 5.7% to us (§4.4).
4. **Re-scale only when the max moves by >8** (`:1475-1483`):
   `need_rescale = (init_max - new_max) < -8.0`, and when it is false
   `new_max_val = init_max_val` — the running max is deliberately *not* updated
   (FA4 semantics, cited in the comment). Rescale of sum+output then happens inside
   `if (need_rescale)` at `:1540-1555`. ~3% to us.
5. **Mask by range** (`:1376-1420`): per token/head it writes `-inf` into
   `[0, left_invalid)` and `[right_invalid, n)` with scalar loops; `apply_mask` is
   called every tile (`:1297`) but both counts are 0 for interior tiles. <1% to us.
6. **KV splits + a reduction pass + atomic work stealing.** `kv_len_per_thread =
   total_kv_len / thread_num` (`:539`), a q-tile longer than the budget is split
   (`:584-600`, `min_split_kv_len`, `local_split_id`), and a `PartitionReduction`
   work item (`:488-497`) combines partials. The combine (`:2160-2200`) is a full
   pass over `head_dim` per split per head: `exp(curr_max - final_max)` rescale,
   `non_scale + scale*rescale`, accumulate. Non-trivial, and only paid above a
   threshold. Their main loop also takes work from a shared atomic counter
   (`metadata.acquire_counter()`, `:1742-1756`) with a spin barrier before the
   reduction phase — that is what `cac81b6ed` (#42666) bought.

### 6.2 Answers to the specific questions

1. **Skip the rescale when the max doesn't move?** Yes, and more aggressively than
   us: they skip at a **−8 threshold** and then *keep the stale max* (`:1479-1483`).
   We rescale on any increase, unconditionally. They track `m`/`l` per row in
   `max_buffer`/`sum_buffer` exactly as we do; no additional trick.
2. **Softmax pass structure?** Same passes as ours — one pass for the max
   (`:1462-1471`), one fused pass for exp+sum that also writes the probabilities
   (`:1489-1520`) — then P@V reads the buffer. **Not** a single fused pass. The
   difference is *only* that their passes run over ~1024 keys per tile instead of
   16. Their block is `vec_num = kv_tile_token_num/16` iterations of
   `FP32Vec16`, so the per-key instruction count is comparable to ours; what they
   amortise is the per-tile scalar part — which §4.4 prices at 6%.
3. **KV splits?** Yes (see 6.1.6). Combine cost = `split_num × q_heads ×
   head_dim/16` vector ops plus one `exp` per (head, split) — O(head_dim), not
   O(S), so it is cheap; the reason to avoid it is the extra scratchpad and the
   barrier, not the arithmetic.
4. **`ProbabilityTokenStore`?** It buys a **bf16, row-pair-interleaved probability
   layout** (`TokenStride=2`, `:371-390`) that is directly the `bfmmla` `a` operand.
   Yes, the VEC/NEON path uses a non-trivial stride — that is the whole point. For
   us this is the prerequisite for §7.1, not an independent optimisation.
5. **Prefetch / unrolling / ILP?** `#pragma GCC unroll 4` on the k loop in both
   `gemm_4x16` (`:151`) and `gemm_8x8` (`:232`); the 8x8 kernel keeps 4 row-pair
   accumulators × 4 N-groups = 16 `bfmmla` in flight against 4 A + 4 B loads. **No
   software prefetch, no streaming stores anywhere in the file.** Their ILP comes
   from the 2x2-tile register blocking, nothing more exotic.
6. **Why 1.49-1.86 `bfmmla`/cycle vs our 1.02-1.30?** Not a scheduling trick:
   **they issue ~2x more `bfmmla` per row-key because P@V is also `bfmmla`**, and
   `gemm_8x8` is fully dense (8 rows × 16 cols per 2 calls, 16 MACs per
   instruction — verified against the operand strides, no waste). Our 1.02-1.30 is
   Q@K^T *only*; our P@V never enters that count. Per §4.3 the two kernels' Q@K^T
   phases are at comparable efficiency, which is why the ratio difference looks
   like a matmul-issue gap but is really a pipe-assignment gap.

### 6.3 Git history: every perf-signalling commit in `csrc/cpu`

Commit bodies are sign-offs only (vLLM squash-merges), so the diff is the message.
The attention-relevant ones:

| commit | PR | technique | do we do it? |
|---|---|---|---|
| `92b40f5a1` | #56045 | **Arm paged-attention refactor, 2026-09**: rewrote `cpu_attn_neon_bfmmla.hpp` (847 lines) into `BfmmlaGemm` with explicit 4x16 / 8x8 micro-tiles, bf16 packed probs, V repacked for a bfmmla P@V | **No** — this is the commit that contains our deficit (§4.3) |
| `1363e3d6d` | #32263 | first NEON bfmmla paged attention (682 lines) | partially (Q@K^T only) |
| `730bd3537` | #29193 | NEON QK/PV GEMMs, `cpu_attn_neon.hpp` | yes |
| `cac81b6ed` | #42666 | thread-utilisation: atomic work stealing + span packing | no (we fork-join round-robin) |
| `2f314bc5e` | #38112 | faster exp for low precision (`fast_exp_f16`) | yes — `_exp2_fast` already matches |
| `9843e332d` | #30068 | Arm Optimized Routines vectorized exp | equivalent to ours |
| `4dffc5e04` | #32161 | split dispatch by head_dim alignment (VEC16 for D%32≠0) | n/a (we do 32/64/128) |
| `26eb87204` | #45844 | fix split-KV scratchpad sizing | n/a (no split path) |

Note `2f314bc5e`/`9843e332d`: their transcendental is **already matched** by our
`_exp2_fast`, so exp is settled and must not be re-opened (it is worth ~1 cycle of
the 7.76 and is already near-minimal: `_exp2_fast` ≈ 11 instructions per 4 lanes).

## 7. Ranked recommendations

Payoff is in cycles/row-key at the two bf16 protocol shapes, from §4.3/§4.4, not
extrapolated from a different box.

### 7.1 ~~P@V on `bfmmla`~~ — **RETRACTED (§0). We already do this.** (was rank 1)

- **Evidence**: §4.3 attributes 2.00 of the 2.36-cycle gap (D=64) and 4.00 of 4.70
  (D=128) to P@V's pipe. vLLM does exactly this (`cpu_attn_neon_bfmmla.hpp:304-313`
  + `prob_buffer_t` bf16 at `:330` + packed store at `:371-390`).
- **Mechanism**: P@V moves from `head_dim/4` fp32 FMA (4 MACs/instr, FMA pipes) to
  `head_dim/16` `bfmmla` (16 MACs/instr, matmul pipe) — 4x fewer instructions and
  a different port. Predicted: D=64 7.76 → 5.8 cycles, D=128 13.2 → 9.8, i.e. at or
  just behind vLLM's 5.40 / 8.5.
- **Effort**: medium. The V packing must become position-major in 4-key groups
  (`V_TOKENS_PER_ROW_BLOCK = 4` is the analogue) and probabilities must be written
  bf16 in that same order — which *subsumes* §7.4 and §7.5.
- **Risk**: bf16 probabilities lose the bit-exactness against `csrc` that x86
  keeps. Target the ARM standard already accepted in `AGENTS.md` §11.4: ≤1e-3 vs
  the fp32 reference, matching `csrc`'s own ARM `prob_buffer_t`.
- **Falsified if**: a prototype on 2 rows × 4 keys measures <10% on
  `tests/bench_dense.py`. Then the pipe model in §4.1 is wrong and the gap is
  latency/scheduling, and the next thing to look at is the serial max chain.

### 7.2 fp32 NEON: L1-tiled accumulator, raise `qt` — up to 1.5-1.8x on fp32 rows (rank 2)

- **Evidence**: §5.1. 24.2 vs 13.4 cycles/row-key at D=64 with an *identical*
  instruction set, 3x off the FMA bound, and `qt=2` on the ARM fp32 path
  (`_kernel.py:323-326`) giving an FMA-per-load ratio of 2 vs vLLM's 8.
- **Mechanism**: not "raise `qt`" — `head_dim/4 = 16` registers per row makes
  `qt=8` spill (the preliminary sweep shows `qt=8` worse). The fix is
  `_dense_rows_mem` with a head-dim tile (`dc = 2*LANES`) plus `qt=8`, keeping the
  accumulator in L1 and amortising each K/V load over 8 rows — the structure
  `AGENTS.md` §4b item 2 already prescribes for ARM, which the fp32 path does not
  currently get.
- **Effort**: medium. **Risk**: low (no numerics change).
- **Falsified if**: a drift-controlled interleaved A/B of `qt=2` vs a tiled `qt=8`
  fp32 kernel is within 10%.

### 7.3 Conditional rescale at a −8 threshold — **~3%** (rank 3)

`acc[r] *= alpha` is unconditional; 16 `fmul` per row-tile = 1 instr/row-key, and
skipping also skips the max update. Trivial effort, low numerics risk (paste the
FA4 threshold; note it *changes results* slightly, so the parity test must be
re-baselined, not merely re-run).

### 7.4 A larger key tile — **≤5.7%** (rank 4, NOT the 1.45x)

- **Evidence**: §4.4 — (a) is 0.47 of 7.76 cycles/row-key.
- **Do not do this first.** With fp32 P@V a 256-key tile turns a 64-float stack
  `probs` into a 4-64 KiB L1 buffer reloaded once per key, which is why the
  earlier "logits staged through L1" variant was neutral and why this is ranked
  below the P@V change. After §7.1 (bf16 packed probs) it becomes cheap and is
  worth revisiting **then**, as a second-order 5% on top.
- **Design, if taken after 7.1**: tile 256 keys (not 1024 — at D=64 that is 4 row
  pairs × 256 × 2 B = 4 KiB of bf16 probs, comfortably L1-resident); logits stay
  fp32 in a per-row-pair scratch (`qt/2 × 256 × 4 B = 2 KiB`); the softmax becomes
  one max pass, one exp+sum+store-bf16 pass (`ProbabilityTokenStore`-style, packed
  in bfmmla order), then P@V consumes it directly. The 2x2 `bfmmla` tiles do not
  change: they already produce 2 keys per key-pair column, which is why the prob
  buffer's key stride is 2 in their layout. Mask-by-range on top of that needs
  `[0, left_invalid)` / `[right_invalid, n)` per row, which is *easier* on a big
  tile than the current per-lane `select` (and would then also cover the
  boundary tile).

### 7.5 probs aliasing the logits buffer — **~3%**, subsumed by 7.1

Only worth doing as part of §7.1, where the prob buffer must be rewritten in
bf16/packed form anyway.

## 8. Do not retry

1. **callgrind / valgrind on this box** for either kernel — `Illegal instruction`
   inside the first kernel call; it counts 3.6 G of torch import and nothing else.
   `--toggle-collect` cannot help (`_C.so` exports no kernel symbol). Don't
   re-derive this; use static asm + wall clock, or a bare-metal box.
2. **Per-lane mask optimisation** — already boundary-only; <1%.
3. **L1-staged logits / prob-buffer aliasing as a standalone change** — measured
   exactly neutral previously, and §5 prices it at 3%; it only pays as part of the
   bf16-prob rewrite.
4. **`std.math.exp` / the transcendental** — already at `_exp2_fast` ≈ 11 instr/4
   lanes; vLLM's `fast_exp`/`fast_exp_f16` (`cpu_arch_macros.h:9-193`) is the same
   family. Settled, per `AGENTS.md` §4c.
5. **A big key tile on its own** — ≤5.7%, and it regresses with fp32 P@V. Only after 7.1.
6. **Raising `qt` alone on fp32** — register wall at `qt>4`. It needs the L1
   accumulator (§7.2), not a knob.
7. **A private thread pool / different dispatch** — `GOMP_parallel` on torch's pool
   is already the answer (`AGENTS.md` §9.3); nothing here changes that. All numbers
   in this report are single-threaded on purpose.

## 9. The one experiment to run next

A **protocol-driven A/B of a P@V-`bfmmla` prototype** (2 rows × 4 keys, bf16
probabilities in `[head_dim/4][key/4][8]` order), interleaved in one process
against the current kernel, at the four protocol shapes, bf16, `taskset -c 3`,
medians of ≥9. It decides §7.1 and it is the only change whose expected payoff
(1.30-1.35x) is comparable to the size of the gap it claims to explain — so it is
also the cheapest way to falsify the pipe model in §4.1. Second, in the same
harness: dump one 4×16 tile's logits and compare each `bfmmla`'s four output lanes
against a scalar reference, because the source-level lane mapping could not be
reconciled with the per-tile instruction count from reading alone (the count
implies a 2x redundancy that the measured `bfmmla`/cycle does **not** support —
§4.1's fit says Q@K^T is densely packed at 16 MACs/instr, so the source reading is
what is wrong). `cpu_attention/_mojo/tests/bfmmla_layout.mojo` already exists for
this.

## 10. Round 2: `qt` / traffic per row-key (replaces rank 1)

`qt` query rows per work item means each K/V byte is reused `qt` times, so **traffic
per row-key ∝ 1/qt**. `_kernel._QT_ENV` is read at call time, so every `qt` is timed
**inside one process, interleaved** (`scripts/prof/arm/qt_sweep.py`), and every `qt`
is asserted against `qt=4` output first, so a fast-but-wrong variant cannot be
reported. Single core, `taskset -c 3`, medians of 5.

| shape | dtype | qt=2 | qt=4 | qt=6 | qt=8 |
|---|---|---:|---:|---:|---:|
| S=1024 D=64 | bf16 | 0.75x | **1.00x** | 0.90x | 0.94x |
| S=2048 D=64 | bf16 | 0.74x | **1.00x** | 0.89x | 0.94x |
| S=4096 D=64 | bf16 | 0.77x | **1.00x** | 0.90x | 0.94x |
| S=1024 D=64 | fp32 | 0.80x | 1.00x | **1.09x** | 0.78x |
| S=2048 D=64 | fp32 | 0.79x | 1.00x | **1.10x** | 0.76x |
| S=4096 D=64 | fp32 | 0.77x | 1.00x | **1.11x** | 0.76x |

Three things, and they are stable across a 4x range of S (so this is traffic, not
fixed cost, and not a short-S artefact):

1. **The mechanism is confirmed**: halving rows (`qt=2`) costs **23-26% in both
   dtypes**. That is the same lever the x86 worker found (`qt=8` worth 1.20-1.34x),
   pointing the same way.
2. **`qt=8` does not transplant to ARM.** bf16 `qt=8` is 6% *worse* at every S, and
   fp32 `qt=8` is 24% worse. Cause: `qt>4` switches the bfmmla path to
   `_dense_rows_bfmmla[..., 4, 8]` (`dcp=4, sub=8`) — half the key tile and double the
   row pairs — so `RP*4` live accumulators hit the 32-register file. For fp32,
   `head_dim/4 = 16` registers per row × 8 rows = 128. This is the register wall, not
   a traffic effect.
3. **fp32 `qt=6` is a free 1.10x** at every long S (parity-verified), and it is a
   pure dispatch change (`_kernel.py:323-329`), no kernel edit.

So my earlier "fp32 is load-bound" (§5.1) is **the same phenomenon the x86 box
measured** — load traffic per row-key — and now has a number: 1/qt, with 26% banked
between qt=2 and qt=4 and a further ~10% at qt=6 for fp32. It does **not** reach the
x86 box's 1.2-1.34x because our `qt>4` config change throws away more than it gains.

### 10.1 Who is right about what

| claim | verdict |
|---|---|
| moderator: L2-sized 1024-key tile is *the* cause | **refuted** — ≤5.7% (§4.4). x86 worker independently bounded it ≤1.18x even driven to zero. Mechanism real, magnitude not. |
| moderator: mask-by-range / probs aliasing / −8 rescale are the story | **refuted** — <1%, ~3%, ~3% (§5) |
| me: P@V on fp32 pipe is 85% of the gap | **retracted** — P@V is `bfmmla` (§0) |
| me: the gap is 100% *overhead* cycles, not matmul | **stands** — both kernels need 4.0 cycles/row-key of `bfmmla` at D=64 (§4.2) |
| x86 worker: same instructions, half the IPC; load-bound; `qt` is the lever | **stands, and reconciles**: on ARM the same holds, with the twist that ARM bf16 is already at its `qt` optimum and the lever only pays ~10% (fp32 qt=6) |
| "instruction mix explains ARM" (either of us, at different times) | **refuted on both boxes** |

The ARM and x86 deficits are one phenomenon seen through two dtypes: **we move more
KV bytes per row-key and stall on them.** x86 has the headroom to fix it with `qt=8`;
ARM bf16 does not, because our tile config is coupled to `qt`.

### 10.2 Highest-expected-value experiment (2 hours, one box)

**Decouple `qt` from the tile config, then measure `qt=8` with `sub=16/dcp=8`.**
Concretely: in `_dense_task`'s bfmmla arm, pick `(dcp, sub)` by `head_dim` only
(8,16 for D<=64) instead of by `qt`, and give `_dense_rows_bfmmla` `RP=4` row pairs
with the accumulator spilled deliberately to an L1 tile (`_dense_rows_mem`-style,
`dc = 2*LANES`) rather than to registers. Rationale: `qt=2→4` banked 26% by doubling
reuse; `qt>4` currently *loses* 6% purely because of the config flip, so recovering
that 6% plus the next increment of reuse is worth an estimated **1.1-1.2x**, and it
composes with the fp32 `qt=6` dispatch change that is already worth 1.10x.

**Abandon it if** the L1-tiled `RP=4` variant is not at least 5% faster than the
current `qt=4` at S=4096 D=64 bf16 in an `tests/ab_two_trees.py` run (interleaved,
alternating rounds). If it is not, the remaining 3.76 cycles of ARM overhead are
latency/spill inside the softmax block rather than traffic, and the next target is the
`[sp, ...]` traffic (21-26% of the super-tile loop body) rather than `qt`.
