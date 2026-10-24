# ARM (Graviton 5, m9g.2xlarge, Neoverse-V3) dense kernel

## Round 8 — `m_r` is *correct*; the bounded base fails on a per-row defect

Round 8's qk+softmax fusion (exp on the `bfmmla` accumulator registers, store
only bf16 P, `pv` untouched) needs an exponent base known before the exp, so it
is the bounded-base path again. Rather than re-derive the round-6 failure I
instrumented it: a one-shot `fprintf` in `qk` printing `c`, `max|k|`, `||q_0||`,
`m_0`, `m_1`, `nrows`. First run:

```
DBG c=0.18034 kmax=3.92188 q0=7.75614 mb0=5.48560 mb1=6.37037 n=8
DBG c=0.18034 kmax=3.92188 q0=7.52134 mb0=5.31954 mb1=6.46604 n=8
```

`max|k|` = 3.92 for 128x64 random bf16, `||q_r||` = 7.5-8.2, `m_r` = 5.3-6.5 and
monotone across the row pair -- **all correct**. So the round-6 hypothesis
(`max|k|` reaching `qk` as 0 or NaN, i.e. the slab-tail store/read disagreeing)
is **falsified**, and the bound itself is fine.

The failing set localises the defect elsewhere, and precisely: **every case with
more than one query row fails, every single-row (`Sq = 1`) case passes.**

```
[FAIL] bf16 B=1 Sq=128 Sk=128 H=8/8 D=64 causal=1: 3.164e-01 (tol 3e-02)
[FAIL] bf16 B=1 Sq=128 Sk=128 H=8/8 D=64 causal=0: 2.246e-01
[FAIL] bf16 B=1 Sq=64  Sk=96  H=8/8 D=32 causal=1: 5.234e-01
[FAIL] bf16 B=1 Sq=17  Sk=301 H=4/4 D=64 causal=0: 1.254e-01
```

`Sq = 17` (rq 16, two chunks) fails, `Sq = 1` passes, and both masks fail --
so it is the per-row state, not the mask and not the bound. `m_r` for rows 0 and
1 prints correctly, which leaves the *other* per-row scratch, `psum[r]` (the
per-tile probability sums that `softmax` flushes into `l`), or the row>0 path of
the bf16 P store. That is the single remaining hypothesis; the instrumented
build is reproducible in three edits (64-byte slab tail + `kmax_of`, the `m_r`
prologue, the in-place exp epilogue) plus the one-line `fprintf`.

Reverted: a kernel failing 24 of 65 parity checks cannot ship. The shipped
binary is unchanged and re-verified below.

## Round 7 — fp32 hi/lo bf16 `bfmmla`: measured at 1.2%, reverted

Implemented as a compile-time `DOT` branch of the fp32 policy (`DenseF32<HD, KTC,
DOT>`, the scalar path kept instantiable behind `CPU_ATTENTION_F32FMA=1`): K
packed as two bf16 planes (`hi = bf16(x)`, `lo = bf16(x-hi)`) in the same
position-paired 64-key-tile form the bf16 policy uses, Q split from the caller's
fp32 tensor into hi/lo operand panels once per tile, and the Q@K^T panel taking
three products (`hi*hi + hi*lo + lo*hi`, `lo*lo` dropped at 2^-16). Two bf16
planes are exactly one fp32 plane, so `slab_bytes`, the V offset, the softmax and
`pv` are all unchanged.

* **Parity: `test_dense` 65/65** — the hi/lo rounding is not the problem.
* **Speed: 2048:8:64 fp32 9177.0 -> 9064.1 us, 0.988x (1.2%).** Alternating-
  process protocol, min of 5 rounds, one process per sample.

**Why it is not the 1.1-1.25x of the round-2 costing.** That costing assumed the
instruction cut (23.5 -> 18 per row-key in `qk`) would convert at the fp32
kernel's achieved IPC. It does not, because the two instructions have different
issue ceilings here: fp32 achieves **2.25 4-wide FMAs/cycle** (14.2
cycles/row-key for 32 FMAs) while `bfmmla` sustains **1.88-2.5/cycle** (measured
per phase, round 2). So 12 `bfmmla`/row-key cost 5-6.4 cycles against the 7.1
that 16 fp32 FMAs cost -- about **1.2x on the matmul, not 1.33x** -- and the
split doubles the operand *load count* (a 16-byte group now carries 4 dims of 2
keys instead of 4 dims of 4), which takes most of the 1.2x back. Extending the
same treatment to P@V would add ~1.5-2%, not 10%.

Reverted under the "a novelty that does not beat baseline by 3-5% is reverted and
recorded" rule; re-verified after the revert (`test_dense` 65/65, paged sweep
504/504, worst bf16 1.70 ulp / fp32 1.61e-06). fp32 stays at 0.962-1.123 vs Mojo,
0.877-1.055 vs vLLM, and it is at its floor: **14.2 cycles/row-key = 32 fp32 FMAs
at 2.25/cycle, demonstrated by the fact that replacing a quarter of those
instructions with 3x-denser ones moves the total by 1.2%.**

## Round 5 — the Cauchy-Schwarz bounded base: implemented, parity fails, reverted

Implemented for bf16 exactly as specified, and it does *not* pass parity, so it
is reverted (the shipped kernel is unchanged and re-verified below).

What was built:

* `pack` accumulates `max|k|` for the head while it already reads every element
  (an unsigned max over the sign-masked bf16 halves -- positive bf16 values
  order like their bit patterns), stored in a 64-byte tail added to
  `slab_bytes`. That placement is what makes both driver paths work: fused (the
  thread packs the head it attends) and split (the pack phase ends in a
  barrier), with no per-thread aux needed.
* `qk` computes `m_r = c * ||q_r|| * max|k|` once per chunk (`k0 == 0`), reading
  `||q_r||` as one bf16-widened dot product per row, and then exponentiates the
  panel *in place*: `p = exp2(min(c*logit - m_r, 0))` straight into the bf16 P
  operand layout. The logits buffer, the tile-max scan, the second traversal
  and the accumulator rescale all disappear; `softmax` becomes the per-tile
  flush of the probability sums plus `alpha = 1`.

Parity: **40/65** `test_dense`, with `max|diff|` 7.0e-2 .. 5.2e-1 against a
3e-2 gate -- for both causal and non-causal cases, and for shapes where no row
is empty and no key is masked. The pattern (a uniform-looking ~20% error on the
output, independent of masking) is what a probability panel that has collapsed
to 1 would produce: `exp2(NaN)` through `vminq_f32(x, 0)` returns 0 (FMIN takes
the non-NaN operand), i.e. `p = 1` for every key. So `m_r` itself is the
suspect -- `NaN`, or 0 from a `max|k|` that never got written -- and not the
exponent path. The two things to look at, in order: whether the slab tail
written by `pack` survives to `kmax_of` on the path the test exercises
(`slab_bytes` is now 64 bytes longer, and `vd`'s V region ends exactly at the
tail), and the `vmaxv_u16` reduction of the 4-lane accumulator.

Second, independent defect: masked keys now exponentiate to 1.2e-38 instead of
exactly 0, because the -126 clamp is inside `exp2v`. For a row with no visible
key that makes `l` tiny-but-nonzero, so the epilogue's `1/l` returns `V`-scale
garbage instead of 0 -- the `Sq=5 Sk=2 causal=1` case at 4.7e-1 is that shape.
The fix is one select per panel (`x > -126 ? exp2(x) : 0`), or keeping the old
`mn == -inf` row path.

Not reached: the fp32 version of the same treatment, and the 41-round matrix --
neither is meaningful on a kernel that does not pass its parity gate. The
shipped binary re-verifies at `test_dense` **65/65**,
`test_paged_sweep` 504/504 (worst bf16 1.85 ulp, fp32 1.58e-06) and 3421-3452 us
for 2048:8:64 bf16, so nothing regressed.

---

## Round 4 — the fused register panel, implemented and falsified

Rewrote `DenseBF`'s three phases into one register-panel loop (branch saved in
the transcript; reverted, so the shipped file is round 3's). What was built:
`qk` packs the chunk's Q into the bf16 (row pair, dim group) operand panel and
records `q_base`/`lim0`/`nrows` in the policy's own aux area; `softmax`
records `m`/`l`/`c` and sets `alpha = 1` so the shared driver skips its rescale;
`pv` runs the panel loop -- 8-key Q@K^T panel into 16 `bfmmla` registers, per-row
mask, per-row max on the accumulators, lazy online update (the accumulator is
rescaled only when the running max actually moves), `exp2` in place, straight
to bf16 P registers (`zip_lo`/`zip_hi` + `vcvt_bf16_f32`), then P@V from those
registers with eight head dims per pass. No 4-byte logits buffer and no second
pass over it.

Measured with the alternating-process protocol (5 rounds, one process per
sample, variants rotated), against the shipped kernel:

| shape | shipped | fused panel | ratio |
|---|---:|---:|---:|
| 2048:8:64 bf16 | 3442.4 us | 5248.2 us | **1.525x slower** |
| 128:8:64 bf16 | 26.3 us | 33.4 us | **1.270x slower** |

It also failed 5 of 65 `test_dense` checks (my panel masking), but the
performance verdict settles it: there is no point debugging a kernel that is
50% slower. Reverted; `test_dense` 65/65 and `test_paged_sweep` 504/504 again
(worst bf16 1.89 ulp, fp32 1.62e-06), and the shipped binary re-confirms at
3426-3445 us.

**Why it loses, with the number that makes it structural.** The logits panel
fits in registers; the *output* accumulator does not (8 rows x 64 dims = 128
fp32 values). So the panel loop has to load and store the accumulator once per
panel per dim pass: at an 8-key panel that is `2 * (HD/dcp) * (RQT*dcp/4)`
register accesses per `RQT * W` row-keys = **4 accumulator loads/stores per
row-key**, against the **0.125** the shipped structure pays (it touches the
accumulator once per 256-key tile). The logits + P round trip it removes costs
1.75. Net: 4.0 - 1.75 = **+2.4 memory instructions per row-key**, and the
measured 1.52x is that plus the loss of the tile-wide rescale amortisation.
A 16-key panel is the widest that still fits 32 registers (4 row pairs x 8 key
pairs = 32 logit registers, leaving nothing for Q/K/V; 16 keys only fits with
`RQT = 4`, i.e. 2 row pairs), and it would halve the accumulator traffic to
2/rk -- still above the 1.75 it replaces, while `RQT = 4` doubles the K/V bytes
and measured **1.302x slower** on its own (round 1).

**The only formulation left** that removes the logits round trip without paying
the accumulator round trip is to delete the tile-wide max, so P can be written
in a single pass from a *fixed* per-row exponent base: bound `max |c q_r . k_j|`
by Cauchy-Schwarz from `max_j |k_j|` (accumulated during the pack, which already
touches every element) and `||q_r||`, and use `m_r = c * ||q_r|| * max|k|` for
the whole call. Then `l` is a plain sum, the accumulator is never rescaled, and
`qk` writes bf16 P directly (one pass, no logits store, no softmax read). The
cost is a looser base (~sqrt(D) x the true max, so probabilities ~1e-9 instead
of ~1) which is inside fp32's range, and the risk is the bound being violated by
a non-random input, which would overflow `exp2` to +inf and produce NaN. That is
the next thing to try, and it is a `pack` + `qk` change only.

---

## Round 3

### 1. `tests/test_paged_sweep.py` is 504/504 — the round-2 failure was the stale oracle

Round 2 reported 336/504 and traced it to the box's `_C.so` still carrying the
GQA store bug (its error on the first GQA case was bit-identical to the buggy
`head * HD` read). The box's venv has `mojo` but not on `PATH`, so
`scripts/build.py` bailed with "mojo not found"; with
`PATH=$PWD/.venv/bin:$PATH` it builds:

```
+ .venv/bin/mojo build cpu_attention/_mojo/src/_C.mojo --emit shared-lib -o cpu_attention/_C.so ...
built cpu_attention/_C.so
paged sweep: 504/504 within tolerance
  worst fp32 1.37e-06 at (32, 16, 16, 16, 17, 1, True)
  worst bf16 1.96 ulp at (128, 64, 16, 8, 33, 3, False)
```

The bf16 worst case is 1.96 against a 2.0 ulp gate — passing, but with no
margin; the two backends disagree by ~2 bf16 ulp at the peak output on
`hd128/page64/16q-8kv`.

### 2. bf16 K pack: four keys per step (-3.1% on `128:8:64`)

Two adjacent key pairs of one dim group are 16 bytes apart in the packed K^T
tile, so 4 keys of one dim group are 32 contiguous bytes: two 16-byte stores
instead of four 8-byte ones, and half as many loop iterations. `pad` is a
multiple of 64 and `t0` a multiple of 4, so a step never straddles a tile.
Alternating-process A/B, min of 5 rounds, `128:8:64 bf16`: **26.1 -> 25.3 us
(0.969x)**. In the 41-round matrix: 25.0 -> 24.8 us and `cpp/vllm` 1.114 ->
1.094 on that row.

The V pack was left alone: its 4-byte `memcpy` per key needs a 4x4 bf16
transpose to widen, and the shuffle cost (~20 instructions per 32 bytes against
~24 for two of the current groups) makes it a ~20% win on that loop rather than
the 2x the store width suggests. Not worth a rewrite on its own.

### 3. Round-3 matrix (41 rounds, `ctrl/mojo` tracks `cpp` within 0.005)

| dense row | dtype | cpp us | mojo us | vllm us | cpp/mojo | cpp/vllm |
|---|---|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 24.8 | 21.0 | 22.6 | 1.177 | 1.094 |
| 128:8:64 | fp32 | 43.0 | 42.2 | 46.0 | 1.017 | 0.934 |
| 512:8:64 | bf16 | 236.5 | 196.6 | 234.5 | 1.203 | 1.009 |
| 512:8:64 | fp32 | 582.7 | 523.1 | 589.3 | 1.114 | 0.989 |
| 1024:8:64 | bf16 | 870.0 | 775.2 | 907.6 | 1.122 | 0.958 |
| 1024:8:64 | fp32 | 2297.7 | 2078.0 | 2240.4 | 1.106 | 1.025 |
| 2048:8:64 | bf16 | 3451.6 | 3151.8 | 3531.8 | 1.095 | 0.977 |
| 2048:8:64 | fp32 | 9118.9 | 8191.9 | 8913.2 | 1.113 | 1.023 |
| 4096:8:64 | bf16 | 13448.2 | 12566.8 | 14050.0 | 1.070 | 0.957 |
| 4096:8:64 | fp32 | 36466.2 | 33415.6 | 34561.0 | 1.091 | 1.055 |
| 256:16:128 | bf16 | 218.9 | 188.9 | 211.9 | 1.159 | 1.033 |
| 256:16:128 | fp32 | 576.9 | 526.2 | 658.1 | 1.096 | 0.877 |
| 1024:16:128 | bf16 | 2869.5 | 2714.9 | 2921.5 | 1.057 | 0.982 |
| 1024:16:128 | fp32 | 8641.6 | 7928.9 | 9539.6 | 1.090 | 0.906 |
| 2048:16:128 | bf16 | 11141.3 | 10873.0 | 11426.6 | 1.025 | 0.975 |
| 2048:16:128 | fp32 | 33906.8 | 31391.1 | 37037.4 | 1.080 | 0.916 |
| 4096:16:128 | bf16 | 43901.1 | 42958.4 | 45205.0 | 1.022 | 0.971 |
| 4096:16:128 | fp32 | 135061.0 | 140133.3 | 144047.2 | **0.964** | 0.938 |
| 512:32:128 | bf16 | 1552.4 | 1441.6 | 1541.5 | 1.077 | 1.007 |
| 512:32:128 | fp32 | 4631.6 | 4124.5 | 4818.9 | 1.123 | 0.961 |
| 1024:8:32 | bf16 | 585.8 | 523.9 | 638.8 | 1.118 | 0.917 |
| 1024:8:32 | fp32 | 1288.5 | 1183.3 | 1321.4 | 1.089 | 0.979 |

Bands: dense `cpp/mojo` **0.964-1.203**, `cpp/vllm` **0.877-1.094**; 9/22 rows
<= 0.97 against vLLM, 1/22 against Mojo. Paged: `cpp/mojo` **0.476-1.012**,
`cpp/vllm` **0.245-1.036**, **19/20 at <= 1.0** (the one row over is
`cross b=1 ctx=256 bf16` at 1.036, where Mojo reads 1.46 — the preallocated
-output asymmetry). `d_cpp` <= 8.7e-3 dense bf16 / 2.5e-6 dense fp32 / 1.3e-3
paged bf16 / 4.8e-7 paged fp32.

### 4. What is left, per row

Nothing shipped this round for B (the fused bf16 register panel) or C (fp32
hi/lo); the costings from round 2 stand. The rows still over 0.97 are:

| row | cpp/mojo | cycles/row-key (ours / Mojo) | instructions/row-key (ours / Mojo) | floor |
|---|---:|---|---|---|
| 512:8:64 bf16 | 1.203 | 5.4 / 4.7 | 18.6 / 18.3 | 3.2 (`bfmmla` 2.5/cycle) |
| 128:8:64 bf16 | 1.177 | 12.9 / 11.0 (fixed cost ~1/3 of the call) | — | pack + fork-join |
| 256:16:128 bf16 | 1.159 | 5.2 / 4.5 | — | 3.2 |
| 1024:8:32 bf16 | 1.118 | 5.5 / 4.9 | — | 1.6 (4 `bfmmla`/rk) |
| fp32 D=64 (5 rows) | 1.09-1.11 | 14.2 / 12.7 | 52 / 49 | 14.2 = 32 FMA at 2.25/cycle |

The bf16 rows are at instruction parity with Mojo and lose on cycles only, which
is the logits round trip that the `qk -> softmax -> pv` contract forces; the fp32
rows are at the scalar-pipe roof (32 FMAs/row-key, 2.25/cycle achieved, i.e. the
FMA stream is the whole runtime) and only the hi/lo `bfmmla` restructure moves
them, at the 1.1-1.25x projection of round 2.

---

## Round 2 (same box, branch `cpp-arm-dense`)

Two things were done and one lever was costed out with the kernel's own measured
constants. Raw JSON: `bench_arm_final2.json` (41 rounds, dense + paged).

### 1. Paged GQA store bug — fixed (real, and it was reading off the tensor)

`PagedF32::store_head` / `PagedBF::store_head` computed `kvh = head / group` for
the cache slot but read the source `k`/`v` at `head * HD`. The fused qkv
projection carries one head per *kv* head, so every query head of a group past 0
read the wrong slice and the last group ran off the tensor. Indexing by `kvh`
matches `cpu_attn_vec.hpp`. Measured against the naive fp32 reference on the
sweep's GQA shapes (bf16, fixed seed):

| shape | before | after |
|---|---|---|
| hd 32, page 64, 16 heads / 4 kv, ctx 2049 b2 | 6.628e-03 | 3.615e-04 |
| hd 128, page 32, 16 heads / 8 kv, ctx 257 b1 | **NaN** | 1.108e-03 |

`tests/test_paged_sweep.py` reads **336/504**, and the cause is *not* this
kernel: the bf16 gate compares `cpp` against `mojo`, and the `mojo` arm on this
box reads **6.628e-03 on the first case and 5.596e-02 on the second against the
same reference** — the first number is bit-identical to the buggy `head * HD`
indexing above, i.e. the box's `_C.so` still has the GQA store bug (or its ARM
bf16 path was missed by the fix). Every bf16 GQA case therefore fails a 2-ulp
cross-backend gate while `cpp` is 4-20x closer to the truth. The fp32 gate
(against the naive reference) passes at **2.00e-06 against a 2e-06 tolerance**,
i.e. at the gate, not over it. Rebuilding `_C.so` on the box from the fixed
source is required before that sweep can be 504/504; nothing in this file can
move it.

### 2. Paged performance: no regression

Paged is byte-identical code this round (the GQA fix only changes which source
slice is stored). Re-measured 20/20 rows: `cpp/vllm` **0.273-1.339**, 18/20 at
<= 1.0, `cpp/mojo` **0.523-1.047**, `d_cpp` <= 1.3e-3 bf16 / 4.8e-7 fp32. The
two rows over 1.0 are again `b=1 ctx=256 bf16`, where the `mojo` arm reads
*1.14/1.48* — the harness asymmetry (vLLM's output is preallocated outside the
timed region), not a kernel difference.

### 3. fp32 hi/lo `bfmmla`: costed, not shipped — and it is not the 1.33x it looks like

The MAC-rate argument for the hi/lo split (`bfmmla` retires 16 MACs against 4
for a 4-wide fp32 FMA) is right about the *instruction*, but this core's
measured issue rates make the three-product form much less than 4x cheaper than
fp32 FMA:

* fp32 dense executes 32 fp32 FMAs per row-key and runs 14.2 cycles/row-key:
  **2.25 fp32 FMA/cycle achieved**, i.e. the fp32 kernel is FMA-throughput
  bound and the matmul *is* the whole runtime.
* bf16 `bfmmla` **per phase, measured by cycle-sampling attribution at
  2048:8:64**: `qk` 4 `bfmmla` in 1.6 cycles (**2.5/cycle**), `pv` 4 in 2.13
  cycles (**1.88/cycle**). A stride-1 microbenchmark of `bfmmla` vs `vbfma`
  could not settle the ceilings — with `-O2` the 16-accumulator array spilled
  and both loops read ~1 op/cycle, which contradicts the kernel's own 2.25, so
  the phase numbers above are the trustworthy ones.

So the hi/lo split costs 3 products x (4 Q@K^T + 4 P@V) = **24 `bfmmla` per
row-key** where fp32 needs 32 FMAs, and the loads double (hi and lo planes for
K, V and P). At the measured per-instruction costs that is 12/2.5 + 12/1.88 =
**11.2 cycles/row-key of matmul against 14.2**, plus the fp32 softmax (6.6
instructions/row-key, heavier than bf16's) which does not overlap — a projected
**1.1-1.25x, i.e. `cpp/mojo` 1.10 -> 0.88-0.98**. It would reach the gate, but
only just, and it is a ~300-line second policy (new K/V/P hi-lo layouts, Q split
on the fly, three accumulator products, a P panel written twice) whose
correctness has to be re-proven on all 22 rows. Not attempted in the window
left; the numbers above are what an implementation should be judged against.

### 4. Round-2 dense matrix (41 rounds, `ctrl/mojo` tracks `cpp` within 0.005)

| dense row | dtype | cpp us | mojo us | vllm us | cpp/mojo | cpp/vllm |
|---|---|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 25.0 | 21.2 | 22.4 | 1.181 | 1.114 |
| 128:8:64 | fp32 | 43.2 | 42.0 | 46.0 | 1.029 | 0.940 |
| 512:8:64 | bf16 | 238.9 | 196.6 | 234.6 | 1.215 | 1.018 |
| 512:8:64 | fp32 | 578.2 | 523.6 | 592.7 | 1.104 | 0.976 |
| 1024:8:64 | bf16 | 885.1 | 773.2 | 924.4 | 1.145 | 0.958 |
| 1024:8:64 | fp32 | 2307.7 | 2077.8 | 2257.0 | 1.111 | 1.022 |
| 2048:8:64 | bf16 | 3428.2 | 3152.8 | 3576.6 | 1.087 | 0.959 |
| 2048:8:64 | fp32 | 9180.7 | 8190.7 | 8907.9 | 1.121 | 1.031 |
| 4096:8:64 | bf16 | 13490.7 | 12553.0 | 14043.4 | 1.075 | 0.961 |
| 4096:8:64 | fp32 | 36551.4 | 33400.9 | 35033.1 | 1.094 | 1.043 |
| 256:16:128 | bf16 | 219.4 | 193.7 | 211.7 | 1.133 | 1.036 |
| 1024:16:128 | bf16 | 2892.3 | 2736.9 | 2947.5 | 1.057 | 0.981 |
| 1024:16:128 | fp32 | 8564.7 | 8003.9 | 9409.9 | 1.070 | 0.910 |
| 2048:16:128 | bf16 | 11160.1 | 10958.5 | 11416.7 | 1.018 | 0.978 |
| 2048:16:128 | fp32 | 33858.6 | 31347.1 | 36432.2 | 1.080 | 0.929 |
| 4096:16:128 | bf16 | 43901.1 | 43052.0 | 45066.4 | 1.020 | 0.974 |
| 4096:16:128 | fp32 | 135056.6 | 140333.8 | 144487.2 | **0.962** | 0.935 |
| 512:32:128 | bf16 | 1551.1 | 1436.1 | 1545.8 | 1.080 | 1.003 |
| 512:32:128 | fp32 | 4642.1 | 4145.2 | 4809.2 | 1.120 | 0.965 |
| 1024:8:32 | bf16 | 585.9 | 527.4 | 640.0 | 1.111 | 0.915 |
| 1024:8:32 | fp32 | 1289.0 | 1187.0 | 1321.2 | 1.086 | 0.976 |

Bands: dense `cpp/mojo` **0.962-1.215**, `cpp/vllm` **0.871-1.114**; 10/22 rows
at <= 0.97 against vLLM, 1/22 against Mojo. Round 2 added no dense performance
change beyond round 1 (`docs/arm/bench_arm_final.json` is the round-1 artifact;
the two runs agree to within the box's 1-2%).

---

# Round 1

`cpu_attention/csrc/cpu_attn_neon.hpp` only. Branch `cpp-arm-dense`, base
`b199f3a`. Harness: `tests/bench_cpp.py --isa neon --rounds 41` (one process,
four arms rotated, min of 41, duplicate `cpp` control arm = noise floor),
vLLM = `~/vllm_cpu_build/vllm_cpu_attn_bench.so` (NEON build). Ratios are
`ours / baseline`, **below 1 is faster**. Raw JSON: `bench_arm_final.json`.

## What changed

Two hot loops in both dense policies (bf16 `DenseBF`, fp32 `DenseF32`), nothing
else — the paged policies, the layouts and the call path are untouched.

1. **`pv` walks K/V with pointers instead of recomputing the address.** The
   bf16 body was 42 instructions for 16 `bfmmla`; 16 of those 42 were integer
   address arithmetic recomputed per 4-key group (`(dim pair, key group)` ->
   index -> byte offset, with the head-dim stride a runtime value). Now the P
   and V panels advance by 16 bytes per key group and the dim-pair bases are
   hoisted. 26.4 -> 21.7 instructions per row-key.
2. **Both loops step 16 keys per iteration instead of 4-8.** This is what
   actually bought the time: the kernel is *latency* bound, not issue bound.
   Cutting 18% of the instructions without widening the step moved 2048:8:64
   bf16 by 0.1% (3827.9 -> 3824.8 us) because the freed slots just filled with
   stall; going from 4 to 16 keys in flight took the same call to 3445.1 us
   (-10.0%). The bf16 softmax keeps eight independent `exp2` chains live
   instead of four, and `pv` issues 64 `bfmmla` per step so its eight loads run
   ahead of the matmul.
3. **The `-inf` row test left the inner loop** in both softmaxes. It was
   loop-invariant (`mn` is the row's running max) but the compiler kept a
   branch and re-splatted the polynomial constants every 4 keys (54
   instructions where 40 are the polynomial). A row with no visible key anywhere
   in the call is now a separate, rare path.

## Result

Measured effect, alternating-process A/B (`min` of 9, one process per sample,
variants alternated round-robin), 2048:8:64:

| variant | bf16 us | fp32 us |
|---|---:|---:|
| base `b199f3a` | 3827.9 | 9466.0 |
| after 1 (addresses only) | 3824.8 | — |
| after 2+3 (this patch) | **3445.1** (-10.0%) | **9185.6** (-3.0%) |

The 41-round three-arm matrix (control arm `ctrl/mojo` reads 0.965-1.222, i.e.
it tracks the `cpp` arm to within 0.005 on every row — the numbers below are
resolvable):

| dense row | dtype | cpp us | mojo us | vllm us | cpp/mojo | cpp/vllm |
|---|---|---:|---:|---:|---:|---:|
| 128:8:64 | bf16 | 25.6 | 21.0 | 22.3 | 1.217 | 1.147 |
| 128:8:64 | fp32 | 43.7 | 41.9 | 45.4 | 1.045 | 0.964 |
| 512:8:64 | bf16 | 242.4 | 202.6 | 232.4 | 1.197 | 1.043 |
| 512:8:64 | fp32 | 584.8 | 518.0 | 588.8 | 1.129 | 0.993 |
| 1024:8:64 | bf16 | 889.6 | 775.2 | 932.1 | 1.148 | 0.954 |
| 1024:8:64 | fp32 | 2306.2 | 2073.2 | 2263.3 | 1.112 | 1.019 |
| 2048:8:64 | bf16 | 3438.3 | 3133.0 | 3569.0 | 1.097 | 0.963 |
| 2048:8:64 | fp32 | 9174.3 | 8220.6 | 8859.0 | 1.116 | 1.036 |
| 4096:8:64 | bf16 | 13479.5 | 12601.3 | 14091.9 | 1.070 | 0.957 |
| 4096:8:64 | fp32 | 36584.3 | 33442.6 | 34520.5 | 1.094 | 1.060 |
| 256:16:128 | bf16 | 224.3 | 191.3 | 211.8 | 1.172 | 1.059 |
| 256:16:128 | fp32 | 584.7 | 523.8 | 658.2 | 1.116 | 0.888 |
| 1024:16:128 | bf16 | 2935.2 | 2754.1 | 2954.5 | 1.066 | 0.993 |
| 1024:16:128 | fp32 | 8673.8 | 8011.8 | 9519.2 | 1.083 | 0.911 |
| 2048:16:128 | bf16 | 11284.3 | 11006.5 | 11446.8 | 1.025 | 0.986 |
| 2048:16:128 | fp32 | 34037.4 | 31366.2 | 36043.7 | 1.085 | 0.944 |
| 4096:16:128 | bf16 | 44301.3 | 43012.9 | 45422.6 | 1.030 | 0.975 |
| 4096:16:128 | fp32 | 135357.8 | 140234.7 | 146406.1 | **0.965** | 0.925 |
| 512:32:128 | bf16 | 1578.0 | 1461.0 | 1552.3 | 1.080 | 1.017 |
| 512:32:128 | fp32 | 4674.7 | 4177.8 | 4812.9 | 1.119 | 0.971 |
| 1024:8:32 | bf16 | 601.2 | 524.0 | 639.8 | 1.147 | 0.940 |
| 1024:8:32 | fp32 | 1289.7 | 1191.5 | 1322.9 | 1.082 | 0.975 |

| | rows | `cpp/vllm` <= 0.97 | `cpp/mojo` <= 0.97 | min-max cpp/mojo | min-max cpp/vllm |
|---|---:|---:|---:|---|---|
| dense bf16 | 11 | 5 | 0 | 1.025-1.217 | 0.940-1.147 |
| dense fp32 | 11 | 4 | 1 | 1.045-1.129 | 0.888-1.060 |
| dense all | 22 | 9 | 1 | 0.965-1.217 | 0.888-1.147 |
| paged (bf16+fp32) | 20 | 18 | 14 | 0.512-1.044 | 0.260-1.360 |

Progress against the starting state of this branch (same harness, 41 rounds):
bf16 `cpp/mojo` was **1.20-1.29** on every row and is now **1.03-1.22**; fp32
was 1.06-1.15 and is now 1.05-1.13. `cpp/vllm` went from 0.90-1.25 (4/22 <= 1)
to 0.89-1.15 (9/22 <= 0.97).

**Paged is untouched** (`git diff` is only the two dense policy structs):
`cpp/vllm` 0.26-1.36, 18/20 <= 1.0, `cpp/mojo` 0.51-1.04. The two rows above 1.0
are `b=1 ctx=256 bf16` (1.19 self, 1.36 cross) where the Mojo arm reads the same
1.20/1.34 — it is the harness asymmetry the CPP report already documents (vLLM's
output tensor is preallocated outside the timed region, `torch.empty` is 1.2 us
of an 8 us call), not a kernel difference. The cache layout is byte-identical
(Mojo and vLLM staging both still read it: `d_cpp` 1.2e-3 bf16 / 3.4e-7 fp32).

## Parity

Every gate passes on the box with `CPU_ATTENTION_BACKEND=cpp` after the change:

| gate | result |
|---|---|
| `tests/test_dense.py` | 65/65 checks |
| `tests/test_flash_api.py` | all checks |
| `tests/fuzz_dense.py --cases 80 --max-seq 900` | 80/80, worst case 0.52x its tolerance |
| `audit/audit_dense.py` (naive fp32 reference) | bf16 <= 9.4e-3 (gate 3e-2), fp32 <= 2.8e-6 (gate 2e-4) |
| `audit/audit_paged.py` | bf16 <= 1.2e-3, fp32 <= 4.2e-7 |
| Mojo backend unchanged | `tests/test_dense.py`, `test_flash_api.py` with `BACKEND=mojo` still pass |

## Falsification ledger

Each measured in-process with the duplicate control arm (or with the
alternating-process protocol for variants that need their own `.so`), one change
at a time on top of the row above.

| idea | number | verdict |
|---|---|---|
| `pv` address strength reduction alone (bf16) | 26.37 -> 22.06 instructions/row-key, **3827.9 -> 3824.8 us (0.1%)** | kept as a prerequisite of the unroll, but it buys no time by itself: the kernel is not issue bound |
| softmax + `pv` at 16 keys per step (bf16) | **3827.9 -> 3445.1 us (0.900x)** | kept |
| softmax `-inf` branch hoisted + 8-key step + `pv` pointer walk (fp32) | **9466.0 -> 9185.6 us (0.970x)** | kept |
| `RQT = 4` for bf16 dense (2 row pairs, Mojo's D=64 choice) | 3827.9 -> 4982.4 us (**1.302x slower**) | reverted |
| `kt = 128` for D<128 (ff. L1-resident K/V tile: 32 KB vs 64 KB) | 3868.9 us (1.011x slower) | reverted |
| `rq = 32` (rows per work item) at D=64 | 3856.7 us (1.008x slower) | reverted |
| `rq` capped at 64 for `seq_q <= 512` | 512:8:64 244.6 vs 244.9 (1.000x), 128:8:64 26.0 vs 26.5 (noise) | not kept |
| 8/4-key `pv` step with 8 loads + 32 `bfmmla` (before widening further) | part of the 0.900x above | superseded by 16 keys |

## Where the remaining gap is, with numbers

At 2048:8:64 bf16 (cycles/row-key from `perf stat` differentials, 5 vs 205
calls, so process overhead is subtracted; 3.11 GHz measured from
cycles/elapsed):

| | cycles/row-key | instructions/row-key | IPC | L1D accesses | L2 accesses | stall_backend |
|---|---:|---:|---:|---:|---:|---:|
| ours, before this patch | 5.99 | 26.4 | 4.40 | 5.57 | 0.78 | 1.56 |
| ours, after | **5.09** | 18.6 | 3.65 | — | — | — |
| Mojo (the target) | 4.67 | 18.3 | 3.78 | 5.08 | 0.56 | 0.94 |
| vLLM NEON | 5.30 | — | — | — | — | — |

Cycle-sampled attribution (`perf record -e cycles:u`, 2048:8:64 bf16): `pv`
36%, `qk` 27%, softmax 27%, `qk` store epilogue 5%. `qk` runs at 2.5
`bfmmla`/cycle and `pv` at 1.88, so **the eight `bfmmla` per row-key are 3.7 of
the 5.1 cycles**; the bf16 matmul floor is 3.2 cycles/row-key at the measured
2.5/cycle issue rate. Instruction count is now at parity with Mojo (18.6 vs
18.3), which means the residue is **latency, not instructions** — the earlier
26.4 -> 21.7 cut moved nothing until the step width changed.

What that leaves, per row:

* **bf16 D=64/32 long rows (`cpp/mojo` 1.07-1.15, worst at 512:8:64 1.197):**
  5.1 vs 4.67 cycles/row-key. Mojo holds the logits *in registers* (16-key
  tile, `qt=4`, `bfmmla` accumulators are the logits) so it has no logits
  store/load round trip and no separate softmax pass; our policy contract
  (`qk` -> `softmax` -> `pv` through the `float* logits` buffer) forces one.
  Closing this needs the fused row kernel, i.e. one function that keeps a
  16-key x 2-row-pair logit panel in the 32-register file, exps it in place and
  feeds P@V from it — a rewrite of the policy, not a knob. Estimated ceiling:
  Mojo's 4.67 x (18.3/18.6) ~ 4.6, i.e. parity, not the 0.97 the goal asks for.
* **fp32 (`cpp/mojo` 1.05-1.13):** 32 scalar-pipe FMAs per row-key is the whole
  budget (16 for Q@K^T, 16 for P@V) and NEON has no fp32 matmul instruction.
  At the measured 2.16 FMA/cycle that is 14.8 cycles/row-key of arithmetic
  against 12.7 for Mojo, whose fp32 kernel uses a larger `qt` and an
  L1-resident accumulator. The only way past 2.2 FMA/cycle is bf16 `bfmmla`
  with a hi/lo split (`q_hi*k_hi + q_hi*k_lo + q_lo*k_hi` = 3 `bfmmla` per 16
  MACs against 4 fp32 FMAs), which is ~25% fewer instructions in both phases at
  a ~1.5e-5 relative error — inside the 2e-4 gate. Not attempted here.
* **Short rows (128:8:64 1.217, 256:16:128 1.172):** per-call fixed cost. The
  V pack for bf16 still moves 4 bytes per store (four `memcpy` of 4 bytes per
  16-byte group); at 128 keys it is ~2.6 us of a 25.6 us call. A two-dim-pair
  pack would halve it (~5% of that row) and nothing more.


## Round 8 (primary, on `main`): the bounded base, re-verified and reverted

The bound is correct. Instrumented on the box with a Python reference of the
same formula (`p = exp2((q.k)*c - m_r)`, `c = scale*log2e`,
`m_r = c*||q_r||*max|k|`), for `Sq=Sk=128, H=8, D=64`, non-causal, seed 0:

| head | kmax | exact sum(r0) | kernel sum(r0) |
|---|---:|---:|---:|
| 0 | 4.4062 | 1.5819 | 1.58198 |
| 1 | 3.6875 | 4.6891 | 4.68932 |
| 2 | 3.6875 | 8.6169 | 8.61731 |
| 4 | 3.6094 | 8.6401 | 8.64054 |
| 7 | 3.8906 | 2.6342 | 2.6343 |

So `pack`'s `max|k|`, the `||q_r||` prologue and the in-place exponential are
all right. (An earlier version read `||q_r||` from the packed Q group, where one
16-byte group holds *both* rows of a row pair -- the bound came out 1.4x too
large and every probability silently shrank; reading the query row directly
fixes it.)

What still fails is the handoff to P@V: heads 1, 4 and 6 of that case give an
output 0.06-0.25 wrong while their `qk` sums are exact, and the implied
numerator (`out * l`) is up to half the exact one for some rows. It reproduces
identically with one worker, and with `CPU_ATTENTION_FUSE=0` and `=1`, so it is
a deterministic row-mapping bug in `qk`-writes-P -> `pv`-reads-P, not a race.
That is the one open path to the aarch64 dense rows; everything needed to
reproduce is: `pack` accumulates `max|k|` into a 64-byte slab tail,
`qk` computes `m_r` per row and exponentiates its 2x2 `bfmmla` accumulators into
the bf16 P operand layout (`stb(dst, combine(cvt(ev0), cvt(od0)))`,
`stb(dst+16, ...)`), `softmax` flushes the per-row sums, `pv` unchanged.
