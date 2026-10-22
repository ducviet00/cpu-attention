# Working protocol for agents on the Graviton box (2026-10-22)

Read this before you run anything. It is mechanics, not conclusions: every
number in `docs/` and `AGENTS.md` is a previous campaign's claim and must be
reproduced before you believe it.

## The objective

`ours / baseline < 1` in **both** columns (`o/sdpa`, `o/vllm`) for every row of
`tests/bench_vs_refs.py`, with parity green, on ARM (Graviton 5 / Neoverse-V3).
Ratio < 1 = we are faster. ARM only this session.

## The box

One shared box, currently **m9g.2xlarge, ubuntu@52.77.215.13**, 8 cores,
Neoverse-V3, no SME, SVE 128-bit, `bf16` `i8mm` `asimdfhm`.
Everything on it is disposable. Do not `rm -rf` anything you did not create.

### Rule zero: one measurement at a time

All builds and all timing runs go through `scripts/box_run.sh`, which takes an
`flock` on the box. Two agents measuring at once invalidate both numbers.

```bash
# from your worktree root
scripts/box_run.sh ubuntu@52.77.215.13 'cd ~/<yourdir> && .venv/bin/python tests/bench_vs_refs.py --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so --isa neon --json ~/out.json'
```

### Your own directory on the box

Each agent owns one remote dir; never sync into another agent's.

```bash
REMOTE_DIR=arm-a scripts/box_sync.sh arm64 ubuntu@52.77.215.13 .   # rsync + rebuild + parity
```

The first time, the remote dir has no venv; link the shared one:

```bash
ssh -i ~/.ssh/id_ed25519 ubuntu@52.77.215.13 'ln -sfn ~/cpu-attention/.venv ~/arm-a/.venv'
```

Build alone is `PATH=$PWD/.venv/bin:$PATH .venv/bin/python scripts/build.py`
(~60 s; the Mojo step dominates). Parity is `tests/test_dense.py` +
`tests/test_flash_api.py`.

### The vLLM reference

`~/vllm_cpu_build/vllm_cpu_attn_bench.so` (built with `-DARM_BF16_SUPPORT
-march=armv8.2-a+bf16+i8mm`, ISA tag `neon`). `~/vllm` is the pinned checkout
(`db3b0f0`); `csrc/cpu/cpu_attn_neon.hpp` is the fp32 kernel and
`cpu_attn_neon_bfmmla.hpp` the bf16 one. Read them; they are the bar to beat.

## How to measure — the rules that came out of the third campaign

These are not preferences; each one cost a session of wrong conclusions.

1. **The box must be idle.** A foreign `mojo build` inflated a row by 1.7x and a
   foreign python by 25%. `scripts/verify.sh` waits for a 1-minute load average
   under 0.3 and prints it; do the same before your own runs
   (`cut -d' ' -f1 /proc/loadavg`).
2. **Rotate the arm order, and use a control.** Whichever arm runs first in a
   round is 1-3% slower: two arms running *identical* code measured 3.4% apart.
   `tests/bench_vs_refs.py` and `tests/ab_two_trees.py` rotate now, and
   `bench_vs_refs --control` adds a duplicate `ours` arm whose ratio is the run's
   noise floor (0.99-1.01). **Any claim under 3% needs a control reading.**
3. **Judge on the minimum, not the median.** Interference can only slow a sample
   down. Both are printed; `scripts/band.py` and `scorecard.py` use the min.
4. **Plan-level knobs must be toggled inside one process** (they key the call
   plan). Toggling one inside a timing loop costs up to 19% by itself.
5. **Build on x86 before every handover.** An ARM-only file without
   `comptime if BFMMLA` breaks the x86 build at codegen.
6. **Do not cache K/V across calls**, however tempting: the benchmark reuses its
   tensors, so it would look like a large win while breaking SDPA semantics.

## How to measure

* `tests/bench_vs_refs.py` is the scorecard: interleaved in one process,
  medians, ratios `ours/baseline`. Use `--json` so `scripts/scorecard.py` can
  read it. `--rounds 5` is enough for a screen; `--rounds 9` for a verdict.
* These boxes throttle. **Never compare two separate runs.** For an A/B use
  `tests/ab_two_trees.py --old <tree> --new <tree>`, which alternates processes
  between two trees and reports a per-row band, or interleave both variants in
  one process yourself.
* Report a band (min..max over repeat runs), not a single number, for anything
  you want merged.
* Parity gates are fixed and are checked by the bench itself (`d=` column,
  fp32 <= 2e-4, bf16 <= 3e-2 vs SDPA) plus `tests/test_dense.py`,
  `tests/test_flash_api.py`, `tests/fuzz_dense.py`.

## Machine facts measured 2026-10-22 on this box (reproduce, do not trust)

| quantity | value |
|---|---|
| clock (serial add chain) | 2.76-3.02 GHz, varies per core and with load |
| `bfmmla` throughput, 8 independent accs | **~105 MAC/ns/core** (~35 MAC/cycle) |
| single-core streaming read | ~26 GB/s (latency-bound loop; treat as an upper-ish bound) |
| L1d / L2 / L3 | 64 KB / 2 MB / 48 MB (L3 shared) |
| vLLM bf16 4096:8:64 | 13.80 ms = **5.4 cycles/row-key** |
| ours bf16 4096:8:64 | 13.89 ms = 5.4 cycles/row-key |

At D=64 a row-key needs 8 `bfmmla` (4 for QK + 4 for P@V), so the matmul floor
is ~3.7 cycles/row-key: everything above that is softmax, loads and stalls.

## Reporting

When you claim a win, hand over: the exact command, the JSON path, the band, and
the parity line. Vague claims get re-measured by the integrator and dropped.
