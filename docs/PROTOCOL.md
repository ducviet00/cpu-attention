# Measurement protocol

The campaign's definition of done: **`ours / baseline < 1` for both baselines,
on both platforms, for every shape in `tests/bench_vs_refs.py`, with parity
green.** Ratios are `ours / baseline`, so below 1 is a win. Everything below
exists so that number means something.

## The three rules

1. **Interleave, never compare runs.** These boxes throttle 2-2.5x under
   sustained vector load (a laptop Ryzen drops to ~2.0 GHz mid-sweep; Graviton
   rows move +/-25%). `tests/bench_vs_refs.py` times every variant in one
   process, round-robin, and reports the median. A/B of two *trees* goes through
   `tests/ab_two_trees.py`, which alternates short processes.
2. **Charge ourselves the pack, charge vLLM nothing.** Ours includes the K/V
   pack and its allocations, because that is what the engine calls. vLLM's cache
   is staged before its timer starts. That asymmetry is part of the contract, so
   the pack is a place to win, not an excuse: `tests/bench_phases.py` splits it
   out.
3. **Parity travels with the number.** `bench_vs_refs.py` prints `max|diff|` vs
   SDPA per row and exits non-zero outside tolerance (fp32 2e-4, bf16 3e-2). A
   fast-but-wrong kernel is not a result.

## What "the box" is

| name | instance | cores seen | ISA |
|---|---|---|---|
| x86 target | `m8i.2xlarge` | 8 vCPU = 4 physical | AVX-512 + `avx512_bf16` + AMX |
| arm target | `m9g.2xlarge` | 8 cores | NEON + `bf16` + `i8mm` + SVE2 |
| x86 dev | `m8i.xlarge` | 4 vCPU = 2 physical | same CPU, half the cores |
| arm dev | `m9g.xlarge` | 4 cores | same CPU, half the cores |
| laptop | Ryzen 7840H | 8 cores | AVX-512 + `avx512_bf16`, **no AMX** |

Development happens on the `xlarge` slices and the laptop; the **target** boxes
are what the scorecard is reported on, because the number of cores changes
per-core bandwidth and therefore the ratio. Conclusions about arithmetic
transfer between them; conclusions about bandwidth do not.

## Running it

```bash
# one box, all 22 rows, both dtypes, causal, with the reference kernel
.venv/bin/python tests/bench_vs_refs.py \
    --vllm ~/vllm_cpu_build/vllm_cpu_attn_bench.so --isa vec --rounds 5 \
    --json /tmp/x86.json

# pack vs row kernel
.venv/bin/python tests/bench_phases.py --vllm ... --isa vec

# the scorecard (a win is < 1 in both ratio columns)
.venv/bin/python scripts/scorecard.py /tmp/x86.json /tmp/arm.json
```

`--isa neon` on ARM. `--shapes 1024:16:128,...` narrows the sweep while
iterating; `--dtype`, `--causal 0`, `--threads N` exist too.

The vLLM reference is `csrc/cpu` pinned at commit `db3b0f0`, built by
`scripts/box_sync.sh` as `~/vllm_cpu_build/vllm_cpu_attn_bench.so` (VEC on x86,
NEON on ARM, no AMX on x86 -- the AMX build is a *separate*, optional reference
since the campaign's target is the VEC/NEON kernel).

## Reporting a result

* the table, from `scorecard.py`, not hand-typed;
* the exact command and the box;
* two runs when a decision depends on a few percent (report the band);
* what was falsified, with the number that falsified it.

## Trusting the notes in this repo

`AGENTS.md`, `docs/` and old commit messages are a previous campaign's notes.
The mechanics in them (build traps, Mojo syntax, layout invariants, failure
modes) have been reliable; several of the performance claims have not survived
re-measurement. Reproduce before building on any of them, and fix the note when
it is wrong.
