# Benchmark history

The current state lives in `docs/STATUS.md`; the measurement rules in
`docs/PROTOCOL.md`. This file keeps the older tables, clearly labelled, because
their shape explains the campaign: every entry below was believed at the time
and several were later shown to be wrong.

## Current code, laptop (Ryzen 7840H, 8 cores), 2026-10-21

See `docs/STATUS.md`. 22/22 rows beat SDPA, 10/22 beat vLLM's VEC kernel.

## Previous campaign — 8-core Graviton 5 and 4-core Xeon, 2026-10-21

Measured on an older tree (before the pack-workspace fix, which alone is
1.4-1.7x on the laptop, and before the query-row and dispatch changes).
Superseded; kept as history. Ratios `ours / baseline`, `<1` is a win.

Graviton 5 (m9g.2xlarge, 8 threads, `isa=neon`):

| shape | bf16 o/sdpa | bf16 o/vllm | fp32 o/sdpa | fp32 o/vllm |
|---|---:|---:|---:|---:|
| S=128  H=8  D=64  | 1.03 | 2.33 | 0.56 | 1.79 |
| S=512  H=8  D=64  | 0.69 | 1.60 | 0.53 | 1.51 |
| S=1024 H=8  D=64  | 0.90 | 1.43 | 0.67 | 1.32 |
| S=2048 H=8  D=64  | 1.03 | 1.40 | 0.84 | 1.38 |
| S=4096 H=8  D=64  | 1.12 | 1.40 | 0.95 | 1.38 |
| S=256  H=16 D=128 | 0.77 | 1.78 | 0.65 | 1.31 |
| S=1024 H=16 D=128 | 0.84 | 1.41 | 0.80 | 1.24 |
| S=2048 H=16 D=128 | 0.92 | 1.40 | 0.99 | 1.26 |
| S=4096 H=16 D=128 | 1.11 | 1.47 | 1.14 | 1.28 |
| S=512  H=32 D=128 | 0.62 | 1.52 | 0.59 | 1.30 |
| S=1024 H=8  D=32  | 0.89 | 1.43 | 0.69 | 1.40 |

Xeon (m8i.2xlarge, 4 physical cores, `isa=vec`):

| shape | bf16 o/sdpa | bf16 o/vllm | fp32 o/sdpa | fp32 o/vllm |
|---|---:|---:|---:|---:|
| S=128  H=8  D=64  | 1.39 | 1.80 | 0.76 | 1.74 |
| S=512  H=8  D=64  | 1.88 | 1.41 | 0.57 | 1.62 |
| S=1024 H=8  D=64  | 2.39 | 1.33 | 1.20 | 1.89 |
| S=2048 H=8  D=64  | 3.38 | 1.51 | 1.31 | 1.97 |
| S=4096 H=8  D=64  | 5.37 | 2.00 | 1.58 | 1.90 |
| S=256  H=16 D=128 | 2.62 | 1.65 | 0.87 | 1.78 |
| S=1024 H=16 D=128 | 4.38 | 1.72 | 1.34 | 1.97 |
| S=2048 H=16 D=128 | 5.66 | 1.69 | 1.62 | 1.98 |
| S=4096 H=16 D=128 | 6.28 | 1.89 | 1.78 | 2.05 |
| S=512  H=32 D=128 | 3.45 | 1.78 | 1.17 | 3.03 |
| S=1024 H=8  D=32  | 1.65 | 1.33 | 0.72 | 1.21 |

## Reading the old Xeon table

The bf16 `o/sdpa` column (1.4-6.3) was read at the time as "a kernel gap that
needs work on the kernel". It is not: torch's SDPA reaches oneDNN's AMX kernels
on this part while every AVX-512 kernel -- ours and vLLM's VEC build -- tops out
3-9x lower. The gap is an instruction-set gap, and the only way through it is an
AMX kernel of our own. That is why the AMX work exists.

The other correction: `--isa vec` was the only vLLM reference the previous
campaign had, and it is not the ISA vLLM would choose on this box. Whatever we
report, the number that matters is against what vLLM actually runs there
(`--isa amx` on x86, `neon` on Graviton). On x86 the AMX path is bf16-only, so
fp32's honest baseline remains VEC.
