# Status: ours vs SDPA vs vLLM `csrc/cpu`

Ratio convention (`AGENTS.md` §0): **`ours / baseline`**, so **< 1 means we win**.
Times are µs of the whole call; ours includes the K/V pack and its allocations,
vLLM's is attention only (its cache is staged before the timer). Measured with
`tests/bench_vs_refs.py` (variants interleaved in one process, medians of 7,
causal). Both platforms same session, 2026-10-21.

## Graviton 5 — m9g.2xlarge, 8 cores, 8 threads, `isa=neon`

| shape | bf16 o/sdpa | bf16 o/vllm | fp32 o/sdpa | fp32 o/vllm |
|---|---:|---:|---:|---:|
| S=128  H=8  D=64  | 1.03 | 2.33 | **0.56** | 1.79 |
| S=512  H=8  D=64  | **0.69** | 1.60 | **0.53** | 1.51 |
| S=1024 H=8  D=64  | **0.90** | 1.43 | **0.67** | 1.32 |
| S=2048 H=8  D=64  | 1.03 | 1.40 | **0.84** | 1.38 |
| S=4096 H=8  D=64  | 1.12 | 1.40 | **0.95** | 1.38 |
| S=256  H=16 D=128 | **0.77** | 1.78 | **0.65** | 1.31 |
| S=1024 H=16 D=128 | **0.84** | 1.41 | **0.80** | 1.24 |
| S=2048 H=16 D=128 | **0.92** | 1.40 | **0.99** | 1.26 |
| S=4096 H=16 D=128 | 1.11 | 1.47 | 1.14 | 1.28 |
| S=512  H=32 D=128 | **0.62** | 1.52 | **0.59** | 1.30 |
| S=1024 H=8  D=32  | **0.89** | 1.43 | **0.69** | 1.40 |

**17 of 22 rows beat SDPA** (all 11 fp32 rows, 6 of 11 bf16). vLLM is ahead
everywhere: 1.24-2.33x.

## Xeon — m8i.2xlarge, 4 physical cores, 4 threads, `isa=vec`

| shape | bf16 o/sdpa | bf16 o/vllm | fp32 o/sdpa | fp32 o/vllm |
|---|---:|---:|---:|---:|
| S=128  H=8  D=64  | 1.39 | 1.80 | **0.76** | 1.74 |
| S=512  H=8  D=64  | 1.88 | 1.41 | **0.57** | 1.62 |
| S=1024 H=8  D=64  | 2.39 | 1.33 | 1.20 | 1.89 |
| S=2048 H=8  D=64  | 3.38 | 1.51 | 1.31 | 1.97 |
| S=4096 H=8  D=64  | 5.37 | 2.00 | 1.58 | 1.90 |
| S=256  H=16 D=128 | 2.62 | 1.65 | **0.87** | 1.78 |
| S=1024 H=16 D=128 | 4.38 | 1.72 | 1.34 | 1.97 |
| S=2048 H=16 D=128 | 5.66 | 1.69 | 1.62 | 1.98 |
| S=4096 H=16 D=128 | 6.28 | 1.89 | 1.78 | 2.05 |
| S=512  H=32 D=128 | 3.45 | 1.78 | 1.17 | 3.03 |
| S=1024 H=8  D=32  | 1.65 | 1.33 | **0.72** | 1.21 |

**4 of 22 rows beat SDPA** (fp32 at short/medium S). Two platform facts, not
kernel facts:

* **This box has AMX and torch SDPA uses it for bf16** (~1.3 TFLOP/s against our
  ~0.3), so the bf16 `o/sdpa` column (1.4-6.3) is an *instruction-set* gap, not a
  kernel gap. vLLM's `vec` build does not use AMX either, which is why our bf16
  `o/vllm` (1.3-2.0) is much closer than `o/sdpa`.
* 4 physical cores at 4 threads: the same kernel that wins on the 8-core laptop
  loses here, because per-core bandwidth per row is worse (see §9.10).

## Summary

| platform | beats SDPA | beats vLLM |
|---|---|---|
| Ryzen 7840H, 8 cores (laptop) | 11 of 12 | bf16 D=64 only |
| Graviton 5, 8 cores | 17 of 22 | none (best 1.24) |
| Xeon m8i, 4 cores | 4 of 22 | none (best 1.21) |

What is left is uniform across platforms: the softmax block, 40% of our call
against ~9% of vLLM's by ablation, with Q@K^T and P@V at parity (`AGENTS.md`
§9.10b). On Xeon, bf16 additionally needs an AMX path to compete with SDPA.
