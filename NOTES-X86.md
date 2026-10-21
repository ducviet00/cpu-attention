# NOTES-X86 — the AMX campaign (x86 kernel agent)

Ratios are `ours/baseline` (below 1 = we win). Box: `ubuntu@3.27.45.233`,
m8i.xlarge = **4 vCPU / 2 physical cores**, Xeon 6975P-C (Granville Rapids),
`avx512_bf16` + `amx_bf16` + `amx_tile` + `amx_int8` (no `amx_fp16` flag),
480 MiB L3, torch 2.14.1+cpu, mojo 1.1.0, gcc 15.2.
Unless stated: `tests/bench_vs_refs.py --rounds 5 --threads 2`, causal, medians
of 5, all variants interleaved in one process.

## 0. Session baseline (reproduced, `--isa vec`)

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

**So AMX is worth 4-9.6x over our kernel and 2.2x over SDPA.** Their AMX kernel
runs at **1.33 TFLOP/s** (2 threads) at S=1024:16:128 — 0.66 TFLOP/s/core
against oneDNN's 1.45 TFLOP/s/core bf16 GEMM ceiling measured in the same
process (`scripts/prof/x86/amx_probe.py`: bf16 matmul 1119-1497 GFLOP/s 1 thread,
fp32 218, SDPA bf16 attention 350-464).

**The target is their AMX number, not their VEC number.** A win against `--isa vec`
that loses to `--isa amx` is hollow; `tests/bench_vs_refs.py --isa amx` is the
honest baseline and I will report both.

## 2. Where the time goes once you have AMX (arithmetic, from measurements above)

Per (query row, key) "logit", attention needs `2*D` MACs. Measured ceilings:
- AVX-512 fp32 FMA: **249.6 GFLOP/s/core** (`fma fp32`, amxpeak) = 64 flops/cycle.
- AVX-512 `vdpbf16ps`: **119 GFLOP/s/core** = 32 flops/cycle — *no better than
  fp32 FMA* on this part. This is the whole reason our bf16 is not faster than
  our fp32 and why the `dpbf16ps` path is a dead end: it costs 16 cycles/logit at
  D=128, and our measured 25.6 cycles/logit is that plus softmax.
- AMX bf16 (oneDNN GEMM): **1.45 TFLOP/s/core** = 372 flops/cycle, 11.7x the
  `dpbf16ps` rate.
- vLLM's *attention* at 1.33 TFLOP/s = 341 flops/cycle → 3.0 cycles/logit at
  D=128 (2 cores, 8.4 M logits, 3241 us). Matmul alone at 372 flops/cycle is
  **1.38 cycles/logit**, so **their softmax+bookkeeping is ~1.6 cycles/logit**.
  At D=64 the matmul term halves (0.69) and they run at ~2.4 c/logit.

Conclusion: an AMX kernel turns attention from matmul-bound (16 c/logit) into
**softmax-bound (~1.5-2 c/logit)**. The whole design must be organised around
the softmax and around keeping the AMX tiles fed, not around the matmul.

## 3. AMX mechanics established on this box (build traps)

- `arch_prctl(ARCH_REQ_XCOMP_PERM /*0x1023*/, XFEATURE_XTILEDATA /*18*/)` must
  succeed before any tile instruction, or you get SIGILL.
- gcc 15's AMX intrinsics (`_tile_loadd`, `_tile_dpbf16ps`, ...) take **literal
  tile numbers** (an `"i"` constraint), not a `__tile` type. Use integer ids.
- `_tilecfg` layout that matters: palette at byte 0, **`colsb[8]` are `uint16_t`
  at offset 16**, `rows[8]` are `uint8_t` at offset 48. Writing colsb as bytes
  gives colsb=0x4040 and `ldtilecfg` faults (I lost an hour to that).
- A tile config must be re-established after `tilerelease`, and the *config is
  lost* if a function returns after `_tile_release()` — re-issue per region.
- **`-mamx-fp16` is useless here**: the CPU does not advertise `amx_fp16`
  (GNR part but the flag is masked in this VM), so `tdpfp16ps` is out. Only
  `amx_bf16` + `amx_int8` are available.
- `std.math`-free microbenchmark: `scripts/prof/x86/amxpeak.c`. Its
  `amx tdpbf16ps` peak loop reads **243 GFLOP/s**, which is 6x off oneDNN's
  1450 — the loop is a 6-accumulator chain with a `tileloadd` WAR hazard per
  iteration and is *not* a valid peak probe. Trust oneDNN's GEMM, not it.
