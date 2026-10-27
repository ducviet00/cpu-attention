# AMX 2026-10-27 (in progress)

Box: GCP c4-standard-8, Xeon Platinum 8581C, 4C/8T; not 6975P-C.
Base: a1d9fdf. vLLM source db3b0f0fae4bc98492b3863685e541e62bc59576.
No local benchmarks. No Mojo edits. No default changes.

## Measurement

OMP_NUM_THREADS=4 OMP_PROC_BIND=spread OMP_PLACES=cores; four torch threads.
All builds/parity/timing serialized with ~/bench.lock. Both public C++/Mojo
wrappers allocate outputs. vLLM uses pre-staged caches and preallocated output,
as specified. BF16 vLLM AMX; fp32 vLLM VEC. Matrix rounds dense/paged both 41.

Old/new A/B loads two independently linked C++ libraries into the same process,
rebinds only dense_forward, and invokes the public wrapper. Both variants allocate.
Raw samples, minimum, median, exact old/new output difference and duplicate arms
are retained. Trials 1/2 used 101 rotated rounds with a new duplicate. From trial
3 the protocol uses all 120 permutations of five arms (old/new plus a duplicate
of each and Mojo); this balances predecessor cache warming as well as position.

## Profiling

The initial ablate2.py profiles are INVALID: importing torch in a parent with
OMP_PROC_BIND pins it before spawning children, which inherit one physical core.
Correct profiles restore the parent main-thread affinity to 0..7 before spawning.
Short BF16: softmax 36.1%, pack 10.8%, QK 16.3%, PV 14.0%, mask 1.7% marginal.
Short fp32: softmax 25.1%, pack 14.2%, QK 17.6%, PV 8.2%, Q panel 4.7%.
Long BF16 1024:16:128: softmax 40.3%, pack 29.4%, QK 30.7%, PV 23.9%.
Marginal shares overlap and need not sum to 100%.
Hardware cycles/instructions are not exposed, even with sudo; software cpu-clock
sampling works. short-perf-report.txt includes the attend lambda, pack and GOMP.

## Experiments

1. Masked -inf stores instead of load/blend/store at causal boundary: exact outputs,
   but short rows 1.026/1.044 new/old, long mostly within control; rejected.
2. BF16 softmax converts 32 probabilities at once with cvtne2ps instead of two
   cvtneps+stores: exact outputs, dense 82/82. Initial short BF16 0.977,
   long BF16 0.970..1.000; fp32 bystander variation up to 1.035. Needs balanced repeat.
3. fp32 K/Q hi/lo panels convert 32 values together: exact outputs, dense 82/82.
   fp32 short 1.014 min / 0.975 median; 1024:8:64 0.963 / 0.958;
   1024:16:128 0.984 / 0.972. Not a resolved short-row fix; needs repeat.
4. Hoist full-32-key BF16 V packing out of tail branches: exact outputs;
   BF16 pooled ratios 0.996, 0.999, 1.020, 1.012, 0.992, 1.011 on the six probes.
   No resolved benefit; rejected.
5. Four independent row chains in softmax, with a four-row max/prob block: running.
