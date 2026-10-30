# Benchmark

`benchmarks/bench.py` times what an end user waits for: one call of each
library's Python API, on a handful of typical shapes. The README shows the
medians; this file has the method and every machine's full results.

## What is compared

| | ours | reference |
|---|---|---|
| prefill | `flash_attn_func(q, k, v, causal=...)` | `F.scaled_dot_product_attention` on the same `[batch, seq, heads, dim]` tensors (transposed views in, the output transposed back), `enable_gqa` for GQA |
| paged decode | `flash_attn_with_kvcache(q, k_cache, v_cache, k=k_new, v=v_new, cache_seqlens=..., page_table=...)` | vLLM's two calls per step: `reshape_and_cache` of the new token, then `cpu_attention_with_kv_cache` |

* vLLM is the `csrc/cpu` attention at the commit pinned in
  `scripts/build_vllm_ref.sh`: AMX for bf16 on AMX CPUs (c4-standard-*), VEC
  for every other x86 case, NEON on aarch64. Its scheduler metadata is built
  once per batch (once per decode step in vLLM), outside the timer, and its
  output is preallocated. Ours includes the output allocation, as a
  `flash_attn_with_kvcache` caller pays it.
* Decode rows with "append" write the new token's K/V in the timed call
  (self-attention); the Whisper cross-attention row reads an encoder's K/V only.
  Repeated calls rewrite the same slot, so the context does not grow.
* Each paged row runs each library at every page size it supports among 16 and
  32 and keeps its best: ours at both, vLLM at 32 on x86 (it takes only
  multiples of 32 there) and both on aarch64. The README marks the rows where
  our page 16 won with ᵖ¹⁶.
* Caches are staged before timing through each library's own API (ours: one
  `flash_attn_with_kvcache` append of the whole prompt per sequence).

## How it measures

* **Warmup and samples.** 3 warmup calls, then 10 timed calls per arm; one call
  per sample, timed with `time.perf_counter_ns` around the Python call.
* **Cold caches.** Before every call, warmup included, a write sweep over a
  buffer twice the size of the last-level cache evicts L2 and L3, so every call
  starts from DRAM like a layer of a model larger than the cache.
* **Interleaving.** Each round runs every arm once, the order rotated each round,
  so slow drift (frequency, a neighbour on the host) hits every arm alike.
* **Statistic.** The median decides, `ours / reference`, below 1 means we are
  faster; min and max are reported next to it.
* **Threads** = physical cores (`torch.set_num_threads`); both libraries run on
  torch's OpenMP pool.
* **Sanity.** Each row records the largest difference between our output and the
  reference's; it is not a gate (the tests are, against a float64 reference).

## Run it

On a fresh GCP VM, `scripts/box_provision.sh` installs what the image lacks (a
toolchain, `python3-dev`, `uv`) and the venv, then:

```bash
scripts/build_vllm_ref.sh            # vLLM reference -> ~/vllm_cpu_build/*.so (pinned commit)
V=~/vllm_cpu_build
# x86 without AMX (c4d-standard-*)
uv run python benchmarks/bench.py --json out.json \
    --vllm-bf16 $V/vllm_cpu_attn_bench.so --vllm-fp32 $V/vllm_cpu_attn_bench.so
# x86 with AMX (c4-standard-*)
uv run python benchmarks/bench.py --json out.json --vllm-isa-bf16 amx \
    --vllm-bf16 $V/vllm_cpu_attn_bench_amx.so --vllm-fp32 $V/vllm_cpu_attn_bench.so
# aarch64 (c4a-standard-*)
uv run python benchmarks/bench.py --json out.json --vllm-isa-bf16 neon --vllm-isa-fp32 neon \
    --vllm-bf16 $V/vllm_cpu_attn_bench.so --vllm-fp32 $V/vllm_cpu_attn_bench.so
python benchmarks/report.py --full out.json        # the tables below
```

`scripts/box_verify.sh BRANCH "ISAS" <bench args>` runs the tests at every tier
and then the benchmark. `--only <name>` runs a subset, `--dtype bf16|fp32` one
dtype, and `--compare <package dir>` times another build of the package in the
same process (an A/B of two builds; compare builds only inside one process,
since these machines drift 5-15% between processes). On a laptop or desktop,
pin the CPU governor to `performance` first (`scripts/pin_governor.sh`).

## Results

FULL_RESULTS
