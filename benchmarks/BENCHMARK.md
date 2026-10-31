# Benchmark

`benchmarks/bench.py` times what an end user waits for: one call of each
library's Python API, on a handful of typical MHA shapes (GQA/MQA are not
supported yet). Rows are labelled by shape in SDPA's legend: N batch, L query
length, S key/value length (`17..2047`: one length per sequence), H heads, E head
dimension; `causal` prefill, and `no append` decode (no K/V store in the call). The README shows the
medians; this file has the method and every machine's full results.

## What is compared

| | ours | reference |
|---|---|---|
| prefill | `flash_attn_func(q, k, v, causal=...)` | `F.scaled_dot_product_attention` on the same `[batch, seq, heads, dim]` tensors (transposed views in, the output transposed back) |
| paged decode | `flash_attn_with_kvcache(q, k_cache, v_cache, k=k_new, v=v_new, cache_seqlens=..., page_table=...)` | vLLM's two calls per step: `reshape_and_cache` of the new token, then `cpu_attention_with_kv_cache` |

* vLLM is the `csrc/cpu` attention at the commit pinned in
  `scripts/build_vllm_ref.sh`: AMX for bf16 on AMX CPUs (c4-standard-*), VEC
  for every other x86 case, NEON on aarch64. Its scheduler metadata is built
  once per batch (once per decode step in vLLM), outside the timer, and its
  output is preallocated. Ours includes the output allocation, as a
  `flash_attn_with_kvcache` caller pays it.
* Decode rows with "append" write the new token's K/V in the timed call
  (self-attention); the `no append` row reads a pre-staged (encoder) K/V only.
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

### c4-standard-8

INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, 4 threads, `b3e10f8`, torch 2.14.1+cpu, Mojo 1.2.0 (2016d0cd); vLLM ISA bf16 `amx`, fp32 `vec`.

| shape | dtype | ours | sdpa | ours / ref |
|---|---|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | bf16 | 151 (147-167) | 378 (357-412) | 0.40 |
| N=1 L=128 S=128 H=16 E=64 causal | fp32 | 266 (255-286) | 470 (437-631) | 0.57 |
| N=1 L=1024 S=1024 H=16 E=64 causal | bf16 | 1,392 (1,343-2,183) | 4,000 (3,378-4,807) | 0.35 |
| N=1 L=1024 S=1024 H=16 E=64 causal | fp32 | 3,379 (2,571-4,236) | 8,684 (7,754-9,478) | 0.39 |
| N=1 L=2048 S=2048 H=32 E=128 causal | bf16 | 12,235 (12,062-12,485) | 30,091 (29,120-30,228) | 0.41 |
| N=1 L=2048 S=2048 H=32 E=128 causal | fp32 | 44,286 (43,720-44,729) | 89,705 (88,986-90,512) | 0.49 |
| N=1 L=4096 S=4096 H=32 E=128 causal | bf16 | 57,631 (57,081-58,146) | 102,633 (101,592-103,915) | 0.56 |
| N=1 L=4096 S=4096 H=32 E=128 causal | fp32 | 142,794 (139,717-143,867) | 307,336 (305,599-309,302) | 0.46 |
| N=8 L=512 S=512 H=16 E=64 | bf16 | 4,471 (4,294-4,561) | 7,985 (7,774-8,077) | 0.56 |
| N=8 L=512 S=512 H=16 E=64 | fp32 | 12,331 (9,056-15,833) | 30,277 (28,385-31,300) | 0.41 |
| N=1 L=1500 S=1500 H=20 E=64 | bf16 | 4,857 (4,814-4,905) | 9,970 (9,579-10,351) | 0.49 |
| N=1 L=1500 S=1500 H=20 E=64 | fp32 | 10,812 (10,686-11,298) | 26,396 (25,899-28,131) | 0.41 |
| N=8 L=256 S=256 H=16 E=32 | bf16 | 1,432 (1,407-1,477) | 2,579 (2,498-2,651) | 0.56 |
| N=8 L=256 S=256 H=16 E=32 | fp32 | 2,374 (2,317-2,426) | 4,178 (4,055-4,367) | 0.57 |

| shape | dtype | ours p16 | ours p32 | vllm p32 | ours / ref |
|---|---|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | bf16 | 78.6 (73.5-83.8) | 79.0 (71.9-96.7) | 104 (95.8-111) | 0.76 |
| N=1 L=1 S=256 H=16 E=64 | fp32 | 95.4 (88.7-106) | 97.9 (91.0-110) | 129 (121-151) | 0.74 |
| N=1 L=1 S=4096 H=32 E=128 | bf16 | 1,747 (1,685-1,899) | 1,751 (1,711-1,828) | 1,613 (1,571-1,689) | 1.08 |
| N=1 L=1 S=4096 H=32 E=128 | fp32 | 2,755 (2,679-2,809) | 2,978 (2,928-3,077) | 2,781 (2,701-2,832) | 0.99 |
| N=8 L=1 S=1024 H=16 E=64 | bf16 | 1,056 (985-1,107) | 983 (898-1,013) | 978 (950-1,045) | 1.01 |
| N=8 L=1 S=1024 H=16 E=64 | fp32 | 1,518 (1,457-1,567) | 1,606 (1,533-1,629) | 1,529 (1,496-1,597) | 0.99 |
| N=8 L=1 S=1024 H=16 E=32 | bf16 | 582 (544-621) | 570 (543-597) | 589 (531-632) | 0.97 |
| N=8 L=1 S=1024 H=16 E=32 | fp32 | 846 (801-913) | 840 (807-876) | 879 (851-935) | 0.95 |
| N=8 L=1 S=4096 H=32 E=128 | bf16 | 13,282 (12,971-13,559) | 12,992 (12,405-13,578) | 11,496 (10,993-11,872) | 1.13 |
| N=8 L=1 S=4096 H=32 E=128 | fp32 | 21,957 (21,480-22,352) | 23,354 (22,993-24,145) | 22,451 (22,104-23,180) | 0.98 |
| N=8 L=1 S=17..2047 H=16 E=64 | bf16 | 684 (667-723) | 654 (620-707) | 666 (643-713) | 0.98 |
| N=8 L=1 S=17..2047 H=16 E=64 | fp32 | 973 (911-1,027) | 1,006 (986-1,026) | 1,037 (1,023-1,075) | 0.94 |
| N=8 L=1 S=1500 H=20 E=64 no append | bf16 | 1,661 (1,527-1,796) | 1,624 (1,518-1,710) | 1,643 (1,580-1,706) | 0.99 |
| N=8 L=1 S=1500 H=20 E=64 no append | fp32 | 2,553 (2,435-2,604) | 2,731 (2,638-2,820) | 2,627 (2,593-2,691) | 0.97 |
| N=32 L=1 S=512 H=16 E=64 | bf16 | 1,868 (1,799-2,050) | 1,812 (1,756-1,887) | 1,848 (1,802-1,892) | 0.98 |
| N=32 L=1 S=512 H=16 E=64 | fp32 | 2,837 (2,768-2,919) | 3,034 (2,976-3,168) | 3,083 (2,979-3,210) | 0.92 |
| N=32 L=1 S=1024 H=32 E=128 | bf16 | 13,497 (13,062-14,027) | 12,926 (12,538-13,300) | 11,572 (11,372-12,062) | 1.12 |
| N=32 L=1 S=1024 H=32 E=128 | fp32 | 21,561 (21,104-21,846) | 23,825 (23,503-24,219) | 22,739 (22,562-23,047) | 0.95 |

### c4-standard-16

INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, 8 threads, `b3e10f8`, torch 2.14.1+cpu, Mojo 1.2.0 (2016d0cd); vLLM ISA bf16 `amx`, fp32 `vec`.

| shape | dtype | ours | sdpa | ours / ref |
|---|---|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | bf16 | 87.2 (67.3-111) | 223 (174-240) | 0.39 |
| N=1 L=128 S=128 H=16 E=64 causal | fp32 | 151 (141-179) | 284 (258-632) | 0.53 |
| N=1 L=1024 S=1024 H=16 E=64 causal | bf16 | 771 (689-2,109) | 2,681 (2,497-3,952) | 0.29 |
| N=1 L=1024 S=1024 H=16 E=64 causal | fp32 | 1,490 (1,397-1,607) | 3,883 (3,790-4,032) | 0.38 |
| N=1 L=2048 S=2048 H=32 E=128 causal | bf16 | 17,273 (5,708-17,563) | 20,947 (14,373-21,649) | 0.82 |
| N=1 L=2048 S=2048 H=32 E=128 causal | fp32 | 31,269 (30,965-31,676) | 49,630 (48,549-50,340) | 0.63 |
| N=1 L=4096 S=4096 H=32 E=128 causal | bf16 | 43,000 (42,238-44,016) | 63,228 (61,829-63,791) | 0.68 |
| N=1 L=4096 S=4096 H=32 E=128 causal | fp32 | 84,263 (83,477-85,674) | 160,105 (159,483-161,246) | 0.53 |
| N=8 L=512 S=512 H=16 E=64 | bf16 | 2,151 (2,021-2,263) | 4,202 (3,898-6,314) | 0.51 |
| N=8 L=512 S=512 H=16 E=64 | fp32 | 4,576 (4,367-4,672) | 13,777 (13,448-14,103) | 0.33 |
| N=1 L=1500 S=1500 H=20 E=64 | bf16 | 2,741 (2,612-3,140) | 5,033 (4,796-5,246) | 0.54 |
| N=1 L=1500 S=1500 H=20 E=64 | fp32 | 5,826 (5,779-6,087) | 13,056 (12,859-13,201) | 0.45 |
| N=8 L=256 S=256 H=16 E=32 | bf16 | 695 (641-740) | 1,337 (1,126-1,416) | 0.52 |
| N=8 L=256 S=256 H=16 E=32 | fp32 | 1,263 (1,119-1,304) | 2,116 (1,989-2,174) | 0.60 |

| shape | dtype | ours p16 | ours p32 | vllm p32 | ours / ref |
|---|---|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | bf16 | 61.3 (54.6-66.5) | 56.9 (52.9-70.0) | 78.5 (72.4-98.5) | 0.73 |
| N=1 L=1 S=256 H=16 E=64 | fp32 | 73.0 (66.5-78.4) | 75.3 (63.1-97.9) | 93.5 (83.7-110) | 0.78 |
| N=1 L=1 S=4096 H=32 E=128 | bf16 | 1,000 (938-1,072) | 986 (931-1,080) | 929 (860-972) | 1.06 |
| N=1 L=1 S=4096 H=32 E=128 | fp32 | 1,704 (1,627-1,773) | 1,726 (1,664-1,785) | 1,551 (1,469-1,607) | 1.10 |
| N=8 L=1 S=1024 H=16 E=64 | bf16 | 626 (584-644) | 608 (564-647) | 573 (541-617) | 1.06 |
| N=8 L=1 S=1024 H=16 E=64 | fp32 | 898 (846-970) | 932 (910-968) | 941 (868-953) | 0.95 |
| N=8 L=1 S=1024 H=16 E=32 | bf16 | 383 (358-405) | 386 (354-406) | 403 (360-416) | 0.95 |
| N=8 L=1 S=1024 H=16 E=32 | fp32 | 543 (496-577) | 531 (501-545) | 533 (494-550) | 1.00 |
| N=8 L=1 S=4096 H=32 E=128 | bf16 | 7,226 (7,105-7,613) | 7,077 (6,812-7,736) | 6,298 (6,140-6,373) | 1.12 |
| N=8 L=1 S=4096 H=32 E=128 | fp32 | 11,732 (11,396-12,007) | 12,637 (12,168-12,972) | 11,966 (11,717-12,049) | 0.98 |
| N=8 L=1 S=17..2047 H=16 E=64 | bf16 | 409 (387-444) | 406 (368-510) | 457 (371-481) | 0.89 |
| N=8 L=1 S=17..2047 H=16 E=64 | fp32 | 564 (547-601) | 601 (566-617) | 626 (556-662) | 0.90 |
| N=8 L=1 S=1500 H=20 E=64 no append | bf16 | 940 (871-981) | 952 (897-987) | 875 (800-931) | 1.07 |
| N=8 L=1 S=1500 H=20 E=64 no append | fp32 | 1,391 (1,340-1,443) | 1,485 (1,391-1,514) | 1,398 (1,364-1,498) | 1.00 |
| N=32 L=1 S=512 H=16 E=64 | bf16 | 1,063 (1,008-1,161) | 1,036 (980-1,228) | 1,047 (990-1,106) | 0.99 |
| N=32 L=1 S=512 H=16 E=64 | fp32 | 1,585 (1,548-1,671) | 1,645 (1,599-1,675) | 1,627 (1,559-1,733) | 0.97 |
| N=32 L=1 S=1024 H=32 E=128 | bf16 | 7,412 (7,101-7,726) | 7,268 (6,988-7,507) | 6,704 (6,632-6,910) | 1.08 |
| N=32 L=1 S=1024 H=32 E=128 | fp32 | 11,281 (10,979-11,696) | 12,517 (11,760-12,867) | 12,072 (11,916-12,413) | 0.93 |

### c4a-standard-8

Neoverse-V2, 8 threads, `b3e10f8`, torch 2.14.1+cpu, Mojo 1.2.0 (2016d0cd); vLLM ISA bf16 `neon`, fp32 `neon`.

| shape | dtype | ours | sdpa | ours / ref |
|---|---|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | bf16 | 63.0 (52.7-66.5) | 158 (134-168) | 0.40 |
| N=1 L=128 S=128 H=16 E=64 causal | fp32 | 114 (107-122) | 365 (338-382) | 0.31 |
| N=1 L=1024 S=1024 H=16 E=64 causal | bf16 | 1,662 (1,640-1,674) | 3,357 (3,338-3,373) | 0.49 |
| N=1 L=1024 S=1024 H=16 E=64 causal | fp32 | 6,795 (4,459-6,894) | 9,403 (9,306-9,423) | 0.72 |
| N=1 L=2048 S=2048 H=32 E=128 causal | bf16 | 20,991 (20,946-21,275) | 35,406 (35,322-35,542) | 0.59 |
| N=1 L=2048 S=2048 H=32 E=128 causal | fp32 | 64,038 (63,470-65,750) | 86,484 (86,433-86,593) | 0.74 |
| N=1 L=4096 S=4096 H=32 E=128 causal | bf16 | 84,879 (84,336-94,218) | 126,181 (125,813-131,486) | 0.67 |
| N=1 L=4096 S=4096 H=32 E=128 causal | fp32 | 251,350 (250,503-252,355) | 301,490 (300,670-316,054) | 0.83 |
| N=8 L=512 S=512 H=16 E=64 | bf16 | 6,410 (6,276-6,557) | 9,729 (9,389-9,791) | 0.66 |
| N=8 L=512 S=512 H=16 E=64 | fp32 | 17,309 (17,288-17,371) | 19,461 (19,305-19,553) | 0.89 |
| N=1 L=1500 S=1500 H=20 E=64 | bf16 | 8,419 (8,393-8,437) | 11,732 (11,651-12,145) | 0.72 |
| N=1 L=1500 S=1500 H=20 E=64 | fp32 | 23,524 (23,322-23,565) | 24,572 (23,858-24,612) | 0.96 |
| N=8 L=256 S=256 H=16 E=32 | bf16 | 1,138 (1,132-1,149) | 1,757 (1,744-1,766) | 0.65 |
| N=8 L=256 S=256 H=16 E=32 | fp32 | 2,736 (2,708-2,756) | 3,013 (2,994-3,030) | 0.91 |

| shape | dtype | ours p16 | vllm p16 | ours p32 | vllm p32 | ours / ref |
|---|---|---:|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | bf16 | 34.0 (26.1-38.1) | 57.0 (43.1-82.5) | 32.3 (23.1-37.3) | 52.0 (38.9-61.2) | 0.62 |
| N=1 L=1 S=256 H=16 E=64 | fp32 | 44.0 (37.8-48.4) | 66.0 (51.5-70.4) | 45.4 (35.9-48.8) | 65.4 (52.9-84.0) | 0.67 |
| N=1 L=1 S=4096 H=32 E=128 | bf16 | 741 (721-863) | 836 (816-849) | 807 (778-945) | 697 (677-717) | 1.06 |
| N=1 L=1 S=4096 H=32 E=128 | fp32 | 1,610 (1,575-1,633) | 1,454 (1,406-1,531) | 1,257 (1,236-1,268) | 1,437 (1,387-1,581) | 0.87 |
| N=8 L=1 S=1024 H=16 E=64 | bf16 | 424 (412-435) | 530 (499-540) | 402 (388-412) | 493 (474-510) | 0.82 |
| N=8 L=1 S=1024 H=16 E=64 | fp32 | 799 (783-815) | 882 (858-906) | 675 (661-680) | 836 (810-861) | 0.81 |
| N=8 L=1 S=1024 H=16 E=32 | bf16 | 195 (185-218) | 255 (241-269) | 212 (196-237) | 279 (261-291) | 0.76 |
| N=8 L=1 S=1024 H=16 E=32 | fp32 | 460 (446-472) | 523 (504-542) | 409 (398-434) | 496 (469-507) | 0.82 |
| N=8 L=1 S=4096 H=32 E=128 | bf16 | 5,476 (5,314-6,411) | 6,552 (6,514-7,252) | 6,575 (6,501-6,805) | 5,106 (5,089-5,138) | 1.07 |
| N=8 L=1 S=4096 H=32 E=128 | fp32 | 10,743 (10,656-16,154) | 10,704 (10,511-11,516) | 10,112 (9,959-10,218) | 10,151 (9,838-10,352) | 1.00 |
| N=8 L=1 S=17..2047 H=16 E=64 | bf16 | 303 (290-316) | 382 (371-406) | 264 (254-283) | 329 (319-358) | 0.80 |
| N=8 L=1 S=17..2047 H=16 E=64 | fp32 | 526 (506-541) | 590 (569-612) | 440 (429-462) | 570 (549-589) | 0.77 |
| N=8 L=1 S=1500 H=20 E=64 no append | bf16 | 681 (656-691) | 857 (820-873) | 635 (612-655) | 755 (729-765) | 0.84 |
| N=8 L=1 S=1500 H=20 E=64 no append | fp32 | 1,465 (1,407-1,480) | 1,558 (1,497-1,582) | 1,159 (1,138-1,178) | 1,313 (1,262-1,357) | 0.88 |
| N=32 L=1 S=512 H=16 E=64 | bf16 | 748 (691-770) | 1,026 (975-1,040) | 710 (696-719) | 958 (922-967) | 0.74 |
| N=32 L=1 S=512 H=16 E=64 | fp32 | 1,565 (1,554-1,581) | 1,773 (1,734-1,804) | 1,288 (1,283-1,318) | 1,599 (1,563-1,637) | 0.81 |
| N=32 L=1 S=1024 H=32 E=128 | bf16 | 5,496 (5,270-7,067) | 6,606 (6,468-7,688) | 6,313 (6,049-6,459) | 5,263 (5,214-5,627) | 1.04 |
| N=32 L=1 S=1024 H=32 E=128 | fp32 | 10,757 (10,627-10,952) | 10,793 (10,592-11,054) | 10,110 (9,905-10,230) | 9,880 (9,335-10,183) | 1.02 |

### c4a-standard-16

Neoverse-V2, 16 threads, `b3e10f8`, torch 2.14.1+cpu, Mojo 1.2.0 (2016d0cd); vLLM ISA bf16 `neon`, fp32 `neon`.

| shape | dtype | ours | sdpa | ours / ref |
|---|---|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | bf16 | 41.9 (31.9-51.6) | 102 (86.0-113) | 0.41 |
| N=1 L=128 S=128 H=16 E=64 causal | fp32 | 68.8 (58.3-78.4) | 205 (188-221) | 0.34 |
| N=1 L=1024 S=1024 H=16 E=64 causal | bf16 | 862 (837-885) | 1,740 (1,712-1,753) | 0.50 |
| N=1 L=1024 S=1024 H=16 E=64 causal | fp32 | 2,287 (2,264-2,306) | 4,704 (4,680-4,731) | 0.49 |
| N=1 L=2048 S=2048 H=32 E=128 causal | bf16 | 10,488 (10,418-10,893) | 18,034 (18,024-18,101) | 0.58 |
| N=1 L=2048 S=2048 H=32 E=128 causal | fp32 | 31,940 (31,873-32,541) | 43,829 (43,769-44,382) | 0.73 |
| N=1 L=4096 S=4096 H=32 E=128 causal | bf16 | 42,724 (42,553-42,947) | 64,006 (63,587-64,180) | 0.67 |
| N=1 L=4096 S=4096 H=32 E=128 causal | fp32 | 127,116 (127,004-127,213) | 152,891 (151,620-158,644) | 0.83 |
| N=8 L=512 S=512 H=16 E=64 | bf16 | 3,173 (3,156-3,179) | 4,791 (4,747-4,811) | 0.66 |
| N=8 L=512 S=512 H=16 E=64 | fp32 | 8,736 (8,713-9,139) | 9,702 (9,659-9,734) | 0.90 |
| N=1 L=1500 S=1500 H=20 E=64 | bf16 | 4,376 (4,373-4,404) | 6,267 (6,232-6,596) | 0.70 |
| N=1 L=1500 S=1500 H=20 E=64 | fp32 | 12,329 (12,183-12,447) | 13,777 (12,865-13,858) | 0.89 |
| N=8 L=256 S=256 H=16 E=32 | bf16 | 588 (574-590) | 930 (905-955) | 0.63 |
| N=8 L=256 S=256 H=16 E=32 | fp32 | 1,387 (1,377-1,392) | 1,569 (1,548-1,585) | 0.88 |

| shape | dtype | ours p16 | vllm p16 | ours p32 | vllm p32 | ours / ref |
|---|---|---:|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | bf16 | 28.8 (22.3-33.2) | 44.1 (33.5-57.2) | 31.6 (23.7-35.2) | 45.9 (33.6-62.1) | 0.65 |
| N=1 L=1 S=256 H=16 E=64 | fp32 | 34.3 (27.8-45.0) | 52.6 (42.8-58.8) | 33.5 (27.2-41.1) | 50.6 (42.6-56.5) | 0.66 |
| N=1 L=1 S=4096 H=32 E=128 | bf16 | 455 (442-481) | 528 (501-543) | 484 (470-494) | 434 (416-458) | 1.05 |
| N=1 L=1 S=4096 H=32 E=128 | fp32 | 951 (928-975) | 911 (881-928) | 775 (764-791) | 922 (843-951) | 0.85 |
| N=8 L=1 S=1024 H=16 E=64 | bf16 | 234 (231-254) | 341 (305-360) | 257 (247-272) | 298 (293-314) | 0.78 |
| N=8 L=1 S=1024 H=16 E=64 | fp32 | 490 (460-501) | 556 (533-1,159) | 450 (430-476) | 510 (478-1,415) | 0.88 |
| N=8 L=1 S=1024 H=16 E=32 | bf16 | 129 (125-138) | 184 (170-195) | 151 (138-159) | 175 (155-180) | 0.74 |
| N=8 L=1 S=1024 H=16 E=32 | fp32 | 252 (240-260) | 318 (305-326) | 260 (248-272) | 306 (290-317) | 0.83 |
| N=8 L=1 S=4096 H=32 E=128 | bf16 | 3,618 (3,478-3,781) | 3,752 (3,716-3,783) | 4,031 (3,901-4,132) | 2,960 (2,927-2,990) | 1.22 |
| N=8 L=1 S=4096 H=32 E=128 | fp32 | 6,759 (6,710-6,779) | 6,431 (6,325-6,502) | 6,113 (6,078-6,151) | 5,875 (5,733-5,974) | 1.04 |
| N=8 L=1 S=17..2047 H=16 E=64 | bf16 | 187 (178-216) | 273 (249-282) | 192 (188-205) | 250 (237-264) | 0.75 |
| N=8 L=1 S=17..2047 H=16 E=64 | fp32 | 365 (349-378) | 387 (366-402) | 305 (301-310) | 379 (358-395) | 0.81 |
| N=8 L=1 S=1500 H=20 E=64 no append | bf16 | 410 (396-430) | 486 (453-508) | 410 (399-422) | 460 (439-488) | 0.89 |
| N=8 L=1 S=1500 H=20 E=64 no append | fp32 | 851 (830-872) | 821 (779-836) | 740 (728-758) | 800 (771-815) | 0.92 |
| N=32 L=1 S=512 H=16 E=64 | bf16 | 466 (453-490) | 568 (552-588) | 442 (433-458) | 555 (534-574) | 0.80 |
| N=32 L=1 S=512 H=16 E=64 | fp32 | 940 (922-965) | 997 (968-1,027) | 849 (841-855) | 959 (921-976) | 0.89 |
| N=32 L=1 S=1024 H=32 E=128 | bf16 | 3,525 (3,433-3,558) | 3,947 (3,820-3,981) | 3,549 (3,445-3,644) | 3,099 (3,086-3,127) | 1.14 |
| N=32 L=1 S=1024 H=32 E=128 | fp32 | 6,654 (6,570-6,685) | 6,516 (6,424-6,591) | 5,502 (5,420-5,550) | 5,940 (5,800-6,188) | 0.93 |

### c4d-standard-8

AMD EPYC 9B45, 4 threads, `b3e10f8`, torch 2.14.1+cpu, Mojo 1.2.0 (2016d0cd); vLLM ISA bf16 `vec`, fp32 `vec`.

| shape | dtype | ours | sdpa | ours / ref |
|---|---|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | bf16 | 72.2 (66.0-87.2) | 321 (300-337) | 0.22 |
| N=1 L=128 S=128 H=16 E=64 causal | fp32 | 107 (98.7-110) | 324 (314-430) | 0.33 |
| N=1 L=1024 S=1024 H=16 E=64 causal | bf16 | 2,007 (1,711-2,111) | 9,607 (9,420-9,892) | 0.21 |
| N=1 L=1024 S=1024 H=16 E=64 causal | fp32 | 3,332 (3,063-3,749) | 9,483 (9,141-9,935) | 0.35 |
| N=1 L=2048 S=2048 H=32 E=128 causal | bf16 | 26,193 (24,575-27,276) | 118,363 (117,133-119,138) | 0.22 |
| N=1 L=2048 S=2048 H=32 E=128 causal | fp32 | 50,079 (49,466-50,505) | 114,460 (113,975-115,094) | 0.44 |
| N=1 L=4096 S=4096 H=32 E=128 causal | bf16 | 102,773 (102,277-103,272) | 430,218 (429,352-431,585) | 0.24 |
| N=1 L=4096 S=4096 H=32 E=128 causal | fp32 | 181,733 (181,108-182,309) | 405,089 (403,190-405,846) | 0.45 |
| N=8 L=512 S=512 H=16 E=64 | bf16 | 6,723 (6,695-6,744) | 26,553 (26,398-26,963) | 0.25 |
| N=8 L=512 S=512 H=16 E=64 | fp32 | 12,214 (12,066-12,313) | 26,862 (26,713-26,882) | 0.45 |
| N=1 L=1500 S=1500 H=20 E=64 | bf16 | 8,432 (8,394-8,455) | 34,403 (34,247-34,708) | 0.25 |
| N=1 L=1500 S=1500 H=20 E=64 | fp32 | 15,212 (15,097-15,546) | 32,200 (32,096-32,301) | 0.47 |
| N=8 L=256 S=256 H=16 E=32 | bf16 | 1,057 (1,004-1,071) | 4,067 (4,033-4,084) | 0.26 |
| N=8 L=256 S=256 H=16 E=32 | fp32 | 1,704 (1,684-1,716) | 4,055 (4,045-4,081) | 0.42 |

| shape | dtype | ours p16 | ours p32 | vllm p32 | ours / ref |
|---|---|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | bf16 | 47.4 (43.6-65.5) | 46.8 (44.1-49.7) | 62.8 (59.1-73.3) | 0.75 |
| N=1 L=1 S=256 H=16 E=64 | fp32 | 77.3 (72.7-83.6) | 76.1 (71.3-92.9) | 87.0 (83.1-96.2) | 0.87 |
| N=1 L=1 S=4096 H=32 E=128 | bf16 | 1,514 (1,411-1,592) | 1,497 (1,359-1,591) | 1,626 (1,555-1,715) | 0.92 |
| N=1 L=1 S=4096 H=32 E=128 | fp32 | 2,821 (2,683-2,961) | 2,812 (2,734-2,894) | 2,880 (2,743-2,941) | 0.98 |
| N=8 L=1 S=1024 H=16 E=64 | bf16 | 776 (701-808) | 765 (701-779) | 840 (817-871) | 0.91 |
| N=8 L=1 S=1024 H=16 E=64 | fp32 | 1,417 (1,348-1,536) | 1,422 (1,333-1,463) | 1,450 (1,367-1,524) | 0.98 |
| N=8 L=1 S=1024 H=16 E=32 | bf16 | 421 (391-430) | 413 (382-429) | 473 (440-480) | 0.87 |
| N=8 L=1 S=1024 H=16 E=32 | fp32 | 772 (745-789) | 774 (741-804) | 800 (775-824) | 0.97 |
| N=8 L=1 S=4096 H=32 E=128 | bf16 | 12,047 (11,604-12,133) | 11,808 (11,443-12,132) | 13,362 (12,908-13,660) | 0.88 |
| N=8 L=1 S=4096 H=32 E=128 | fp32 | 22,112 (21,332-23,878) | 22,482 (21,309-22,947) | 23,472 (22,526-23,952) | 0.94 |
| N=8 L=1 S=17..2047 H=16 E=64 | bf16 | 480 (459-502) | 481 (442-519) | 578 (527-599) | 0.83 |
| N=8 L=1 S=17..2047 H=16 E=64 | fp32 | 887 (835-922) | 883 (847-968) | 963 (912-977) | 0.92 |
| N=8 L=1 S=1500 H=20 E=64 no append | bf16 | 1,495 (1,405-1,607) | 1,460 (1,387-1,505) | 1,594 (1,527-1,788) | 0.92 |
| N=8 L=1 S=1500 H=20 E=64 no append | fp32 | 2,705 (2,503-2,791) | 2,623 (2,476-2,760) | 2,761 (2,689-2,865) | 0.95 |
| N=32 L=1 S=512 H=16 E=64 | bf16 | 1,460 (1,418-1,548) | 1,519 (1,418-1,594) | 1,718 (1,679-1,762) | 0.85 |
| N=32 L=1 S=512 H=16 E=64 | fp32 | 2,889 (2,705-3,086) | 2,857 (2,736-2,886) | 2,971 (2,859-3,135) | 0.96 |
| N=32 L=1 S=1024 H=32 E=128 | bf16 | 12,294 (11,752-12,836) | 12,182 (11,781-12,552) | 14,390 (13,662-15,610) | 0.85 |
| N=32 L=1 S=1024 H=32 E=128 | fp32 | 22,556 (21,681-23,369) | 22,372 (21,970-22,982) | 23,882 (23,403-24,391) | 0.94 |

### c4d-standard-16

AMD EPYC 9B45, 8 threads, `b3e10f8`, torch 2.14.1+cpu, Mojo 1.2.0 (2016d0cd); vLLM ISA bf16 `vec`, fp32 `vec`.

| shape | dtype | ours | sdpa | ours / ref |
|---|---|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | bf16 | 56.5 (49.6-60.9) | 220 (187-252) | 0.26 |
| N=1 L=128 S=128 H=16 E=64 causal | fp32 | 88.9 (78.4-105) | 210 (193-331) | 0.42 |
| N=1 L=1024 S=1024 H=16 E=64 causal | bf16 | 933 (885-955) | 5,018 (4,801-5,111) | 0.19 |
| N=1 L=1024 S=1024 H=16 E=64 causal | fp32 | 1,681 (1,631-2,423) | 5,282 (4,789-5,622) | 0.32 |
| N=1 L=2048 S=2048 H=32 E=128 causal | bf16 | 14,662 (12,755-16,034) | 61,497 (59,531-62,438) | 0.24 |
| N=1 L=2048 S=2048 H=32 E=128 causal | fp32 | 28,363 (28,106-28,736) | 60,031 (59,806-61,065) | 0.47 |
| N=1 L=4096 S=4096 H=32 E=128 causal | bf16 | 56,447 (55,898-56,799) | 220,596 (220,067-221,011) | 0.26 |
| N=1 L=4096 S=4096 H=32 E=128 causal | fp32 | 94,258 (93,743-94,643) | 208,321 (207,954-209,050) | 0.45 |
| N=8 L=512 S=512 H=16 E=64 | bf16 | 3,548 (3,500-3,582) | 13,454 (13,330-13,579) | 0.26 |
| N=8 L=512 S=512 H=16 E=64 | fp32 | 6,341 (6,239-9,537) | 13,743 (13,700-14,496) | 0.46 |
| N=1 L=1500 S=1500 H=20 E=64 | bf16 | 4,541 (4,348-4,612) | 17,416 (17,112-17,573) | 0.26 |
| N=1 L=1500 S=1500 H=20 E=64 | fp32 | 7,481 (7,348-7,737) | 16,460 (16,259-16,548) | 0.45 |
| N=8 L=256 S=256 H=16 E=32 | bf16 | 522 (504-544) | 2,087 (2,068-2,180) | 0.25 |
| N=8 L=256 S=256 H=16 E=32 | fp32 | 856 (848-874) | 2,070 (2,037-2,172) | 0.41 |

| shape | dtype | ours p16 | ours p32 | vllm p32 | ours / ref |
|---|---|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | bf16 | 44.4 (42.4-45.9) | 41.6 (37.5-50.6) | 54.0 (51.5-64.0) | 0.77 |
| N=1 L=1 S=256 H=16 E=64 | fp32 | 67.5 (63.1-83.2) | 67.5 (64.2-81.8) | 75.3 (72.5-76.6) | 0.90 |
| N=1 L=1 S=4096 H=32 E=128 | bf16 | 1,281 (1,262-1,303) | 1,262 (1,258-1,277) | 1,298 (1,293-1,345) | 0.97 |
| N=1 L=1 S=4096 H=32 E=128 | fp32 | 2,467 (2,447-2,482) | 2,460 (2,452-2,483) | 2,480 (2,467-2,505) | 0.99 |
| N=8 L=1 S=1024 H=16 E=64 | bf16 | 664 (659-677) | 666 (655-670) | 704 (685-718) | 0.94 |
| N=8 L=1 S=1024 H=16 E=64 | fp32 | 1,262 (1,249-1,271) | 1,257 (1,250-1,271) | 1,281 (1,267-1,300) | 0.98 |
| N=8 L=1 S=1024 H=16 E=32 | bf16 | 373 (352-378) | 361 (353-370) | 387 (379-412) | 0.93 |
| N=8 L=1 S=1024 H=16 E=32 | fp32 | 674 (659-679) | 676 (661-683) | 691 (674-719) | 0.98 |
| N=8 L=1 S=4096 H=32 E=128 | bf16 | 9,852 (9,775-9,937) | 9,834 (9,777-9,889) | 10,413 (10,329-10,479) | 0.94 |
| N=8 L=1 S=4096 H=32 E=128 | fp32 | 19,274 (19,254-19,317) | 19,278 (19,267-19,335) | 19,504 (19,469-19,528) | 0.99 |
| N=8 L=1 S=17..2047 H=16 E=64 | bf16 | 427 (415-436) | 423 (408-436) | 471 (466-504) | 0.90 |
| N=8 L=1 S=17..2047 H=16 E=64 | fp32 | 809 (791-815) | 803 (789-807) | 832 (819-845) | 0.96 |
| N=8 L=1 S=1500 H=20 E=64 no append | bf16 | 1,171 (1,155-1,193) | 1,166 (1,158-1,178) | 1,194 (1,180-1,215) | 0.98 |
| N=8 L=1 S=1500 H=20 E=64 no append | fp32 | 2,255 (2,250-2,267) | 2,258 (2,249-2,269) | 2,273 (2,264-2,286) | 0.99 |
| N=32 L=1 S=512 H=16 E=64 | bf16 | 1,289 (1,273-1,318) | 1,291 (1,275-1,306) | 1,337 (1,316-1,385) | 0.96 |
| N=32 L=1 S=512 H=16 E=64 | fp32 | 2,543 (2,500-2,676) | 2,516 (2,470-2,628) | 2,623 (2,583-2,789) | 0.96 |
| N=32 L=1 S=1024 H=32 E=128 | bf16 | 9,784 (9,738-10,139) | 9,766 (9,734-10,003) | 10,355 (10,317-10,561) | 0.94 |
| N=32 L=1 S=1024 H=32 E=128 | fp32 | 19,329 (19,303-19,361) | 19,302 (19,279-19,345) | 19,717 (19,682-19,745) | 0.98 |

<!-- OPENVINO RESULTS START -->
## OpenVINO comparison

OpenVINO 2026.4.1 CPU execution, compared in the same process on identical inputs. Ratios are **ours / OpenVINO** (below 1 favors ours); ᵖ¹⁶ marks our winning page size. The existing torch/vLLM measurements above are unchanged.

`benchmarks/bench_openvino.py` reuses the shapes and timer from `bench.py`: 3 warmups, 10 rotated/interleaved samples, a write sweep of twice LLC before every call, threads = physical cores. The reported median is the upper-middle sample, matching the existing `bench.py` convention. Ryzen's governor is pinned to performance and restored afterwards. `OMP_WAIT_POLICY=PASSIVE` avoids idle OpenMP workers competing with OpenVINO's separate CPU worker pool.

OpenVINO runs synchronous `InferRequest.infer(share_outputs=True)` on a compiled one-attention graph, one stream, explicit f32/bf16 inference and cache precision, no quantization, CPU pinning and hyperthreading disabled. Torch-backed input/output buffers are bound once (bf16 is shared as raw 16-bit storage), so Python/NumPy conversion, compilation, initial cache staging, and metadata construction are outside timing. OpenVINO reuses its output; ours includes output allocation, as in the existing vLLM comparison. Dense BLHD/BHLD transposes are inside the compiled graph, not pre-packed outside the timer. Paged caches are staged through each library's own writer; append rows rewrite the same final slot on every call; read-only rows use `write_kv_cache=false`. OpenVINO requires page 32; ours runs both 16 and 32. SDPA runs both static and dynamic-sequence graphs and reports OpenVINO's faster arm per shape.

Every row is checked against eager float64 attention before timing (64 query rows at a time to bound memory). Both outputs must be finite and within 2e-4 fp32 / 3e-2 bf16 max absolute error; raw JSON additionally records the README's stricter per-operation limits, all samples, actual runtime graphs/precisions, and unsupported-operation diagnostics. Static SDPA may execute as fused CPU subgraphs rather than a node named SDPA; the runtime graph is the authority. These are API latency results, not isolated kernel times.

**Axion precision limitation:** bf16 labels describe input/output storage. OpenVINO's Axion SDPA subgraphs execute in fp32 (conversions are timed), unlike our native bf16 path. Its paged f32 fallback did not preserve writes to shared bf16 caches: the initial attempt failed float64 validation and a cache-mutation diagnostic is retained with the raw results. Those paged bf16 timings are excluded, not replaced with larger fp32 caches. Paged fp32 is compared normally.

Rebuild the kernel from the current checkout with Mojo 1.2.0 using `python scripts/build.py` and run the supported-tier tests first; do not reuse an older shared object. Reproduce with `OMP_WAIT_POLICY=PASSIVE scripts/pin_governor.sh python benchmarks/bench_openvino.py --json /path/outside/repo/result.json`, then `python benchmarks/report.py --write-openvino <machine.json> ...` to regenerate both sections. Install the optional `openvino==2026.4.1` package to run this benchmark and `tests/test_benchmark_openvino.py`.

### ducviet00-NUC

AMD Ryzen 7 7840H w/ Radeon 780M Graphics, 8 threads, `20e9e7f`, torch 2.14.0+cpu, Mojo 1.2.0 (2016d0cd); OpenVINO 2026.4.1-22982-e213a147257-releases/2026/4.

Float64 validation: 96/96 timed arms also pass the README's stricter per-operation limits.


| shape | dtype | ours | openvino static | openvino dynamic | ours / ref |
|---|---|---:|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | bf16 | 121 (109-140) | 257 (208-276) | 373 (313-446) | 0.47 |
| N=1 L=128 S=128 H=16 E=64 causal | fp32 | 150 (140-158) | 382 (336-407) | 497 (463-528) | 0.39 |
| N=1 L=1024 S=1024 H=16 E=64 causal | bf16 | 2,613 (2,553-2,691) | 3,677 (3,624-3,764) | 4,172 (4,060-4,317) | 0.71 |
| N=1 L=1024 S=1024 H=16 E=64 causal | fp32 | 2,806 (2,774-4,766) | 6,616 (6,569-6,732) | 7,838 (7,697-9,321) | 0.42 |
| N=1 L=2048 S=2048 H=32 E=128 causal | bf16 | 27,076 (19,909-36,048) | 60,382 (56,389-81,596) | 61,072 (59,949-83,897) | 0.45 |
| N=1 L=2048 S=2048 H=32 E=128 causal | fp32 | 44,831 (41,058-51,398) | 140,512 (116,313-150,279) | 106,188 (104,455-145,503) | 0.42 |
| N=1 L=4096 S=4096 H=32 E=128 causal | bf16 | 81,647 (80,257-86,255) | 306,589 (296,674-322,990) | 282,098 (269,998-301,905) | 0.29 |
| N=1 L=4096 S=4096 H=32 E=128 causal | fp32 | 161,548 (160,113-167,173) | 462,379 (425,016-478,182) | 440,735 (435,672-475,078) | 0.37 |
| N=8 L=512 S=512 H=16 E=64 | bf16 | 9,048 (5,588-9,787) | 9,218 (8,892-15,498) | 10,975 (10,536-17,554) | 0.98 |
| N=8 L=512 S=512 H=16 E=64 | fp32 | 10,927 (10,269-19,056) | 18,302 (17,288-18,981) | 19,392 (18,783-21,732) | 0.60 |
| N=1 L=1500 S=1500 H=20 E=64 | bf16 | 8,480 (7,954-9,964) | 8,722 (8,416-14,402) | 10,663 (10,429-18,344) | 0.97 |
| N=1 L=1500 S=1500 H=20 E=64 | fp32 | 14,533 (13,118-18,123) | 16,912 (15,568-28,250) | 18,356 (16,465-29,262) | 0.86 |
| N=8 L=256 S=256 H=16 E=32 | bf16 | 1,307 (989-1,562) | 1,719 (1,544-2,686) | 1,884 (1,733-3,042) | 0.76 |
| N=8 L=256 S=256 H=16 E=32 | fp32 | 1,804 (1,676-2,849) | 2,640 (2,514-4,358) | 2,942 (2,809-3,186) | 0.68 |

| shape | dtype | ours p16 | ours p32 | openvino p32 | ours / ref |
|---|---|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | bf16 | 96.2 (83.2-116) | 84.1 (70.0-99.8) | 220 (184-278) | 0.38 |
| N=1 L=1 S=256 H=16 E=64 | fp32 | 165 (123-180) | 137 (130-167) | 261 (229-333) | 0.53 |
| N=1 L=1 S=4096 H=32 E=128 | bf16 | 2,327 (2,246-2,445) | 2,352 (2,298-2,449) | 2,160 (2,079-2,319) | 1.08 |
| N=1 L=1 S=4096 H=32 E=128 | fp32 | 3,555 (3,475-3,815) | 3,475 (3,265-3,607) | 3,881 (3,525-4,139) | 0.90 |
| N=8 L=1 S=1024 H=16 E=64 | bf16 | 1,139 (1,058-1,243) | 1,168 (1,076-1,229) | 1,295 (1,184-1,372) | 0.88 |
| N=8 L=1 S=1024 H=16 E=64 | fp32 | 2,386 (2,223-2,636) | 2,245 (2,118-2,358) | 2,310 (2,091-7,866) | 0.97 |
| N=8 L=1 S=1024 H=16 E=32 | bf16 | 762 (711-873) | 692 (624-952) | 800 (628-1,835) | 0.87 |
| N=8 L=1 S=1024 H=16 E=32 | fp32 | 1,320 (1,265-1,395) | 1,296 (1,138-1,432) | 1,409 (1,356-1,498) | 0.92 |
| N=8 L=1 S=4096 H=32 E=128 | bf16 | 13,203 (12,544-14,673) | 13,102 (12,681-18,114) | 11,809 (11,646-12,014) | 1.11 |
| N=8 L=1 S=4096 H=32 E=128 | fp32 | 20,773 (19,989-21,520) | 19,237 (19,168-19,523) | 19,445 (19,321-26,657) | 0.99 |
| N=8 L=1 S=17..2047 H=16 E=64 | bf16 | 977 (724-1,009) | 943 (730-979) | 1,017 (883-1,083) | 0.93 |
| N=8 L=1 S=17..2047 H=16 E=64 | fp32 | 1,636 (1,573-1,665) | 1,545 (1,493-1,556) | 1,622 (1,488-1,680) | 0.95 |
| N=8 L=1 S=1500 H=20 E=64 no append | bf16 | 1,866 (1,803-1,900) | 1,785 (1,687-1,883) | 1,964 (1,892-2,152) | 0.91 |
| N=8 L=1 S=1500 H=20 E=64 no append | fp32 | 2,950 (2,866-3,019) | 2,976 (2,894-3,034) | 2,951 (2,914-2,966) | 1.00 |
| N=32 L=1 S=512 H=16 E=64 | bf16 | 1,939 (1,884-2,022) | 1,912 (1,875-1,955) | 2,058 (2,019-2,177) | 0.93 |
| N=32 L=1 S=512 H=16 E=64 | fp32 | 3,274 (3,202-3,339) | 3,230 (3,165-3,336) | 3,239 (3,195-3,269) | 1.00 |
| N=32 L=1 S=1024 H=32 E=128 | bf16 | 12,181 (11,824-16,953) | 12,732 (12,478-14,588) | 11,048 (10,870-11,319) | 1.10 |
| N=32 L=1 S=1024 H=32 E=128 | fp32 | 21,491 (21,248-27,898) | 19,777 (19,591-21,462) | 19,210 (19,165-19,656) | 1.03 |

### c4a-standard-8

Neoverse-V2, 8 threads, `20e9e7f`, torch 2.14.0+cpu, Mojo 1.2.0 (2016d0cd); OpenVINO 2026.4.1-22982-e213a147257-releases/2026/4.

Float64 validation: 87/87 timed arms also pass the README's stricter per-operation limits.

- Unsupported: N=1 L=1 S=256 H=16 E=64, bf16, openvino p32: `bf16 paged attention/cache is not native on this CPU: ['f32']; f32 fallback does not preserve the shared bf16 cache writer`
- Unsupported: N=1 L=1 S=4096 H=32 E=128, bf16, openvino p32: `bf16 paged attention/cache is not native on this CPU: ['f32']; f32 fallback does not preserve the shared bf16 cache writer`
- Unsupported: N=8 L=1 S=1024 H=16 E=64, bf16, openvino p32: `bf16 paged attention/cache is not native on this CPU: ['f32']; f32 fallback does not preserve the shared bf16 cache writer`
- Unsupported: N=8 L=1 S=1024 H=16 E=32, bf16, openvino p32: `bf16 paged attention/cache is not native on this CPU: ['f32']; f32 fallback does not preserve the shared bf16 cache writer`
- Unsupported: N=8 L=1 S=4096 H=32 E=128, bf16, openvino p32: `bf16 paged attention/cache is not native on this CPU: ['f32']; f32 fallback does not preserve the shared bf16 cache writer`
- Unsupported: N=8 L=1 S=17..2047 H=16 E=64, bf16, openvino p32: `bf16 paged attention/cache is not native on this CPU: ['f32']; f32 fallback does not preserve the shared bf16 cache writer`
- Unsupported: N=8 L=1 S=1500 H=20 E=64 no append, bf16, openvino p32: `bf16 paged attention/cache is not native on this CPU: ['f32']; f32 fallback does not preserve the shared bf16 cache writer`
- Unsupported: N=32 L=1 S=512 H=16 E=64, bf16, openvino p32: `bf16 paged attention/cache is not native on this CPU: ['f32']; f32 fallback does not preserve the shared bf16 cache writer`
- Unsupported: N=32 L=1 S=1024 H=32 E=128, bf16, openvino p32: `bf16 paged attention/cache is not native on this CPU: ['f32']; f32 fallback does not preserve the shared bf16 cache writer`

| shape | dtype | ours | openvino static | openvino dynamic | ours / ref |
|---|---|---:|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | bf16 | 155 (115-195) | 646 (407-701) | 802 (517-908) | 0.24 |
| N=1 L=128 S=128 H=16 E=64 causal | fp32 | 167 (148-329) | 559 (381-597) | 725 (497-750) | 0.30 |
| N=1 L=1024 S=1024 H=16 E=64 causal | bf16 | 3,312 (1,707-4,025) | 12,398 (11,136-17,767) | 16,407 (12,875-17,933) | 0.27 |
| N=1 L=1024 S=1024 H=16 E=64 causal | fp32 | 8,017 (4,489-10,778) | 15,991 (10,949-18,450) | 12,881 (12,756-19,112) | 0.62 |
| N=1 L=2048 S=2048 H=32 E=128 causal | bf16 | 22,413 (20,886-34,669) | 146,330 (142,699-156,747) | 159,755 (153,922-184,531) | 0.15 |
| N=1 L=2048 S=2048 H=32 E=128 causal | fp32 | 64,618 (63,456-90,702) | 145,842 (140,940-154,189) | 153,714 (151,255-160,635) | 0.44 |
| N=1 L=4096 S=4096 H=32 E=128 causal | bf16 | 84,764 (84,558-92,864) | 573,046 (566,771-583,174) | 617,534 (611,790-623,185) | 0.15 |
| N=1 L=4096 S=4096 H=32 E=128 causal | fp32 | 251,351 (250,729-259,173) | 572,780 (564,133-575,514) | 613,630 (609,873-617,401) | 0.44 |
| N=8 L=512 S=512 H=16 E=64 | bf16 | 9,748 (6,318-10,616) | 26,863 (22,918-35,402) | 25,911 (24,594-33,619) | 0.38 |
| N=8 L=512 S=512 H=16 E=64 | fp32 | 17,593 (17,317-26,443) | 26,365 (22,205-31,794) | 28,418 (23,832-33,253) | 0.67 |
| N=1 L=1500 S=1500 H=20 E=64 | bf16 | 9,508 (8,427-12,367) | 29,012 (28,197-39,653) | 34,991 (29,815-38,622) | 0.33 |
| N=1 L=1500 S=1500 H=20 E=64 | fp32 | 23,424 (23,386-24,371) | 32,213 (28,030-36,369) | 29,735 (29,664-44,056) | 0.79 |
| N=8 L=256 S=256 H=16 E=32 | bf16 | 2,280 (1,184-2,310) | 4,550 (4,356-7,825) | 4,790 (4,764-8,134) | 0.50 |
| N=8 L=256 S=256 H=16 E=32 | fp32 | 2,754 (2,745-5,414) | 4,221 (4,183-4,380) | 4,631 (4,606-8,958) | 0.65 |

| shape | dtype | ours p16 | ours p32 | openvino p32 | ours / ref |
|---|---|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | bf16 | 46.9 (44.6-53.9) | 47.8 (35.6-55.1) | - | - |
| N=1 L=1 S=256 H=16 E=64 | fp32 | 79.7 (55.4-114) | 71.7 (57.0-96.8) | 232 (209-286) | 0.31 |
| N=1 L=1 S=4096 H=32 E=128 | bf16 | 684 (668-691) | 674 (668-684) | - | - |
| N=1 L=1 S=4096 H=32 E=128 | fp32 | 1,463 (1,419-2,267) | 1,263 (1,254-1,278) | 2,365 (1,856-3,937) | 0.53 |
| N=8 L=1 S=1024 H=16 E=64 | bf16 | 413 (405-423) | 431 (419-440) | - | - |
| N=8 L=1 S=1024 H=16 E=64 | fp32 | 871 (845-1,420) | 718 (703-731) | 1,157 (1,032-1,292) | 0.62 |
| N=8 L=1 S=1024 H=16 E=32 | bf16 | 208 (194-217) | 265 (242-277) | - | - |
| N=8 L=1 S=1024 H=16 E=32 | fp32 | 711 (467-753) | 448 (426-695) | 816 (711-977) | 0.55 |
| N=8 L=1 S=4096 H=32 E=128 | bf16 | 5,486 (5,338-5,576) | 5,740 (5,701-5,794) | - | - |
| N=8 L=1 S=4096 H=32 E=128 | fp32 | 10,945 (10,802-11,020) | 9,906 (9,735-9,946) | 16,696 (14,790-18,792) | 0.59 |
| N=8 L=1 S=17..2047 H=16 E=64 | bf16 | 286 (275-291) | 279 (271-285) | - | - |
| N=8 L=1 S=17..2047 H=16 E=64 | fp32 | 527 (504-858) | 475 (460-660) | 808 (741-935) | 0.59 |
| N=8 L=1 S=1500 H=20 E=64 no append | bf16 | 626 (608-640) | 600 (585-615) | - | - |
| N=8 L=1 S=1500 H=20 E=64 no append | fp32 | 1,377 (1,355-1,395) | 1,186 (1,149-1,195) | 1,885 (1,681-2,446) | 0.63 |
| N=32 L=1 S=512 H=16 E=64 | bf16 | 666 (647-692) | 624 (606-625) | - | - |
| N=32 L=1 S=512 H=16 E=64 | fp32 | 1,484 (1,467-2,449) | 1,291 (1,267-1,307) | 2,030 (1,858-2,270) | 0.64 |
| N=32 L=1 S=1024 H=32 E=128 | bf16 | 5,498 (5,314-5,587) | 5,835 (5,739-5,941) | - | - |
| N=32 L=1 S=1024 H=32 E=128 | fp32 | 11,013 (10,952-16,115) | 9,858 (9,670-12,519) | 15,640 (14,508-16,483) | 0.63 |

### c4-standard-8

INTEL(R) XEON(R) PLATINUM 8581C CPU @ 2.30GHz, 4 threads, `20e9e7f`, torch 2.14.0+cpu, Mojo 1.2.0 (2016d0cd); OpenVINO 2026.4.1-22982-e213a147257-releases/2026/4.

Float64 validation: 96/96 timed arms also pass the README's stricter per-operation limits.


| shape | dtype | ours | openvino static | openvino dynamic | ours / ref |
|---|---|---:|---:|---:|---:|
| N=1 L=128 S=128 H=16 E=64 causal | bf16 | 166 (154-184) | 407 (348-460) | 576 (544-611) | 0.41 |
| N=1 L=128 S=128 H=16 E=64 causal | fp32 | 281 (245-315) | 477 (420-742) | 644 (604-720) | 0.59 |
| N=1 L=1024 S=1024 H=16 E=64 causal | bf16 | 1,357 (1,271-1,405) | 4,627 (4,484-4,976) | 5,602 (5,465-6,959) | 0.29 |
| N=1 L=1024 S=1024 H=16 E=64 causal | fp32 | 2,874 (2,731-3,075) | 11,348 (11,046-11,544) | 11,253 (10,939-11,600) | 0.26 |
| N=1 L=2048 S=2048 H=32 E=128 causal | bf16 | 11,883 (11,704-12,894) | 55,531 (49,769-58,375) | 46,947 (46,492-47,865) | 0.25 |
| N=1 L=2048 S=2048 H=32 E=128 causal | fp32 | 29,558 (29,123-33,912) | 152,250 (151,831-153,964) | 156,930 (155,444-158,192) | 0.19 |
| N=1 L=4096 S=4096 H=32 E=128 causal | bf16 | 43,315 (42,982-47,860) | 201,961 (194,832-204,109) | 212,312 (208,589-220,183) | 0.21 |
| N=1 L=4096 S=4096 H=32 E=128 causal | fp32 | 117,278 (115,700-121,289) | 588,858 (587,408-598,712) | 626,272 (622,297-688,566) | 0.20 |
| N=8 L=512 S=512 H=16 E=64 | bf16 | 4,467 (4,381-4,516) | 9,322 (9,006-9,442) | 10,523 (10,022-10,780) | 0.48 |
| N=8 L=512 S=512 H=16 E=64 | fp32 | 9,188 (8,835-9,303) | 23,500 (22,939-24,025) | 25,805 (25,082-26,551) | 0.39 |
| N=1 L=1500 S=1500 H=20 E=64 | bf16 | 4,821 (4,748-6,283) | 9,778 (9,661-9,852) | 12,765 (12,638-12,888) | 0.49 |
| N=1 L=1500 S=1500 H=20 E=64 | fp32 | 10,763 (10,388-10,840) | 24,278 (23,991-24,386) | 26,263 (25,993-26,622) | 0.44 |
| N=8 L=256 S=256 H=16 E=32 | bf16 | 1,337 (1,260-1,407) | 2,701 (2,469-2,736) | 3,036 (2,834-3,123) | 0.50 |
| N=8 L=256 S=256 H=16 E=32 | fp32 | 2,299 (2,249-2,332) | 3,868 (3,767-3,932) | 4,360 (4,329-4,532) | 0.59 |

| shape | dtype | ours p16 | ours p32 | openvino p32 | ours / ref |
|---|---|---:|---:|---:|---:|
| N=1 L=1 S=256 H=16 E=64 | bf16 | 91.2 (83.6-96.9) | 86.0 (82.9-97.1) | 322 (293-345) | 0.27 |
| N=1 L=1 S=256 H=16 E=64 | fp32 | 125 (118-138) | 131 (109-139) | 359 (323-388) | 0.35 |
| N=1 L=1 S=4096 H=32 E=128 | bf16 | 1,762 (1,598-1,880) | 1,702 (1,593-1,802) | 2,312 (2,168-2,528) | 0.74 |
| N=1 L=1 S=4096 H=32 E=128 | fp32 | 2,631 (2,560-2,687) | 2,852 (2,681-2,956) | 2,994 (2,848-3,177) | 0.88 |
| N=8 L=1 S=1024 H=16 E=64 | bf16 | 939 (862-988) | 885 (830-932) | 1,276 (1,222-1,402) | 0.69 |
| N=8 L=1 S=1024 H=16 E=64 | fp32 | 1,314 (1,250-2,231) | 1,384 (1,335-1,453) | 1,720 (1,666-1,803) | 0.76 |
| N=8 L=1 S=1024 H=16 E=32 | bf16 | 545 (517-559) | 530 (465-683) | 878 (783-1,055) | 0.60 |
| N=8 L=1 S=1024 H=16 E=32 | fp32 | 736 (671-799) | 719 (676-763) | 1,088 (1,000-1,134) | 0.66 |
| N=8 L=1 S=4096 H=32 E=128 | bf16 | 13,358 (13,135-16,066) | 12,781 (12,474-12,890) | 14,338 (14,024-14,668) | 0.89 |
| N=8 L=1 S=4096 H=32 E=128 | fp32 | 19,578 (19,031-20,300) | 21,078 (20,700-21,568) | 20,843 (20,488-22,183) | 0.94 |
| N=8 L=1 S=17..2047 H=16 E=64 | bf16 | 609 (556-632) | 599 (579-618) | 956 (882-1,012) | 0.63 |
| N=8 L=1 S=17..2047 H=16 E=64 | fp32 | 858 (793-905) | 891 (855-931) | 1,181 (1,125-1,271) | 0.73 |
| N=8 L=1 S=1500 H=20 E=64 no append | bf16 | 1,509 (1,441-1,557) | 1,548 (1,488-1,586) | 2,034 (1,917-2,223) | 0.74 |
| N=8 L=1 S=1500 H=20 E=64 no append | fp32 | 2,376 (2,308-2,481) | 2,482 (2,442-2,557) | 2,946 (2,907-3,099) | 0.81 |
| N=32 L=1 S=512 H=16 E=64 | bf16 | 1,747 (1,696-1,795) | 1,764 (1,743-1,894) | 2,427 (2,384-2,576) | 0.72 |
| N=32 L=1 S=512 H=16 E=64 | fp32 | 2,570 (2,487-2,783) | 2,776 (2,675-2,806) | 3,246 (3,117-3,364) | 0.79 |
| N=32 L=1 S=1024 H=32 E=128 | bf16 | 12,711 (12,315-12,792) | 12,253 (12,000-12,526) | 14,042 (13,425-14,235) | 0.87 |
| N=32 L=1 S=1024 H=32 E=128 | fp32 | 19,519 (19,168-19,680) | 21,277 (20,887-21,547) | 21,158 (20,707-21,532) | 0.92 |

<!-- OPENVINO RESULTS END -->
