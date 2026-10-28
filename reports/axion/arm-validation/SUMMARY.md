# Axion ARM handoff — 2026-10-28

Partial improvement only. The highest-performance objective is **not met**.

## Source and isolation

- Kernel commit: `8b6521a` on `afbea06509ab3440c1a329c77c70561fa4c49f2d`.
  Only `cpu_attention/_mojo/src/kvcache.mojo` changes: ARM `_pv` dispatch and
  `_pv_arm`. No changes to `_body`, the store scheduler, dense interfaces, or
  `flash_neon.mojo`.
- Worktree: `/home/ducviet00/cpu-attention/.claude/worktrees/agent-abe75746e503eeaa3`.
  Initially clean. Primary shared files were not edited.
- Existing VM `fe-arm-20261026-173549`, `34.87.120.65`, user `ducviet00`:
  Neoverse-V2, 8 cores, 8 torch threads. **Not created or deleted here.**
- Remote existing `~/agent-arm` had uncommitted changes and was left untouched.
  Work ran in `~/arm-validation-afbea06`; all heavy work held `~/bench.lock`.
  Process snapshots are `processes*.txt`; no competing benchmark was observed.
- Compiler Mojo 1.2.0 (`2016d0cd`), torch 2.14.1+cpu, Python 3.14.4,
  GCC 15.2.0. Full environment, source hashes, library/Python fingerprints,
  samples, minima and medians are in the JSON files.
- Built both explicit ARM Mojo tiers with `scripts/build.py`, and C++ with
  `scripts/build_cpp.py`. The final `--tests` build reproduced the same hashes.
- Final `kvcache.mojo` SHA256:
  `158dac6d778d86394330c097f8efa231f98713f631916bfba6d06e1ae648867e`.
- Final bf16-tier library SHA256:
  `8131e6831df337c110c6242a21826125a6dc2d04f24c27da19e5c41633b45c4b`.
  C++ library:
  `1af7d0e6ea640fd800d2be1f8a8a40da1db481b3a1ab7ce9f3cdd2b739fac478`.
  Other tiers and exact commands: `final-sha256.txt`, `build-*.log`.

## Mechanism and measurement

The unified ARM bf16 P@V widened paired bf16 V to fp32, deinterleaved it and
used scalar-probability FMA. Sampling attributed about 70% of the MHA bf16
body to this phase (`perf-bf16-annotate.txt`, addresses `0x3e080..0x3e400`,
approximate sampled share). `perf record` worked; hardware `perf stat` counters
did not: this VM does not support the requested cycles event.

The candidate reads the existing paired V layout directly with `bfdot`.
Probabilities use bf16 **hi/lo** components, preserving the ARM accuracy gate;
an odd tail uses only the valid key, avoiding `0 * NaN` from poisoned padding.
No cache layout changes. Lower ARM tier and non-ARM policies stay unchanged.

Every reported timing ratio is from one process, 41 rotated/interleaved rounds,
minimum per arm, fastest valid C++ per row, and independently loaded saved old
package plus its duplicate control. Never compare times between JSON files.

Primary's updated harness is saved under `harness/` and was used for the hi/lo
and rq128 runs. The first baseline predates those harness fixes. The direct-dot
development run uses `compare.py` with separate old/old-control and new/new-control
modules; do not interpret its control column as an old-build control spread.

## Full paged/store manifest

All files below are relative to this directory. Every manifest includes 92 rows.

| Run | Win | Unresolved | Loss | New invalid | Notes |
|---|---:|---:|---:|---:|---|
| `paged-full-afbea06.json` | 11 | 24 | 55 | 0 | 2 rows lacked a valid baseline under old screening |
| `paged-full-hilo-scalar-peer.json` | 12 | 27 | 53 | 0 | updated harness, safe scalar peer, vLLM/oracle included |
| `paged-full-hilo-rotated.json` | 21 | 20 | 51 | 0 | shuffled package load/allocation order, no vLLM/oracle arm |

The hi/lo candidate improved **all 44 bf16 decode/API rows beyond their own
old-build control band** in the scalar-peer run: new/previous `0.590..0.924`.
The shuffled run independently reports `0.546..0.892`; these are separate
same-process observations, not a cross-process speedup comparison.

Remaining store-only rows in the scalar-peer run:

| Row | New / fastest valid C++ | Verdict |
|---|---:|---|
| bf16 b32, one token | 1.056 | loss; valid scalar baseline, NEON invalid |
| fp32 b32, one token | 1.159 | loss |
| bf16 b1, 1024-token prompt | 0.762 | win |
| fp32 b1, 1024-token prompt | 1.026 | unresolved |

Store-only code is unchanged; the primary owns scheduling experiments there.

There are fp32 timing regressions against the saved build despite identical
fp32 mnemonic sequences/instruction counts in all nine specialized paged bodies
(`fp32-assembly.txt`). In the shuffled run, four rows exceed their control band:
engine self b8/1024 (`1.030`), engine cross b8/257 (`1.058`), engine cross
b8/1024 GQA 16/4 (`1.063`), engine_out self b1/256 (`1.032`). These module/cache
placement effects remain unresolved; do not claim a no-regression result.

Every row, including every loss/unresolved and old-build ratio, is in `ROWS.md`.

## Dense rq128 experiment (shared-interface recommendation only)

`flash-full-rq128.json`: 78 rows, 54 win / 22 unresolved / 2 loss, all outputs
valid. `rq128.py` modifies only a private copied package's geometry rule;
neither `_kernel.py` nor the dense policy in this worktree was edited.

At fp32 D=128, new/previous was `0.936` at 2048:16:128, `0.906` at
4096:16:128 and `0.966` at 2049:16:128, with old-control spread <=0.002.
Short D=128 rows regressed up to `1.095` (bf16 129:16:128); bf16 large rows
were mostly neutral. **Reject global rq128.** A candidate for the primary is
fp32 D=128 long square causal rows only, followed by full integration gates.
Other geometries are not justified by these data.

The two C++ losses in this run are bf16 128:8:64 (`1.079`) and bf16 129x127:8:64
(`1.054`). All flash losses/unresolved are listed in `ROWS.md`.

## Correctness

| Gate | Final Mojo bf16 tier | Mojo lower ARM tier | C++ |
|---|---|---|---|
| `test_dense.py` | 82/82 | 82/82 | 89/89 |
| `test_flash_api.py` | pass | pass | pass |
| `fuzz_dense.py --cases 80 --max-seq 900` | 80/80 + 24/24 stray reads | same | same |
| `test_paged_sweep.py` | 504/504 | 504/504 | 504/504 |
| extra `test_regressions.py` | 412/412 | 412/412 | NEON 386/412; scalar 412/412 |
| fast-encdec `test_cpu_attention.py` | 45 checks pass | not rerun | 45 checks pass |
| standalone Mojo test | pass | not separate | n/a |

Final Mojo paged sweep: worst fp32 `9.54e-7`, bf16 `1.89` peak-relative ulp.
Logs are under `hilo-gates/`, baseline C++ logs under `gates/`.

Fast-encdec source was archived from `dd3b28e8bbc0076a4118814040d670f17123ee49`
to `~/arm-validation-engine-afbea06`; dependencies were installed into its
private `deps/`, not an existing environment. Both-backend baseline logs and
final Mojo engine log are retained. Its tiny-model token sequences are
degenerate, so they are not a replacement for scalar-reference tests.

## Falsification and protocol findings

1. **Single-bf16 probabilities + direct bfdot: rejected.** Improved timing,
   but both heterogeneous bf16 rows failed the 1e-3 excess gate even though
   the standard suites passed. Raw: `paged-full-bfdot.json`; exact candidate
   source: `direct-bfdot.mojo`. Hi/lo is the new mechanism that fixes this.
2. **Global D=128 rq128: rejected.** Up to 9.5% short-row regression; see above.
   This is the unified Mojo geometry experiment, not a replay of C++ rq tuning.
3. **Invalid peer poisons validation.** The default bf16 peer is C++ NEON,
   which returns NaN on the poisoned one-token store row. Then even finite
   new/scalar results get `err=NaN` and every arm appears invalid.
   `paged-full-bfdot-hilo.json` records this. `safe_reference.py` / the shuffled
   wrapper select C++ scalar as the peer; output checks and fastest-valid-C++
   selection remain unchanged. New output then passes. Primary should fix
   peer selection rather than treating that initial NaN as a new-kernel failure.
4. C++ NEON is also numerically invalid on heterogeneous bf16 contexts and
   fails poisoned-tail regressions. Scalar passes; invalid NEON rows are never
   used as the fastest-valid baseline in the final runs.

No objective/acceptance claim is made. Artifacts and the retained kernel are a
handoff for primary integration, not a unified-tree acceptance result.
