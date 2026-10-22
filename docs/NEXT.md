# Next steps, in order

## 0. AWS access is the blocker (needs a human)

The CLI's login refresh token is rejected at the signin endpoint
(`TOKEN_EXPIRED` from `POST https://ap-southeast-2.signin.aws.amazon.com/v1/token`,
seen with `aws sts get-caller-identity --debug`), so no box can be launched or
stopped from here. The only fix is an interactive `aws login` on a machine with
a browser; there is no credential path in this checkout that survives it.

The m9g.xlarge at `13.210.162.215` is still running and is what every number in
`docs/STATUS.md` was measured on. It costs ~$0.10/hr and holds nothing that is
not in git.

## 1. Re-verify x86 (never done since the shared-code changes)

The last x86 verification was at `164e8e6`, **before** the second ARM campaign
and before the host-path rewrite. Both are shared code:

* `cpu_attention/_kernel.py` — the whole dense call path was restructured
  (`_dense_plan`, per-geometry cache). The AMX path is entered from `_dense`
  through `_amx_dense`, whose arguments come from a different branch, and the
  Python `_dense` path is what runs when AMX is unavailable.
* `cpu_attention/flash_attn_interface.py` — argument checks restructured.
* `cpu_attention/_mojo/src/dense.mojo` — the ARM kernels, plus shared
  constants; the x86 AVX-512 path in the same file is untouched but must still
  compile and pass.

```
scripts/verify.sh x86_64 ubuntu@<m8i ip> "$PWD" x86-after-arm2
```

That runs the VEC matrix and also builds and times vLLM's AMX variant (bf16
only). Expectation from `164e8e6` is 21-22/22 against VEC, with `128:8:64 fp32`
the known-weak row; the point is to confirm the shared changes did not move it.

## 2. Re-verify ARM on the *target* slice

Every ARM number so far is `m9g.xlarge` (4 cores). The scorecard's target is
`m9g.2xlarge` (8 cores), where bandwidth per core is roughly halved — and we are
the side that pays a K+V repack per call and re-reads V at long S, so if the
ratio moves it likely moves against us. `docs/CEILINGS.md` has the per-row
efficiency table that the 8-core numbers should be compared against.

```
scripts/verify.sh arm64 ubuntu@<m9g.2xlarge ip> "$PWD" arm-8core
ROUNDS=21
```

## 3. The open technical items (in the order they are worth attacking)

1. **Per-work-item cost, ~780 cycles** (`b = 0.24 us` from `tests/bench_scale.py`,
   37% of a 128-key item). The item is ~580 instructions = ~165 cycles, so ~600
   of the 780 is **stall, not arithmetic** -- do not attack the prologue's
   instruction count. The two candidates left are the Q-load -> panel build ->
   first Q@K^T chain (test: prefetch `task + threads`' Q rows, known one item
   ahead) and load imbalance across the 16-block causal spread. `qt` is already
   best at 8 (`tests/bench_qt.py`), and the S=128 row cannot be won even with a
   perfect per-item cost and a free pack (~1.06 floor).
2. ~~**The K^T pack layout**: swap the two middle axes so one sub-block's K read
   is contiguous~~ -- **tried and falsified** (NOTES-ARM, fourth campaign): with
   `[tile][key pair][dim group][8]` and parity green, `l1d_cache_lmiss_rd` is
   unchanged (0.2749 -> 0.2789 at 2048:8:64, 0.6252 -> 0.6261 at 1024:16:128)
   and the whole call is 1.00-1.02x. The misses are not the K stride.

   What the counters say at **1024:16:128 bf16**, ours against vLLM: cycles/
   row-key 9.2 vs **8.58**, instructions 32.4 vs **35.7**, IPC 3.5 vs **4.16**,
   `l1d_cache_lmiss_rd` 0.87 vs **0.167**, `stall_backend` 2.2-2.5 vs **1.60**,
   `stall_backend_mem` ~0.4 vs **0.04**. vLLM executes *ten percent more
   instructions* at D=128 and wins on IPC alone, so no instruction-count change
   reaches those rows. The memory fourth of the gap is **V**: the P@V loop walks
   `dcp = 4` dim pairs per key group with the d-chunk loop outermost, so at
   head_dim 128 it makes 16 passes over the tile touching 64 B of every 1 KB key
   group. A `[d-chunk][key group][dim pair][8]` V layout makes each pass
   contiguous; the kernel must derive the d-chunk stride from `seq_k_pad`
   (`((seq_k+63)//64)*16*32` for the shipped scratch shape) because the host owns
   the allocation.
3. ~~The K/V pack throughput at short S~~ **closed**: called on its own the
   pack is 2.90 us at 128:8:64 (181 GB/s) against a 206 GB/s plain-copy floor,
   and 88% of that floor at every size. The "6 us" figure came from
   `bench_phases.py`'s `NO_FUSE` extra fork-join. `tests/pack_only.py`.
4. `qt` per shape: the table in `_qt_for` was measured before the per-work-item
   cost was understood, and a *larger* `qt` at short S is now a plausible win
   (fewer work items, so fewer per-item costs). `tests/bench_qt.py` and
   `scripts/prof/arm/qt_sweep.py` interleave every `qt` in one process;
   `_kernel.set_qt()` is the hook.

## 4. Closed — do not reopen without new hardware

* **int8 Q@K^T** (`smmla`): 19% faster, misses the 3e-2 bf16 gate at every
  scaling that fits it; `tests/int8_qk_error.py` has the numbers including four
  adversarial input families, and the shipped bf16 kernel is itself at 2.6-7.4e-2
  on those. `comptime INT8_QK` stays False.
* **fp16 MMA** (`fmmla`): the `f32mm` extension is not implemented on any
  shipping core; LLVM has no instruction-selection pattern for the intrinsic.
* **Wider SIMD**: SVE on Neoverse-V3 is 128-bit
  (`/proc/sys/abi/sve_default_vector_length` = 16).
* **A bf16 schedule rewrite for the long-S D=128 rows**: vLLM is at 95% of the
  measured `bfmmla` ceiling there, so parity is the best a bf16 kernel can do.
