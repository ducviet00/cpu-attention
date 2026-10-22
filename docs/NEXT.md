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

1. **Per-work-item cost, ~1000 cycles**, measured by `tests/bench_scale.py`
   (heads scaling at fixed seq): the kernel's own ceiling at a 128-key range is
   44% of the `bfmmla` ceiling however many heads are in flight, against 74% at
   2048 keys. That is the whole S<=512 story and it is worth 13-26% on the four
   worst rows.
2. **The K^T pack layout** (`[tile][dim group][key pair][8]`) is read 16-strided;
   L1 read misses are 0.348 per row-key against vLLM's 0.086. Swapping the two
   middle axes makes one sub-block's K read contiguous. That is the D=128 rows
   (1.06-1.09).
3. **The K/V pack throughput at short S** (6 us of a 51 us call at 128:8:64,
   85 GB/s) — measure the memcpy floor for the same size and thread count before
   spending anything on it.
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
