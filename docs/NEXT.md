# When AWS access is back

1. `aws login` (the CLI session expired mid-session; the x86 spot box was
   reclaimed and could not be checked or replaced).

2. Re-verify `main` on the Xeon, which is the one table that is stale:

       scripts/verify.sh x86_64 ubuntu@<m8i ip> "$PWD" final-x86   # ROUNDS=21 is the default in the script? (no: pass ROUNDS=21)

   The last full x86 verification was at `164e8e6`, before the Python dispatch
   change (one `.stride()` per tensor, 5% at S=128) and the ARM merge. Both are
   shared-code changes and the x86 kernel is untouched, so the expectation is
   22/22 in both columns and slightly better at S=128 — but it must be measured,
   not assumed.

3. Optional, if ARM is to be pushed further: the only untried routes are a
   from-scratch vLLM-shaped kernel (the previous campaign's two prototypes lost
   20-26%) or an instruction-by-instruction study of vLLM's `tile_gemm` to find
   the 0.38 IPC that the static diff did not show. Everything local has been
   measured: tile sizes, adaptive tiles, fp16 softmax, co-issue, D=128 panels,
   pack, Python dispatch, alignment, unrolling, pointer hoists.
