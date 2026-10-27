AMX subagent owns /tmp/opencode/cpp-amx-20261027 (branch cpp-amx-20261027, a1d9fdf).
Remote ~/cpu-attention is my isolated a1d9fdf checkout; do not sync over it.
Box actual identity: Xeon Platinum 8581C, 4 physical cores / 8 threads.
Toolchain/uv provisioned. Mojo/C++ builds running. vLLM csrc copied from exact
db3b0f0fae4bc98492b3863685e541e62bc59576, VEC/AMX refs next.
No local benchmarking, no shared-tree edits. Paged forwards to vec in my owned
header; b1-cross fixes likely belong to primary/VEC owner.
All remote timing/build/parity use ~/bench.lock after provisioning.
Mojo and C++ built successfully, C++ --test passed 82/82 dense + flash API.
Found an existing nested-process affinity pitfall: ablate2.py imports torch in
the parent, OMP_PROC_BIND pins parent to {0,4}, then subprocesses inherit only
that single core. First profiles are explicitly INVALID. Corrected outer driver
restores main affinity to range(8) after import before spawning. Same-process
old/new AB uses allocated outputs for all variants and avoids this trap.
Hardware PMU is unavailable even under sudo (cycles event unsupported). Software
cpu-clock sampling and corrected phase ablations are being collected.
