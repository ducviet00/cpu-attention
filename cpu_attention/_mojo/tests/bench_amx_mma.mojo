"""AMX `tdpbf16ps` issue rate and the cost of the tile loads that feed it.

The attention loop's matmul share only means something against this ceiling.
Clock is taken as 2.36 GHz (a 1024^3 fp32 matmul measures 151 GFLOP/s/core,
which is exactly the 64-flops/cycle FMA peak), so every cycle figure here is
+/-1% rather than measured in-situ.

    mojo build -I cpu_attention/_mojo/src \
        cpu_attention/_mojo/tests/bench_amx_mma.mojo -o /tmp/b && /tmp/b
"""

from amx import (
    amx_init,
    tile_cfg_load,
    tile_config_bf16,
    tile_dpbf16ps,
    tile_load,
    tile_release,
    tile_store,
    tile_zero,
)
from std.memory import stack_allocation
from std.time import perf_counter_ns

comptime D = 128
comptime KEYS = 32
comptime ROWS = 32
comptime REP = 200
comptime DCH = D // 32
comptime NCH = KEYS // 16
comptime GHZ = 2.36


def main():
    if not amx_init():
        print("AMX unusable here")
        return
    var cfg = stack_allocation[64, UInt8, alignment=64]()
    tile_config_bf16(16, 64, cfg)
    var a = stack_allocation[ROWS * D, BFloat16, alignment=64]()
    var b = stack_allocation[KEYS * D, BFloat16, alignment=64]()
    var c = stack_allocation[ROWS * KEYS, Float32, alignment=64]()
    for i in range(ROWS * D):
        a[i] = BFloat16(Float32(i % 61) * 0.0009765625 - 0.03)
    for i in range(KEYS * D):
        b[i] = BFloat16(Float32(i % 53) * -0.0009765625 + 0.02)
    for i in range(ROWS * KEYS):
        c[i] = Float32(0)
    tile_cfg_load(Int(cfg))
    tile_zero[0]()
    tile_zero[1]()
    tile_load[2](Int(b), 64)
    tile_load[3](Int(b) + 1024, 64)

    # --- mma with 8 independent accumulators: throughput, not latency ---
    var t0 = perf_counter_ns()
    for _ in range(REP):
        comptime for _ in range(4):
            tile_dpbf16ps[0, 2, 3]()
            tile_dpbf16ps[1, 2, 3]()
            tile_dpbf16ps[4, 2, 3]()
            tile_dpbf16ps[5, 2, 3]()
            tile_dpbf16ps[6, 2, 3]()
            tile_dpbf16ps[7, 2, 3]()
            tile_dpbf16ps[4, 2, 3]()
            tile_dpbf16ps[5, 2, 3]()
    var dt = Float64(perf_counter_ns() - t0)
    var n = Float64(REP * 32)
    print(
        "mma8 ",
        dt / n,
        "ns/mma  ",
        dt / n * GHZ,
        "cyc/mma  ",
        n * 8192.0 / dt / GHZ,
        "MAC/cyc  ",
        n * 16384.0 / dt,
        "GFLOP/s"
    )

    # --- mma with 4 accumulators (the kernel's C tiles): latency exposed ---
    t0 = perf_counter_ns()
    for _ in range(REP):
        comptime for _ in range(8):
            tile_dpbf16ps[4, 2, 3]()
            tile_dpbf16ps[5, 2, 3]()
            tile_dpbf16ps[6, 2, 3]()
            tile_dpbf16ps[7, 2, 3]()
    dt = Float64(perf_counter_ns() - t0)
    n = Float64(REP * 32)
    print(
        "mma4 ",
        dt / n,
        "ns/mma  ",
        dt / n * GHZ,
        "cyc/mma  ",
        n * 8192.0 / dt / GHZ,
        "MAC/cyc"
    )

    # --- loads only: 4 tile loads per iteration, no mma ---
    t0 = perf_counter_ns()
    for _ in range(REP):
        for dc in range(DCH):
            comptime for kb in range(NCH):
                var kof = (kb * DCH + dc) * 512 * 2
                tile_load[0](Int(a) + dc * 64, D * 2)
                tile_load[1](Int(a) + ROWS // 2 * D * 2 + dc * 64, D * 2)
                tile_load[2](Int(b) + kof, 64)
                tile_load[3](Int(b) + kof + DCH * 512 * 2, 64)
    tile_store[0](Int(c), 128)
    dt = Float64(perf_counter_ns() - t0)
    n = Float64(REP * DCH * NCH * 4)
    print("loads", dt / n, "ns/tile  ", dt / n * GHZ, "cyc/tile  ", n * 1024.0 / dt / GHZ, "B/cyc")

    # --- the shipped QK pattern: 4 loads + 4 mma + 4 stores per dim chunk ---
    t0 = perf_counter_ns()
    for _ in range(REP):
        for dc in range(DCH):
            comptime for kb in range(NCH):
                var kof = (kb * DCH + dc) * 512 * 2
                tile_zero[4]()
                tile_zero[5]()
                tile_zero[6]()
                tile_zero[7]()
                tile_load[0](Int(a) + dc * 64, D * 2)
                tile_load[1](Int(a) + ROWS // 2 * D * 2 + dc * 64, D * 2)
                tile_load[2](Int(b) + kof, 64)
                tile_load[3](Int(b) + kof + DCH * 512 * 2, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
                tile_store[4](Int(c), 128)
                tile_store[5](Int(c) + 64, 128)
                tile_store[6](Int(c) + 16 * 128, 128)
                tile_store[7](Int(c) + 16 * 128 + 64, 128)
    dt = Float64(perf_counter_ns() - t0)
    var logits = Float64(REP * DCH * NCH) * Float64(ROWS) * 16.0
    print("qk   ", dt / logits, "ns/logit  ", dt / logits * GHZ, "cyc/logit")

    tile_release()
