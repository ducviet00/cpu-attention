"""Standalone microbenchmark of the AMX pack, three ways.

Why: in the full call the K pack moves 512 KB (read+write) in 14-16 us at
S=128, i.e. 32 GB/s against a measured 139 GB/s for an L2-resident copy of the
same size, and its instruction budget is ~96 per 1 KB tile (~25 K instructions
per thread, ~2 us at 1.2 IPC). Neither traffic nor instruction count explains
it, so isolate the two halves:

  full      the real per-tile work: 16 loads of 64 B, the 16x16 32-bit
            transpose (4 stages x 16 shuffles), 16 stores of 64 B
  noload    the loads replaced by a fixed register value (keeps the whole
            transpose network and all the stores)
  notp      the transpose replaced by a copy (keeps every load and store)

If `noload` is ~= `full`, the network is the cost; if `notp` is ~= `full`, the
loads are.

    PATH=$PWD/.venv/bin:$PATH .venv/bin/mojo build -I cpu_attention/_mojo/src \
        --target-cpu graniterapids cpu_attention/_mojo/tests/pack_bench.mojo \
        -o /tmp/pack_bench && /tmp/pack_bench
"""

from std.memory import MutPointer, stack_allocation
from std.origin import MutUntrackedOrigin
from std.time import perf_counter_ns


comptime TILE_KEYS = 16
comptime TILE_BYTES = 1024


@inline(.always)
def _ptr[T: AnyType](address: Int) -> MutPointer[T, MutUntrackedOrigin]:
    return MutPointer[T, MutUntrackedOrigin](unsafe_from_address=address)


@inline(.always)
def _tp16_i32(m: Array[SIMD[DType.int32, 16], 16]) -> Array[
    SIMD[DType.int32, 16], 16
]:
    """The kernel's transpose: in place, two temporaries, four stages."""
    var a = m.copy()
    comptime for k in range(4):
        comptime step = 1 << k
        comptime for i in range(16):
            comptime if (i >> k) & 1 == 0:
                comptime b = i | step
                comptime if k == 0:
                    var t0 = a[i].shuffle[
                        0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30
                    ](a[b])
                    var t1 = a[i].shuffle[
                        1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
                comptime if k == 1:
                    var t0 = a[i].shuffle[
                        0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29
                    ](a[b])
                    var t1 = a[i].shuffle[
                        2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
                comptime if k == 2:
                    var t0 = a[i].shuffle[
                        0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27
                    ](a[b])
                    var t1 = a[i].shuffle[
                        4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
                comptime if k == 3:
                    var t0 = a[i].shuffle[
                        0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23
                    ](a[b])
                    var t1 = a[i].shuffle[
                        8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31
                    ](a[b])
                    a[i] = t0
                    a[b] = t1
    return a.copy()


def _pack_tile(
    variant: Int, src: Int, dst: Int, k0: Int, stride_t: Int, d0: Int
) -> None:
    """One 16-key x 32-dim tile, with one half of the work removable.

    variant 0 = full, 1 = no loads, 2 = no transpose.
    """
    var rows = Array[SIMD[DType.int32, 16], 16](uninitialized=True)
    var sp = _ptr[Int32](src)
    comptime for i in range(16):
        if variant == 1:
            rows[i] = SIMD[DType.int32, 16](i)
        else:
            rows[i] = _ptr[Int32](
                src + ((k0 + i) * stride_t + d0) * 2
            ).unsafe_load[width=16]()
    var cols = rows.copy() if variant == 2 else _tp16_i32(rows)
    var dp = _ptr[Int32](dst)
    comptime for i in range(16):
        dp.unsafe_store[width=16](i * 16, cols[i])
    _ = sp


def main():
    comptime D = 64
    comptime SEQ_K = 4096
    comptime KB16 = SEQ_K // TILE_KEYS  # 256 key blocks
    comptime DC32 = D // 32  # 2 dim chunks per key block
    comptime TILES = KB16 * DC32
    var src_buf = stack_allocation[SEQ_K * D, BFloat16, alignment=64]()
    var dst_buf = stack_allocation[TILES * 512, BFloat16, alignment=64]()
    for i in range(SEQ_K * D):
        src_buf[i] = Scalar[DType.bfloat16](Float32(i % 251) * 0.015625)
    for i in range(TILES * 512):
        dst_buf[i] = Scalar[DType.bfloat16](0)
    var src = Int(src_buf)
    var dst = Int(dst_buf)
    var order = stack_allocation[3, Int]()
    # interleave the variants so drift cannot favour one (these boxes throttle)
    for rep in range(9):
        for v in range(3):
            order[v] = (v + rep) % 3
        for vi in range(3):
            var variant = order[vi]
            var t0 = perf_counter_ns()
            var reps = 8
            for r in range(reps):
                for kb in range(KB16):
                    for dc in range(DC32):
                        _pack_tile(
                            variant,
                            src,
                            dst + ((kb * DC32 + dc) * 512) * 2,
                            kb * TILE_KEYS,
                            D,
                            dc * 32,
                        )
            var t1 = perf_counter_ns()
            var ns = Float64(t1 - t0) / Float64(reps * KB16 * DC32)
            var gbs = (
                Float64(reps * TILES * TILE_BYTES * 2)
                / Float64(t1 - t0)
            )
            var name = "full"
            if variant == 1:
                name = "noload"
            elif variant == 2:
                name = "notp  "
            print(rep, name, " ns/tile", Int(ns), " GB/s", Int(gbs))
