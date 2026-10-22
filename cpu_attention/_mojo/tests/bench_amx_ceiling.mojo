"""AMX + softmax ceiling probe: what the roof of this core actually is.

Before tuning a kernel it is worth knowing the machine's roofs, because every
design decision after that is a choice about which roof binds. Measured with the
*same* 2-2-4 tile pattern and the same dependency structure the attention kernel
has (four accumulators, each a chain of `dc` dots):

    mode 0  dots only, no memory operands                     (issue roof)
    mode 1  16 dots + 16 tile loads, all from fixed L1 lines   (QK loop, hot)
    mode 2  same, B streaming over a 32 KB arena               (QK loop, walking)
    mode 3  mode 2 + 4 tile stores per group                   (QK into logits)
    mode 4  the 16 tile loads, no dots                         (load roof)
    mode 5  A held in tiles, 1 B load + 2 dots + 2 stores      (QK, N=16 variant)
    mode 7  one dependent chain of dots                        (dot latency)
    mode 9  softmax pass over 32 x kt logits, 1 row in flight
    mode 10 as 9, 2 rows in flight
    mode 11 as 9, 4 rows in flight
    mode 12 mode 11 + the running max (a stale-max softmax)
    mode 13 mode 11 with a degree-2 polynomial
    mode 14 mode 11 at kt = 128 (logits+probs comfortably L1-resident)
    mode 15 mode 11 at kt = 1024 (logits+probs spill to L2)
    mode 16 mode 12 at kt = 128

The softmax modes matter because that pass is the dominant *non*-matmul work in
a dense attention kernel: their unit is one logit.

    .venv/bin/python tests/bench_amx_ceiling.py          # builds + runs this
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
from std.ffi import external_call
from std.math import fma
from std.memory import MutPointer, bitcast, stack_allocation
from std.origin import MutUntrackedOrigin
from std.time import perf_counter_ns

comptime ARENA = 1 << 15
"""Bytes of B operand material per thread (32 KB: L1/L2 resident)."""

comptime OUTER = 64
"""Dot groups per loop iteration: keeps the loop overhead out of the measurement."""

comptime ROWS = 32
"""Rows in the softmax microbenchmark: one work item's worth."""

comptime KBUF = 1024
"""Logits columns per thread buffer (the largest kt probed)."""

comptime V16 = SIMD[DType.float32, 16]
comptime MAGIC = 12582912.0


@inline(.always)
def _exp2v[deg: Int](x: SIMD[DType.float32, 16]) -> SIMD[DType.float32, 16]:
    """2**x with a comptime polynomial degree; exactly 0 once x <= -127."""
    var xc = min(max(x, V16(-127.0)), V16(127.0))
    var n = (xc + V16(MAGIC)) - V16(MAGIC)
    var f = xc - n
    var p = SIMD[DType.float32, 16](0.0013333558146428443)
    comptime if deg >= 5:
        p = fma(p, f, SIMD[DType.float32, 16](0.009618129107628477))
    comptime if deg >= 4:
        p = fma(p, f, SIMD[DType.float32, 16](0.05550410866482158))
    comptime if deg >= 3:
        p = fma(p, f, SIMD[DType.float32, 16](0.2402265069591007))
    comptime if deg == 2:
        p = SIMD[DType.float32, 16](0.2402265069591007)
    p = fma(p, f, SIMD[DType.float32, 16](0.6931471805599453))
    p = fma(p, f, SIMD[DType.float32, 16](1.0))
    var ni = n.cast[DType.int32]()
    var bits = (ni + SIMD[DType.int32, 16](127)) << SIMD[DType.int32, 16](23)
    return p * bitcast[DType.float32, 16](bits)


@inline(.always)
def _sm_rows[
    rilp: Int, deg: Int, withmax: Bool
](
    lp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[BFloat16, MutUntrackedOrigin],
    kt: Int,
    c1v: SIMD[DType.float32, 16],
    off: SIMD[DType.float32, 16],
) -> Float32:
    """The softmax pass of `rilp` rows: load, scale, exp2, sum, bf16, store.

    This is the whole of the per-logit vector work in the attention kernel's
    dominant phase, so its ns/logit is the number everything else is judged
    against.
    """
    var sums = Array[SIMD[DType.float32, 16], rilp](uninitialized=True)
    var mx = Array[SIMD[DType.float32, 16], rilp](uninitialized=True)
    comptime for u in range(rilp):
        sums[u] = SIMD[DType.float32, 16](0.0)
        mx[u] = SIMD[DType.float32, 16](-3.4028234663852886e38)
    var j = 0
    while j < kt:
        comptime for u in range(rilp):
            var x = lp.unsafe_load[width=16](u * kt + j)
            comptime if withmax:
                mx[u] = max(mx[u], x)
            var e = _exp2v[deg](fma(x, c1v, off))
            sums[u] += e
            pp.unsafe_store[width=16](u * kt + j, e.cast[DType.bfloat16]())
        j += 16
    var s = SIMD[DType.float32, 16](0.0)
    var m = SIMD[DType.float32, 16](0.0)
    comptime for u in range(rilp):
        s += sums[u]
        comptime if withmax:
            m = max(m, mx[u])
    return s.reduce_add() + m.reduce_max()


@inline(.always)
def _sm_bench[
    rilp: Int, deg: Int, withmax: Bool
](
    iters: Int,
    lp: MutPointer[Float32, MutUntrackedOrigin],
    pp: MutPointer[BFloat16, MutUntrackedOrigin],
    kt: Int,
) -> Float32:
    """`iters` softmax passes over a ROWS x `kt` logits buffer."""
    var c1v = SIMD[DType.float32, 16](0.1275)
    var off = SIMD[DType.float32, 16](-2.0)
    var s = Float32(0.0)
    for _ in range(iters):
        for rg in range(0, ROWS, 16):
            for rq in range(0, 16, rilp):
                var base = (rg + rq) * kt
                s += _sm_rows[rilp, deg, withmax](
                    lp.unsafe_offset(base), pp.unsafe_offset(base), kt, c1v, off
                )
    return s


struct PJob(Copyable, ImplicitlyCopyable, Movable):
    """One probe run: mode, iteration count, and the per-thread arenas."""

    var mode: Int
    var iters: Int
    var arena: Int
    var cout: Int
    var logits: Int
    var prob: Int
    var kt: Int
    var astride: Int

    def __init__(out self):
        self.mode = 0
        self.iters = 0
        self.arena = 0
        self.cout = 0
        self.logits = 0
        self.prob = 0
        self.kt = 256
        self.astride = 64


@inline(.always)
def _dc_loop(arena: Int, b_off: Int) -> Int:
    """One QK group: 4 dim chunks x (2 A loads + 2 B loads + 4 dots).

    Four accumulators, each a chain of four dots -- exactly the dependency
    structure `dense_amx.mojo` runs at head_dim 128. Returns the advanced B
    offset (kept mod ARENA by the caller).
    """
    tile_zero[4]()
    tile_zero[5]()
    tile_zero[6]()
    tile_zero[7]()
    var off = b_off
    comptime for dc in range(4):
        tile_load[0](arena, 64)
        tile_load[1](arena + 1024, 64)
        tile_load[2](arena + off, 64)
        tile_load[3](arena + off + 1024, 64)
        tile_dpbf16ps[4, 0, 2]()
        tile_dpbf16ps[5, 0, 3]()
        tile_dpbf16ps[6, 1, 2]()
        tile_dpbf16ps[7, 1, 3]()
        off = (off + 2048) & (ARENA - 1)
    return off


@inline(.always)
def _qk_real(
    iters: Int, a: Int, astride: Int, kbase: Int, logits: Int, lstride: Int
) -> None:
    """The kernel's QK inner loop shape: 8 chunks x (2 A + 2 B loads + 4 dots)
    and 4 logits stores per chunk, with runtime base pointers and a strided A.
    """
    for _ in range(iters):
        var kc = kbase
        var lg = logits
        for ch in range(8):
            tile_zero[4]()
            tile_zero[5]()
            tile_zero[6]()
            tile_zero[7]()
            comptime for dc in range(4):
                tile_load[0](a + dc * 64, astride)
                tile_load[1](a + dc * 64 + 16 * astride, astride)
                tile_load[2](kc + dc * 1024, 64)
                tile_load[3](kc + dc * 1024 + 4096, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
            tile_store[4](lg, lstride)
            tile_store[5](lg + 64, lstride)
            tile_store[6](lg + 16 * lstride, lstride)
            tile_store[7](lg + 16 * lstride + 64, lstride)
            kc += 8192
            lg += 128


@inline(.always)
def _probe(job: PJob, tid: Int, threads: Int) -> None:
    var cfg = stack_allocation[64, UInt8, alignment=64]()
    tile_config_bf16(16, 64, cfg)
    tile_cfg_load(Int(cfg))
    var arena = job.arena + tid * ARENA
    var cout = job.cout + tid * 4096
    var lp = MutPointer[Float32, MutUntrackedOrigin](
        unsafe_from_address=job.logits + tid * (ROWS * KBUF) * 4
    )
    var pp = MutPointer[BFloat16, MutUntrackedOrigin](
        unsafe_from_address=job.prob + tid * (ROWS * KBUF) * 2
    )
    var off = 0
    var iters = job.iters
    var kt = job.kt
    if job.mode == 0:
        # dots only: the issue-rate roof
        for _ in range(iters):
            comptime for u in range(OUTER):
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
    elif job.mode == 1:
        # + 4 loads per 4 dots, every address fixed and L1-hot
        for _ in range(iters):
            comptime for u in range(OUTER):
                tile_load[0](arena, 64)
                tile_load[1](arena + 1024, 64)
                tile_load[2](arena + 4096, 64)
                tile_load[3](arena + 5120, 64)
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
    elif job.mode == 2:
        # B walks a 32 KB arena, A fixed: the real QK loop
        for _ in range(iters):
            comptime for u in range(OUTER):
                off = _dc_loop(arena, off)
    elif job.mode == 3:
        # + the logits store the QK phase has to do
        for _ in range(iters):
            comptime for u in range(OUTER):
                off = _dc_loop(arena, off)
                tile_store[4](cout, 64)
                tile_store[5](cout + 64, 64)
                tile_store[6](cout + 1024, 64)
                tile_store[7](cout + 1088, 64)
    elif job.mode == 4:
        # loads only
        for _ in range(iters):
            comptime for u in range(OUTER):
                tile_load[0](arena, 64)
                tile_load[1](arena + 1024, 64)
                tile_load[2](arena + off, 64)
                tile_load[3](arena + off + 1024, 64)
                off = (off + 2048) & (ARENA - 1)
    elif job.mode == 5:
        # A lives in tiles for the whole loop; one B load feeds two dots.
        # 16 rows x 16 keys per group, 2 dots, 1 load, 2 stores.
        tile_load[0](arena, 64)
        tile_load[1](arena + 1024, 64)
        for _ in range(iters):
            comptime for u in range(OUTER):
                tile_zero[4]()
                tile_zero[5]()
                comptime for dc in range(4):
                    tile_load[2](arena + off, 64)
                    tile_dpbf16ps[4, 0, 2]()
                    tile_dpbf16ps[5, 1, 2]()
                    off = (off + 1024) & (ARENA - 1)
                tile_store[4](cout, 64)
                tile_store[5](cout + 1024, 64)
    elif job.mode == 7:
        # one dependent chain: measures tdpbf16ps latency
        for _ in range(iters):
            comptime for u in range(OUTER):
                tile_dpbf16ps[4, 0, 2]()
    elif job.mode == 9:
        _sm_bench[1, 4, False](iters, lp, pp, kt)
    elif job.mode == 10:
        _sm_bench[2, 4, False](iters, lp, pp, kt)
    elif job.mode == 11:
        _sm_bench[4, 4, False](iters, lp, pp, kt)
    elif job.mode == 12:
        _sm_bench[4, 4, True](iters, lp, pp, kt)
    elif job.mode == 13:
        _sm_bench[4, 2, False](iters, lp, pp, kt)
    elif job.mode == 14:
        _sm_bench[4, 4, False](iters, lp, pp, 128)
    elif job.mode == 15:
        _sm_bench[4, 4, False](iters, lp, pp, 1024)
    elif job.mode == 16:
        _sm_bench[4, 4, True](iters, lp, pp, 128)
    elif job.mode == 30:
        # 4 dots with 8 independent exp2 chains between them: can the AMX pipe
        # and the vector pipes run at the same time? 4 dots alone are 19.4 ns,
        # 8 exp2 alone are ~16 ns; "overlap" reads ~20, "serial" reads ~35.
        var x = Array[SIMD[DType.float32, 16], 8](uninitialized=True)
        comptime for u in range(8):
            x[u] = SIMD[DType.float32, 16](Float32(u) * -0.1)
        for _ in range(iters):
            comptime for u in range(OUTER):
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
                comptime for v in range(8):
                    x[v] = _exp2v[4](
                        fma(
                            x[v],
                            SIMD[DType.float32, 16](0.03125),
                            SIMD[DType.float32, 16](0.5),
                        )
                    )
        if x[0][0] == 12345.0:
            print("no")
    elif job.mode == 31:
        # the same 8 exp2 chains with no dots: what the vector work costs alone
        var x = Array[SIMD[DType.float32, 16], 8](uninitialized=True)
        comptime for u in range(8):
            x[u] = SIMD[DType.float32, 16](Float32(u) * -0.1)
        for _ in range(iters):
            comptime for u in range(OUTER):
                comptime for v in range(8):
                    x[v] = _exp2v[4](
                        fma(
                            x[v],
                            SIMD[DType.float32, 16](0.03125),
                            SIMD[DType.float32, 16](0.5),
                        )
                    )
        if x[0][0] == 12345.0:
            print("no")
    elif job.mode == 32:
        # 4 dots + 96 plain vector ALU ops (no transcendentals): is *any*
        # vector work able to issue while the tile pipe is busy?
        var x = Array[SIMD[DType.float32, 16], 8](uninitialized=True)
        comptime for u in range(8):
            x[u] = SIMD[DType.float32, 16](Float32(u) * -0.1)
        for _ in range(iters):
            comptime for u in range(OUTER):
                tile_dpbf16ps[4, 0, 2]()
                tile_dpbf16ps[5, 0, 3]()
                tile_dpbf16ps[6, 1, 2]()
                tile_dpbf16ps[7, 1, 3]()
                comptime for v in range(8):
                    comptime for w in range(12):
                        x[v] = fma(
                            x[v],
                            SIMD[DType.float32, 16](1.0001),
                            SIMD[DType.float32, 16](1e-7),
                        )
        if x[0][0] == 12345.0:
            print("no")
    elif job.mode == 20:
        # the kernel's QK loop: 8192 logits per iteration (8 chunks x 1024).
        # Per-thread slices of the big A and K arenas (see _run_strided).
        _qk_real(
            iters,
            job.arena + tid * (1 << 17),
            job.astride,
            job.cout + tid * (1 << 17),
            job.logits + tid * 32768 * 4,
            4096,
        )
    tile_release()


@export
def _probe_gomp(data: Int) abi("C"):
    var p = MutPointer[PJob, MutUntrackedOrigin](unsafe_from_address=data)
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    _probe(p[], tid, threads)


def _run(mode: Int, iters: Int, threads: Int, kt: Int) -> None:
    """One timed region; prints one RESULT line."""
    var arena = stack_allocation[ARENA * 8, UInt8, alignment=64]()
    var cout = stack_allocation[4096 * 4, Float32, alignment=64]()
    var logits = stack_allocation[ROWS * KBUF * 4, Float32, alignment=64]()
    var prob = stack_allocation[ROWS * KBUF * 4, BFloat16, alignment=64]()
    for i in range(0, ARENA * 8, 64):
        arena.unsafe_store(i, UInt8((i * 31) & 255))
    for i in range(4096 * 4):
        cout.unsafe_store(i, Float32(0.0))
    for i in range(ROWS * KBUF * 4):
        logits.unsafe_store(i, Float32(Float32(i % 97) * 0.01 - 0.5))
        prob.unsafe_store(i, BFloat16(0.0))
    var job = PJob()
    job.mode = mode
    job.iters = iters
    job.arena = Int(arena)
    job.cout = Int(cout)
    job.logits = Int(logits)
    job.prob = Int(prob)
    job.kt = kt
    var state = stack_allocation[1, PJob]()
    state[] = job
    comptime entry = _probe_gomp
    # warm the pool and the tiles: first touch of #XF is a fault, and the pool is
    # created on the first region
    external_call["omp_set_num_threads", NoneType](Int32(threads))
    external_call["GOMP_parallel", NoneType](
        entry, Int(state), Int32(threads), Int32(0)
    )
    var t0 = perf_counter_ns()
    external_call["GOMP_parallel", NoneType](
        entry, Int(state), Int32(threads), Int32(0)
    )
    var t1 = perf_counter_ns()
    print(
        "RESULT mode=",
        mode,
        " threads=",
        threads,
        " iters=",
        iters,
        " ns=",
        Int(t1 - t0),
        " kt=",
        kt,
    )


def _run_strided(mode: Int, iters: Int, threads: Int, astride: Int) -> None:
    """`_run` with the A-operand stride as a parameter (mode 20).

    Four 128 KB slices each of A material and K material, plus 128 KB of
    logits: the A tile load is 16 rows `astride` bytes apart, so the arena has
    to be 16 * 4096 bytes to take the widest stride probed.
    """
    var abuf = stack_allocation[1 << 19, UInt8, alignment=64]()
    var kbuf = stack_allocation[1 << 19, UInt8, alignment=64]()
    var logits = stack_allocation[ROWS * KBUF * 4, Float32, alignment=64]()
    for i in range(0, 1 << 19, 64):
        abuf.unsafe_store(i, UInt8((i * 31) & 255))
        kbuf.unsafe_store(i, UInt8((i * 17) & 255))
    for i in range(ROWS * KBUF * 4):
        logits.unsafe_store(i, Float32(0.0))
    var job = PJob()
    job.mode = mode
    job.iters = iters
    job.arena = Int(abuf)
    job.cout = Int(kbuf)
    job.logits = Int(logits)
    job.astride = astride
    var state = stack_allocation[1, PJob]()
    state[] = job
    comptime entry = _probe_gomp
    external_call["omp_set_num_threads", NoneType](Int32(threads))
    external_call["GOMP_parallel", NoneType](
        entry, Int(state), Int32(threads), Int32(0)
    )
    var t0 = perf_counter_ns()
    external_call["GOMP_parallel", NoneType](
        entry, Int(state), Int32(threads), Int32(0)
    )
    var t1 = perf_counter_ns()
    print(
        "RESULT mode=",
        mode,
        " threads=",
        threads,
        " iters=",
        iters,
        " ns=",
        Int(t1 - t0),
        " kt=",
        astride,
    )


def main():
    if not amx_init():
        print("AMX unusable here")
        return
    print("omp_get_max_threads:", external_call["omp_get_max_threads", Int32]())
    for t in [Int(1), Int(4)]:
        for ast in [Int(64), Int(256), Int(1024), Int(4096)]:
            _run_strided(20, 200, t, ast)
        _run(0, 400, t, 0)
        _run(30, 200, t, 0)
        _run(31, 200, t, 0)
        _run(32, 100, t, 0)
