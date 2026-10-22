"""The AMX attention work item in isolation: cycles per logit, per phase.

`dense_amx.mojo`'s item body with the geometry at compile time (the kernel takes
D/kt/seq_k at run time, which costs address arithmetic and dynamic loop bounds in
every inner loop -- one thing this probe separates). One `run` walks
`KEYS_TOTAL` keys for one work item (32 query rows) over a 1 MB K/V working set,
repeated so the buffers are warm; the reported number is the best run, because
interference can only make a run slower.

Phases are selected at compile time so each can be timed on its own:

  full   qk + softmax + pv, as shipped
  qk     Q@K^T only (logits written, never read)
  sm     softmax only, over a filled logits buffer
  pv     P@V only (prob read, accumulator round-tripped)

    mojo build -I cpu_attention/_mojo/src \
        cpu_attention/_mojo/tests/bench_amx_item.mojo -o /tmp/i && /tmp/i
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
from std.math import fma, iota
from std.memory import bitcast, stack_allocation
from std.time import perf_counter_ns

comptime GHZ = 2.36
comptime D = 128
comptime ROWS = 32
comptime KT = 256
comptime KEYCH = 32
comptime KEYS_TOTAL = 4096
comptime REP = 5
comptime DC16 = D // 16
comptime NCH = KT // KEYCH


@always_inline
def _exp2(x: SIMD[DType.float32, 16]) -> SIMD[DType.float32, 16]:
    comptime V = SIMD[DType.float32, 16]
    var xc = x.clamp(V(-127.0), V(127.0))
    comptime MAGIC = 12582912.0
    var t = xc + V(MAGIC)
    var ni = bitcast[DType.int32, 16](t) - SIMD[DType.int32, 16](
        bitcast[DType.int32, 16](V(MAGIC))[0]
    )
    var f = xc - (t - V(MAGIC))
    var p = V(0.0139755595)
    p = fma(p, f, V(0.0519902928))
    p = fma(p, f, V(0.2416609733))
    p = fma(p, f, V(0.6929159775))
    p = fma(p, f, V(1.0))
    var bits = (ni + SIMD[DType.int32, 16](127)) << SIMD[DType.int32, 16](23)
    return p * bitcast[DType.float32, 16](bits)


struct Bufs(Movable):
    var q: Pointer[BFloat16, MutUntrackedOrigin]
    var kp: Pointer[BFloat16, MutUntrackedOrigin]
    var vp: Pointer[BFloat16, MutUntrackedOrigin]
    var logits: Pointer[Float32, MutUntrackedOrigin]
    var prob: Pointer[BFloat16, MutUntrackedOrigin]
    var acc: Pointer[Float32, MutUntrackedOrigin]
    var rm: Pointer[Float32, MutUntrackedOrigin]

    def __init__(
        out self,
        q: Pointer[BFloat16, MutUntrackedOrigin],
        kp: Pointer[BFloat16, MutUntrackedOrigin],
        vp: Pointer[BFloat16, MutUntrackedOrigin],
        logits: Pointer[Float32, MutUntrackedOrigin],
        prob: Pointer[BFloat16, MutUntrackedOrigin],
        acc: Pointer[Float32, MutUntrackedOrigin],
        rm: Pointer[Float32, MutUntrackedOrigin],
    ):
        self.q = q
        self.kp = kp
        self.vp = vp
        self.logits = logits
        self.prob = prob
        self.acc = acc
        self.rm = rm



@always_inline
def softmax_chunk(
    ch: Int,
    m_used: SIMD[DType.float32, 16],
    s_acc: SIMD[DType.float32, 16],
    logits: Pointer[Float32, MutUntrackedOrigin],
    prob: Pointer[BFloat16, MutUntrackedOrigin],
    rm: Pointer[Float32, MutUntrackedOrigin],
    c1v: SIMD[DType.float32, 16],
) -> None:
    """One 32-key chunk: row sums of exp2(logit*c1 - m), written as bf16 P."""
    comptime V16 = SIMD[DType.float32, 16]
    var m_sc = stack_allocation[16, Float32]()
    m_sc.unsafe_store[width=16](0, m_used)
    for rg in range(2):
        var lp = logits.unsafe_offset(rg * 16 * KT + ch * KEYCH)
        for r in range(16):
            var lr = lp.unsafe_offset(r * KT)
            var pp = prob.unsafe_offset((rg * 16 + r) * KT + ch * KEYCH)
            var off = V16(-m_sc.unsafe_load(r) * 2.0)
            var acc_s = V16(0.0)
            comptime for j in range(0, KEYCH, 16):
                var e = _exp2(fma(lr.unsafe_load[width=16](j), c1v, off))
                acc_s = acc_s + e
                pp.unsafe_store[width=16](j, e.cast[DType.bfloat16]())
            rm.unsafe_store(r, Scalar[DType.float32](acc_s.reduce_add()))


def run[phase: Int](bufs: Bufs, keys_total: Int) -> Float64:
    """Cycles per logit for one whole item over KEYS_TOTAL keys."""
    comptime V16 = SIMD[DType.float32, 16]
    comptime DCH = D // 32
    comptime KSTRIDE = 512
    comptime VSTRIDE = 512
    var q = bufs.q
    var kp = bufs.kp
    var vp = bufs.vp
    var logits = bufs.logits
    var prob = bufs.prob
    var acc = bufs.acc
    var rm = bufs.rm
    var c1v = V16(2.0)
    var m0 = V16(0.0)
    var s0 = V16(0.0)
    var qstride = D * 2
    var lstride = KT * 4
    var pstride = KT * 2
    var astride = D * 4
    var best = -1.0
    var nkeys = 0
    for _ in range(REP):
        var tq = perf_counter_ns()
        var t = 0
        while t < keys_total:
            var nk = min(KT, keys_total - t)
            var nch = (nk + KEYCH - 1) // KEYCH
            if phase == 0 or phase == 1:
                for ch in range(nch):
                    var k0 = t + ch * KEYCH
                    var kb = (k0 // 16) * DCH * KSTRIDE
                    tile_zero[4]()
                    tile_zero[5]()
                    tile_zero[6]()
                    tile_zero[7]()
                    comptime for dc in range(DCH):
                        var kt_hi = Int(kp) + (kb + dc * KSTRIDE) * 2
                        tile_load[0](Int(q) + dc * 64, qstride)
                        tile_load[1](Int(q) + dc * 64 + 16 * qstride, qstride)
                        tile_load[2](kt_hi, 64)
                        tile_load[3](kt_hi + DCH * KSTRIDE * 2, 64)
                        tile_dpbf16ps[4, 0, 2]()
                        tile_dpbf16ps[5, 0, 3]()
                        tile_dpbf16ps[6, 1, 2]()
                        tile_dpbf16ps[7, 1, 3]()
                    var lp = Int(logits) + ch * KEYCH * 4
                    tile_store[4](lp, lstride)
                    tile_store[5](lp + 64, lstride)
                    tile_store[6](lp + 16 * lstride, lstride)
                    tile_store[7](lp + 16 * lstride + 64, lstride)
            if phase == 0 or phase == 2:
                for rg in range(2):
                    var lp = logits.unsafe_offset(rg * 16 * KT)
                    var m_old = m0
                    for r in range(16):
                        var v = lp.unsafe_load[width=16](r * KT)
                        comptime for j in range(1, KT // 16):
                            v = max(v, lp.unsafe_load[width=16](r * KT + j * 16))
                        rm.unsafe_store(r, Scalar[DType.float32](v.reduce_max()))
                    var m_cand = rm.unsafe_load[width=16]()
                    var need = m_cand.gt(m_old + V16(8.0))
                    var m_new = need.select(m_cand, m_old)
                    var alpha = _exp2((m_old - m_new) * c1v)
                    m0 = m_new
                    var m_sc = stack_allocation[16, Float32]()
                    m_sc.unsafe_store[width=16](0, m_new)
                    for r in range(16):
                        var lr = lp.unsafe_offset(r * KT)
                        var pp = prob.unsafe_offset((rg * 16 + r) * KT)
                        var off = V16(-m_sc.unsafe_load(r) * 2.0)
                        var acc_s = V16(0.0)
                        comptime for j in range(0, KT, 16):
                            var e = _exp2(fma(lr.unsafe_load[width=16](j), c1v, off))
                            acc_s = acc_s + e
                            pp.unsafe_store[width=16](j, e.cast[DType.bfloat16]())
                        rm.unsafe_store(r, Scalar[DType.float32](acc_s.reduce_add()))
                    var rows = rm.unsafe_load[width=16]()
                    s0 = fma(s0, alpha, rows)
            if phase == 0 or phase == 3:
                var nks = (nk + 31) // 32
                var first = t == 0
                var dc = 0
                while dc < DC16:
                    if first:
                        tile_zero[4]()
                        tile_zero[5]()
                        tile_zero[6]()
                        tile_zero[7]()
                    else:
                        tile_load[4](Int(acc) + dc * 64, astride)
                        tile_load[5](Int(acc) + (dc + 1) * 64, astride)
                        tile_load[6](Int(acc) + (16 * D + dc * 16) * 4, astride)
                        tile_load[7](
                            Int(acc) + (16 * D + (dc + 1) * 16) * 4, astride
                        )
                    for ks in range(nks):
                        var vb = ((t // 32) + ks) * DC16 * VSTRIDE
                        var vt_hi = Int(vp) + (vb + dc * VSTRIDE) * 2
                        tile_load[0](Int(prob) + (ks * 32) * 2, pstride)
                        tile_load[1](Int(prob) + (16 * KT + ks * 32) * 2, pstride)
                        tile_load[2](vt_hi, 64)
                        tile_load[3](vt_hi + VSTRIDE * 2, 64)
                        tile_dpbf16ps[4, 0, 2]()
                        tile_dpbf16ps[5, 0, 3]()
                        tile_dpbf16ps[6, 1, 2]()
                        tile_dpbf16ps[7, 1, 3]()
                    tile_store[4](Int(acc) + dc * 64, astride)
                    tile_store[5](Int(acc) + (dc + 1) * 64, astride)
                    tile_store[6](Int(acc) + (16 * D + dc * 16) * 4, astride)
                    tile_store[7](
                        Int(acc) + (16 * D + (dc + 1) * 16) * 4, astride
                    )
                    dc += 2

            if phase == 4:
                # Pipelined: the QK of chunk `ch` is followed by the softmax of
                # chunk ch-1, so the AMX work and the vector work are
                # independent and the out-of-order engine can overlap them.
                # The exponent uses the *previous* super-tile's running max
                # (fixed for the whole super-tile, because P@V accumulates all
                # chunks with one scale); the acc/sum rescale after the
                # super-tile keeps that exact.
                for ch in range(nch):
                    var k0 = t + ch * KEYCH
                    var kb = (k0 // 16) * DCH * KSTRIDE
                    tile_zero[4]()
                    tile_zero[5]()
                    tile_zero[6]()
                    tile_zero[7]()
                    comptime for dc in range(DCH):
                        var kt_hi = Int(kp) + (kb + dc * KSTRIDE) * 2
                        tile_load[0](Int(q) + dc * 64, qstride)
                        tile_load[1](Int(q) + dc * 64 + 16 * qstride, qstride)
                        tile_load[2](kt_hi, 64)
                        tile_load[3](kt_hi + DCH * KSTRIDE * 2, 64)
                        tile_dpbf16ps[4, 0, 2]()
                        tile_dpbf16ps[5, 0, 3]()
                        tile_dpbf16ps[6, 1, 2]()
                        tile_dpbf16ps[7, 1, 3]()
                    var lp = Int(logits) + ch * KEYCH * 4
                    tile_store[4](lp, lstride)
                    tile_store[5](lp + 64, lstride)
                    tile_store[6](lp + 16 * lstride, lstride)
                    tile_store[7](lp + 16 * lstride + 64, lstride)
                    if ch > 0:
                        softmax_chunk(ch - 1, m0, s0, logits, prob, rm, c1v)
                softmax_chunk(nch - 1, m0, s0, logits, prob, rm, c1v)
            t += KT
        var dt = Float64(perf_counter_ns() - tq)
        if best < 0 or dt < best:
            best = dt
    nkeys = keys_total
    tile_store[7](Int(acc), astride)
    var n = Float64(nkeys) * Float64(ROWS)
    return best / n * GHZ


def main():
    if not amx_init():
        print("AMX unusable here")
        return
    var cfg = stack_allocation[64, UInt8, alignment=64]()
    tile_config_bf16(16, 64, cfg)
    tile_cfg_load(Int(cfg))
    var b = Bufs(
        stack_allocation[ROWS * D, BFloat16, alignment=64](),
        stack_allocation[KEYS_TOTAL * D, BFloat16, alignment=64](),
        stack_allocation[KEYS_TOTAL * D, BFloat16, alignment=64](),
        stack_allocation[ROWS * KT, Float32, alignment=64](),
        stack_allocation[ROWS * KT, BFloat16, alignment=64](),
        stack_allocation[ROWS * D, Float32, alignment=64](),
        stack_allocation[16, Float32](),
    )
    for i in range(ROWS * D):
        b.q[i] = BFloat16(Float32(i % 61) * 0.0078125 - 0.25)
    for i in range(KEYS_TOTAL * D):
        b.kp[i] = BFloat16(Float32(i % 53) * -0.0078125 + 0.2)
        b.vp[i] = BFloat16(Float32(i % 47) * 0.00390625 - 0.1)
    for i in range(ROWS * KT):
        b.logits[i] = Float32(0.25)
        b.prob[i] = BFloat16(0.5)
    for i in range(ROWS * D):
        b.acc[i] = Float32(0)
    print("D", D, "ROWS", ROWS, "KT", KT, " cyc/logit")
    for keys in [128, 256, 512, 1024, 2048, 4096]:
        print(
            "  keys",
            keys,
            " qk",
            run[1](b, keys),
            " sm",
            run[2](b, keys),
            " pv",
            run[3](b, keys),
            " full",
            run[0](b, keys),
            " pipe",
            run[4](b, keys)
        )
    tile_release()
