"""Standalone parity test of the KV-cache kernel against a scalar fp32 reference.

Stages every token through the store path (`do_attend=False`), then attends,
for fp32 and bf16, page sizes 16/32/48/64 (multi-tile pages), GQA, and contexts
that end mid-tile. Built and run by `scripts/build.py --tests`.
"""

from std.math import exp
from std.memory import stack_allocation

from isa import ptr
from paged_attn import PagedJob, paged_run


def _fill[dtype: DType](p: Int, n: Int, seed: Int) -> Int:
    var s = seed
    var d = ptr[Scalar[dtype]](p)
    for i in range(n):
        s = (s * 1103515245 + 12345) % 2147483648
        d.unsafe_store(i, Scalar[dtype](Float32(s % 2001 - 1000) / 1000.0))
    return s


def _case[dtype: DType](page: Int, len0: Int, len1: Int, heads: Int = 4, kvh: Int = 4) raises -> Bool:
    comptime HD = 64
    var seqs = 2
    var lens = [len0, len1]
    var pages_per = (max(len0, len1) + page - 1) // page
    var nblocks = seqs * pages_per
    var cache_n = nblocks * kvh * page * HD
    var kc = stack_allocation[1 << 17, Scalar[dtype]]()
    var vc = stack_allocation[1 << 17, Scalar[dtype]]()
    var tokens = len0 + len1
    var kn = stack_allocation[1 << 17, Scalar[dtype]]()
    var vn = stack_allocation[1 << 17, Scalar[dtype]]()
    var q = stack_allocation[1024, Scalar[dtype]]()
    var out = stack_allocation[1024, Scalar[dtype]]()
    var slots = stack_allocation[4096, Int64]()
    var bt = stack_allocation[256, Int32]()
    var ln = stack_allocation[2, Int32]()
    if cache_n > (1 << 17) or tokens * kvh * HD > (1 << 17):
        raise Error("test case too large")
    var seed = _fill[dtype](Int(kn), tokens * kvh * HD, 7)
    seed = _fill[dtype](Int(vn), tokens * kvh * HD, seed)
    seed = _fill[dtype](Int(q), seqs * heads * HD, seed)
    # Block table: sequence s owns pages s*pages_per .. (in reverse order, so the
    # cache order differs from the sequence order).
    for s in range(seqs):
        for p in range(pages_per):
            bt[s * pages_per + p] = Int32(s * pages_per + (pages_per - 1 - p))
    var t = 0
    for s in range(seqs):
        ln[s] = Int32(lens[s])
        for i in range(lens[s]):
            slots[t] = Int64(Int(bt[s * pages_per + i // page]) * page + i % page)
            t += 1
    var job = PagedJob()
    job.k = Int(kn)
    job.v = Int(vn)
    job.k_cache = Int(kc)
    job.v_cache = Int(vc)
    job.slots = Int(slots)
    job.num_seqs = tokens
    job.num_heads = kvh
    job.kv_heads = kvh
    job.head_dim = HD
    job.page = page
    job.kv_stride = kvh * HD
    job.dtype = 0 if dtype == DType.float32 else 1
    job.workers = 2
    job.do_store = True
    job.do_attend = False
    paged_run(job)
    var a = PagedJob()
    a.q = Int(q)
    a.k_cache = Int(kc)
    a.v_cache = Int(vc)
    a.out = Int(out)
    a.block_table = Int(bt)
    a.lens = Int(ln)
    a.num_seqs = seqs
    a.num_heads = heads
    a.kv_heads = kvh
    a.head_dim = HD
    a.page = page
    a.bt_stride = pages_per
    a.q_stride = heads * HD
    a.out_stride = heads * HD
    a.scale = 0.125
    a.dtype = job.dtype
    a.workers = 2
    a.do_attend = True
    a.chunks = 1
    paged_run(a)
    var worst = Float32(0)
    var base = 0
    for s in range(seqs):
        for h in range(heads):
            var g = h // (heads // kvh)
            var sc = List[Float32]()
            var mx = Float32(-1e30)
            for i in range(lens[s]):
                var dot = Float32(0)
                for d in range(HD):
                    dot += Float32(q[(s * heads + h) * HD + d]) * Float32(kn[((base + i) * kvh + g) * HD + d])
                sc.append(dot * 0.125)
                mx = max(mx, dot * 0.125)
            var sum = Float32(0)
            for i in range(lens[s]):
                sc[i] = exp(sc[i] - mx)
                sum += sc[i]
            for d in range(HD):
                var acc = Float32(0)
                for i in range(lens[s]):
                    acc += sc[i] * Float32(vn[((base + i) * kvh + g) * HD + d])
                var got = Float32(out[(s * heads + h) * HD + d])
                worst = max(worst, abs(got - acc / sum))
        base += lens[s]
    var tol = Float32(1e-5) if dtype == DType.float32 else Float32(2e-2)
    var ok = worst <= tol
    print("dtype", dtype, "page", page, "lens", len0, len1, "heads", heads, "/", kvh,
          "max|diff|", worst, "OK" if ok else "FAIL")
    return ok


def main() raises:
    var ok = True
    for page in [16, 32, 48, 64]:
        ok = _case[DType.float32](page, 1, 37) and ok
        ok = _case[DType.float32](page, page * 2 + 1, 95) and ok
        ok = _case[DType.bfloat16](page, 33, page * 3 - 1) and ok
        ok = _case[DType.bfloat16](page, 79, 190) and ok
    ok = _case[DType.float32](32, 70, 129, 8, 2) and ok
    ok = _case[DType.bfloat16](16, 70, 129, 8, 1) and ok
    if not ok:
        raise Error("parity failures")
    print("all parity cases passed")
