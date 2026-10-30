"""One parallel region on torch's OpenMP pool, for every driver.

`GOMP_parallel` on the libgomp pool torch keeps warm costs ~2.4 us; MAX's
`parallelize` measured 6.3 us warm / 103-140 us cold and a private pthread pool
9.4 / 103 us (docs/Mojo.md §6), so there is no other pool. A driver hands a thin
`body(ctx, tid, threads)` and a context address; the body slices its own work.

Slices must use the *effective* team size (`omp_get_num_threads()`), not the
request: libgomp clamps the request to `OMP_NUM_THREADS`, and a slice computed
from the request would leave work unexecuted.
"""

from std.ffi import external_call
from std.memory import MutPointer, stack_allocation
from std.origin import MutUntrackedOrigin

comptime Body = def(Int, Int, Int) thin -> None


struct _Region(ImplicitlyCopyable, Movable):
    var body: Body
    var ctx: Int

    def __init__(out self, body: Body, ctx: Int):
        self.body = body
        self.ctx = ctx


@export
def _cpu_attention_region(data: Int) abi("C"):
    var r = MutPointer[_Region, MutUntrackedOrigin](unsafe_from_address=data)[]
    var threads = Int(external_call["omp_get_num_threads", Int32]())
    var tid = Int(external_call["omp_get_thread_num", Int32]())
    r.body(r.ctx, tid, threads)


def parallel(body: Body, ctx: Int, workers: Int):
    """Run `body(ctx, tid, threads)` on `workers` threads (inline for one)."""
    if workers <= 1:
        body(ctx, 0, 1)
        return
    var st = stack_allocation[1, _Region]()
    st[] = _Region(body, ctx)
    comptime entry = _cpu_attention_region
    external_call["GOMP_parallel", NoneType](entry, Int(st), Int32(workers), Int32(0))


@always_inline
def barrier():
    """All threads of the current region (only inside a `parallel` body)."""
    external_call["GOMP_barrier", NoneType]()
