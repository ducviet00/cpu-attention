"""Runtime choice of dtype, head_dim and ISA policy for flash_attn_func.

`_with_policy` is the one table from (dtype, head_dim, isa) to the `FlashPolicy`
compiled into this tier; running a call and sizing its workspace both go through
it. `isa` comes from the plan (the Python loader picks it from the tier and the
runtime toggles).
"""

from flash_attn import FlashJob, FlashPolicy, aux_bytes, flash_drive
from flash_amx import AmxPolicy
from flash_neon import NeonMmlaPolicy
from flash_vec import VecPolicy
from isa import AMX_TIER, NEON_BF16

comptime ISA_VEC = 0
comptime ISA_AMX = 1
comptime ISA_MMA_ARM = 2
"""aarch64-bf16: `bfmmla` for bf16, the vector policy for fp32."""


def _with_isa[dtype: DType, hd: Int, body: def[P: FlashPolicy]() raises capturing -> None](isa: Int) raises:
    if isa == ISA_VEC:
        body[VecPolicy[dtype, hd]]()
        return
    if isa == ISA_AMX:
        comptime if AMX_TIER:
            body[AmxPolicy[dtype, hd]]()
            return
    if isa == ISA_MMA_ARM:
        comptime if NEON_BF16:
            comptime if dtype == DType.bfloat16:
                body[NeonMmlaPolicy[hd]]()
            else:
                body[VecPolicy[dtype, hd]]()
            return
    raise Error("cpu_attention: ISA policy not in this build")


def _with_policy[body: def[P: FlashPolicy]() raises capturing -> None](dtype: Int, hd: Int, isa: Int) raises:
    if dtype != 0 and dtype != 1:
        raise Error("cpu_attention: unsupported dtype")
    comptime for d in range(2):
        comptime dt = DType.float32 if d == 0 else DType.bfloat16
        if dtype == d:
            comptime for hdc in [32, 64, 128]:
                if hd == hdc:
                    _with_isa[dt, hdc, body](isa)
                    return
    raise Error("cpu_attention: head_dim must be 32, 64 or 128")


def flash_run(job: FlashJob, dtype: Int, hd: Int, isa: Int) raises:
    def run[P: FlashPolicy]() raises capturing:
        flash_drive[P](job)

    _with_policy[run](dtype, hd, isa)


def flash_sizes(seq_k: Int, rq: Int, hd: Int, dtype: Int, isa: Int) raises -> Tuple[Int, Int]:
    """(slab bytes per (batch, head), scratch bytes per thread)."""
    var sizes = (0, 0)

    def size[P: FlashPolicy]() raises capturing:
        sizes = (P.slab_bytes(seq_k), aux_bytes[P](rq))

    _with_policy[size](dtype, hd, isa)
    return sizes
