"""Runtime choice of dtype, head_dim and ISA policy for flash_attn_func.

Every policy compiled into this tier is listed here; `isa` comes from the plan
(the Python loader picks it from the tier and the runtime toggles).
"""

from flash_attn import FlashJob, FlashPolicy, aux_bytes, flash_drive
from flash_neon import NeonF32Policy, NeonMmlaPolicy
from flash_vec import VecPolicy
from isa import NEON_BF16

comptime ISA_VEC = 0
comptime ISA_NEON_MMLA = 2


def _policy[dtype: DType, hd: Int](job: FlashJob, isa: Int) raises:
    if isa == ISA_VEC:
        flash_drive[VecPolicy[dtype, hd]](job)
    elif isa == ISA_NEON_MMLA:
        comptime if NEON_BF16:
            comptime if dtype == DType.bfloat16:
                flash_drive[NeonMmlaPolicy[hd]](job)
            else:
                flash_drive[NeonF32Policy[hd]](job)
        else:
            raise Error("cpu_attention: ISA policy not in this build")
    else:
        raise Error("cpu_attention: ISA policy not in this build")


def _hd[dtype: DType](job: FlashJob, hd: Int, isa: Int) raises:
    if hd == 64:
        _policy[dtype, 64](job, isa)
    elif hd == 128:
        _policy[dtype, 128](job, isa)
    elif hd == 32:
        _policy[dtype, 32](job, isa)
    else:
        raise Error("cpu_attention: head_dim must be 32, 64 or 128")


def flash_run(job: FlashJob, dtype: Int, hd: Int, isa: Int) raises:
    if dtype == 0:
        _hd[DType.float32](job, hd, isa)
    elif dtype == 1:
        _hd[DType.bfloat16](job, hd, isa)
    else:
        raise Error("cpu_attention: unsupported dtype")


def _sizes[P: FlashPolicy](seq_k: Int, rq: Int) -> Tuple[Int, Int]:
    return (P.slab_bytes(seq_k), aux_bytes[P](rq))


def _sizes_isa[dtype: DType, hd: Int](seq_k: Int, rq: Int, isa: Int) raises -> Tuple[Int, Int]:
    if isa == ISA_VEC:
        return _sizes[VecPolicy[dtype, hd]](seq_k, rq)
    if isa == ISA_NEON_MMLA:
        comptime if NEON_BF16:
            comptime if dtype == DType.bfloat16:
                return _sizes[NeonMmlaPolicy[hd]](seq_k, rq)
            else:
                return _sizes[NeonF32Policy[hd]](seq_k, rq)
    raise Error("cpu_attention: ISA policy not in this build")


def flash_sizes(seq_k: Int, rq: Int, hd: Int, dtype: Int, isa: Int) raises -> Tuple[Int, Int]:
    """(slab bytes per (batch, kv head), scratch bytes per thread)."""
    if dtype == 1:
        if hd == 64:
            return _sizes_isa[DType.bfloat16, 64](seq_k, rq, isa)
        elif hd == 128:
            return _sizes_isa[DType.bfloat16, 128](seq_k, rq, isa)
        return _sizes_isa[DType.bfloat16, 32](seq_k, rq, isa)
    if hd == 64:
        return _sizes_isa[DType.float32, 64](seq_k, rq, isa)
    elif hd == 128:
        return _sizes_isa[DType.float32, 128](seq_k, rq, isa)
    return _sizes_isa[DType.float32, 32](seq_k, rq, isa)
