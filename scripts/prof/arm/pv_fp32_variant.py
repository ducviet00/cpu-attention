"""Build-time variant: P@V in fp32 FMA instead of `bfmmla` (scratch tree only).

Purpose is falsification, not shipment. `_dense_rows_bfmmla` runs `bfmmla` on
*both* phases; this script replaces only the P@V super-tile's `_bfmmla` with a
lane-equivalent fp32 `fma`, leaving Q@K^T, the softmax, the probability packing
and every tiling parameter byte-identical. Timing the normal tree against the
patched one interleaved therefore prices exactly the P@V instruction form.

The bfmmla operand mapping (from the kernel's layout docstring) is:
  ap[rp] = 8 bf16 = four keys of row 2rp, then four of row 2rp+1   (A: 2 rows x 4 keys)
  vv     = 8 bf16 = one head-dim pair across four keys             (B: 4 keys x 2 dims)
  accv   = 4 fp32 = (row0,d0), (row0,d1), (row1,d0), (row1,d1)
so the fp32 equivalent is one 4-lane `fma` per (key, dim pair, row pair):
  p = (p0, p0, p1, p1)   v = (v0, v1, v0, v1)

usage: python scripts/prof/arm/pv_fp32_variant.py <tree>/cpu_attention/_mojo/src/dense.mojo
"""

from __future__ import annotations

import sys

OLD = """                comptime for d in range(dcp):
                    var vv = vt_base.unsafe_load[width=8](
                        (dc + d) * job.v_stride_t + (t0 // 4 + kg) * 8
                    )
                    comptime for rp in range(RP):
                        accv[rp * dcp + d] = _bfmmla(
                            accv[rp * dcp + d], ap[rp], vv
                        )"""

NEW = """                comptime for d in range(dcp):
                    var vv = vt_base.unsafe_load[width=8](
                        (dc + d) * job.v_stride_t + (t0 // 4 + kg) * 8
                    )
                    # VARIANT: fp32 FMA P@V. Same math, same layout, same tiling.
                    var vf = vv.cast[DType.float32]()
                    comptime for rp in range(RP):
                        var af = ap[rp].cast[DType.float32]()
                        comptime for k in range(4):
                            var pvec = SIMD[DType.float32, 4](
                                af[k], af[k], af[4 + k], af[4 + k]
                            )
                            var vvec = SIMD[DType.float32, 4](
                                vf[2 * k], vf[2 * k + 1], vf[2 * k], vf[2 * k + 1]
                            )
                            accv[rp * dcp + d] = fma(
                                pvec, vvec, accv[rp * dcp + d]
                            )"""


def main() -> int:
    path = sys.argv[1]
    src = open(path).read()
    n = src.count(OLD)
    if n != 1:
        print(f"expected exactly 1 P@V bfmmla site, found {n}", file=sys.stderr)
        return 2
    open(path, "w").write(src.replace(OLD, NEW))
    print(f"patched {path}: P@V bfmmla -> fp32 fma")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
