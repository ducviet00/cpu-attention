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


def main():
    if not amx_init():
        print("AMX unusable here")
        return
    var cfg = stack_allocation[64, UInt8, alignment=64]()
    tile_config_bf16(16, 64, cfg)
    # a: the A operand, 16 rows x 32 contraction elements, row-major.
    var a = stack_allocation[16 * 32, BFloat16, alignment=64]()
    # kd: the maths (16 keys x 32 head dims, row-major); b: the *packed* B tile.
    var kd = stack_allocation[16 * 32, BFloat16, alignment=64]()
    var b = stack_allocation[16 * 32, BFloat16, alignment=64]()
    var c = stack_allocation[16 * 16, Float32, alignment=64]()
    for i in range(16 * 32):
        a[i] = BFloat16(Float32((i * 37) % 101) * 0.03125 - 1.5)
        kd[i] = BFloat16(Float32((i * 53) % 97) * -0.0234375 + 1.0)
    # The B operand's packing, measured by 512 single-bit probes in
    # _csrc/dense_amx.c: element (tile row r, index t) is B[2r + (t & 1)][t >> 1]
    # -- inside one 64-byte tile row the first 16 bf16 are contraction element
    # 2r across the 16 outputs, the next 16 are 2r+1. (vLLM's
    # reshape_and_cache states the same: "head elements should be packed as
    # quadwords".) Neither "row-major 16x32" nor its transpose is right.
    for r in range(16):
        for t in range(32):
            b[r * 32 + t] = kd[(t >> 1) * 32 + 2 * r + (t & 1)]
    tile_cfg_load(Int(cfg))
    tile_zero[2]()
    tile_load[0](Int(a), 64)
    tile_load[1](Int(b), 64)
    tile_dpbf16ps[2, 0, 1]()
    tile_store[2](Int(c), 64)
    tile_release()
    var worst = Float32(0.0)
    for i in range(16):
        for j in range(16):
            var acc = Float32(0.0)
            for kk in range(32):
                acc += Float32(a[i * 32 + kk]) * Float32(kd[j * 32 + kk])
            worst = max(worst, abs(acc - c[i * 16 + j]))
    print("C = a @ kd^T, 16x16x32 bf16, max|diff| vs scalar:", worst)
