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
    var a = stack_allocation[16 * 32, BFloat16, alignment=64]()
    var b = stack_allocation[16 * 32, BFloat16, alignment=64]()
    var c = stack_allocation[16 * 16, Float32, alignment=64]()
    for i in range(16 * 32):
        a[i] = BFloat16(Float32(i % 7) * 0.25)
        b[i] = BFloat16(Float32(i % 5) * -0.5)
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
                acc += Float32(a[i * 32 + kk]) * Float32(b[j * 32 + kk])
            worst = max(worst, abs(acc - c[i * 16 + j]))
    print("max|diff| vs scalar:", worst)
