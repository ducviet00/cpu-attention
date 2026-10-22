"""Time the K/V pack alone, one GOMP region, exactly the kernel's work.

`tests/bench_phases.py` measures the pack with `CPU_ATTENTION_NO_FUSE=1`, which
splits the call into two OpenMP regions, so its "pack" column carries an extra
fork-join. That inflated the pack's cost by 2x: `bench_phases` reports 6 us at
S=128/8/64 where the pack alone is **2.90 us**, i.e. 181 GB/s against a 206
GB/s plain-copy floor for the same 512 KB (m9g, 4 threads,
`tests/copy_floor.py`). The pack is at 88% of memcpy and is not a lever.

This calls the extension's transpose entry directly, replicating the arguments
`_dense` builds, so the number is the kernel's own work with the region the
fused call would have anyway.
"""
import statistics, sys, time
import torch
sys.path.insert(0, ".")
import cpu_attention._kernel as K
torch.set_num_threads(8)

for S, H, D, dt in ((128, 8, 64, torch.bfloat16), (512, 8, 64, torch.bfloat16),
                    (256, 16, 128, torch.bfloat16), (512, 32, 128, torch.bfloat16),
                    (2048, 8, 64, torch.bfloat16), (128, 8, 64, torch.float32),
                    (4096, 8, 64, torch.bfloat16)):
    q = torch.randn(1, S, H, D, dtype=dt)
    # run one real call so the scratch buffers exist and the plan is built
    K._dense(q, q, q, D ** -0.5, True)
    # the most recently built plan is this shape's (the key carries private
    # knobs -- qt, the head-mode limit, the LLC size -- so rebuild it by hand at
    # your peril)
    plans = K._plans()
    plan = plans[max(plans, key=lambda k: 0)] if False else list(plans.values())[-1]
    (seq_q, num_heads, batch, num_kv_heads, seq_k, head_dim, dtc, mode, ktb, ktd,
     ktt, kt_blk, qt, workers, kt_bh, kt_d, vt_bh, vt_d, kt_ptr, vt_ptr,
     v_ptr_packed, ks_ptr, ks_bh, k0, k1, k2, v0, v1, v2, qb, qtok, qh, vb,
     vtok, vh, ob, otok, oh, pack_workers, _kt, _vt, _ks) = plan
    kp, vp, qp = q.data_ptr(), q.data_ptr(), q.data_ptr()
    tp = K._transpose_attention()
    args = (kp, kt_ptr, vp if not vt_ptr else qp, vt_ptr, batch, num_kv_heads,
            seq_k, head_dim, k0, k1, k2, kt_bh, kt_d, kt_blk, v0, v1, v2,
            vt_bh, vt_d, dtc, mode, pack_workers, ks_ptr, ks_bh)
    f = lambda: tp(*args)
    for _ in range(5):
        f()
    s = []
    for _ in range(21):
        t0 = time.perf_counter(); f(); s.append((time.perf_counter() - t0) * 1e6)
    t = statistics.median(s)
    # traffic: K read + K^T write + V read + V^T write, 2 or 4 bytes per element
    es = 2 if dt == torch.bfloat16 else 4
    moved = 4 * S * H * D * es
    print(f"S={S:5d} H={H:2d} D={D:3d} {'bf16' if es == 2 else 'fp32'}: "
          f"pack {t:7.2f} us  {moved / t / 1e3:6.1f} GB/s  ({moved/1e6:.2f} MB moved)")
