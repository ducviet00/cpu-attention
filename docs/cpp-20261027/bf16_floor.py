"""An exact counterexample to an absolute 1e-3 gate with bf16 output."""
import json
import platform
import torch
import cpu_attention as A
from cpu_attention import _kernel as K

torch.set_num_threads(1)
dt = torch.bfloat16
q = torch.zeros(1, 1, 32, dtype=dt)
k = torch.zeros(2, 1, 32, dtype=dt)
v = torch.ones(2, 1, 32, dtype=dt)
v[1].fill_(1.0078125)
truth = 1.00390625
rows = []
for be in ("cpp", "mojo"):
    K.set_backend(be)
    kc = torch.zeros(1, 1, 16, 32, dtype=dt)
    vc = torch.zeros_like(kc)
    A.reshape_and_cache(k, v, kc, vc, torch.arange(2, dtype=torch.int64))
    out = A.cross_attention(q, kc, vc, torch.zeros(1, 1, dtype=torch.int32),
                            torch.tensor([2], dtype=torch.int32), 32 ** -.5)
    error = (out.float() - truth).abs().max().item()
    rows.append(dict(machine=platform.machine(), backend=be,
                     inputs=[1.0, 1.0078125], exact_output=truth,
                     bf16_output=out.flatten()[0].item(), max_abs_error=error,
                     minimum_bf16_rounding_error=0.00390625,
                     requested_absolute_gate=0.001,
                     gate_possible=False))
    assert error == 0.00390625
print(json.dumps(rows, indent=2))
