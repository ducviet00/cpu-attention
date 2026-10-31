"""OpenVINO 2026.4.1 CPU attention adapters; no conversion in the timed call."""

import numpy as np
import openvino as ov
from openvino import opset13 as ops
import torch


def tensor(x):
    raw = x.view(torch.uint16).numpy() if x.dtype == torch.bfloat16 else x.numpy()
    kind = ov.Type.bf16 if x.dtype == torch.bfloat16 else ov.Type(raw.dtype)
    return ov.Tensor(raw, list(x.shape), kind)


class Runner:
    def __init__(self, model, inputs, shape, threads):
        self.inputs = inputs
        dtype = inputs[0].dtype
        kind = ov.Type.bf16 if dtype == torch.bfloat16 else ov.Type.f32
        self.compiled = ov.Core().compile_model(model, "CPU", {
            "INFERENCE_NUM_THREADS": threads, "NUM_STREAMS": 1,
            "PERFORMANCE_HINT": "LATENCY", "INFERENCE_PRECISION_HINT": kind,
            "KV_CACHE_PRECISION": kind, "ENABLE_CPU_PINNING": False,
            "ENABLE_HYPER_THREADING": False,
        })
        self.request = self.compiled.create_infer_request()
        self.buffers = [tensor(x) for x in inputs]
        for i, buf in enumerate(self.buffers):
            self.request.set_input_tensor(i, buf)
        self.output = torch.empty(shape, dtype=dtype)
        self.output_buffer = tensor(self.output.reshape(inputs[0].shape))
        self.request.set_output_tensor(0, self.output_buffer)
        self.runtime = [dict(name=n.get_friendly_name(), **{
            key: str(value.value) for key, value in n.get_rt_info().items()
            if key in ("layerType", "runtimePrecision", "primitiveType", "outputPrecisions",
                       "originalLayersNames")})
            for n in self.compiled.get_runtime_model().get_ordered_ops()]

    def run(self):
        self.request.infer(share_outputs=True)


def parameter(x, name, dynamic_tokens=False):
    shape = list(x.shape)
    if dynamic_tokens:
        shape[0] = -1
    kind = ov.Type.bf16 if x.dtype == torch.bfloat16 else ov.Type(x.numpy().dtype)
    return ops.parameter(shape, kind, name=name)


def flash(q, k, v, causal, threads, dynamic=False):
    params = [parameter(x, n) for x, n in zip((q, k, v), ("q", "k", "v"))]
    if dynamic:
        for p in params:
            shape = p.get_partial_shape()
            shape[1] = -1
            p.set_partial_shape(shape)
    qt, kt, vt = (ops.transpose(x, [0, 2, 1, 3]) for x in params)
    attn = ops.scaled_dot_product_attention(qt, kt, vt, causal=causal)
    result = ops.transpose(attn, [0, 2, 1, 3])
    return Runner(ov.Model([result], params), [q, k, v], q.shape, threads)


def paged_model(inputs, h, d, max_len, append):
    params = [parameter(x, n, i < 3 or i == 6) for i, (x, n) in enumerate(zip(inputs,
        ("q", "k", "v", "key_cache.0", "value_cache.0", "past_lens",
         "subsequence_begins", "block_indices", "block_indices_begins")))]
    const = lambda x, dt: ops.constant(np.array(x, dtype=dt))
    i32, f32 = np.int32, np.float32
    args = params + [
        const(d ** -0.5, f32), const(0, i32), const([], f32), const(max_len, i32),
        const([], i32), const([], i32), const([], i32), const([], f32),
        const([], f32), const(64, i32), const(8, i32), const([], f32),
        const(0, i32), const([], i32), const([], i32), const([], i32),
        const([], i32), const([], np.uint8), const([], i32),
    ]
    attn = ov.op._PagedAttentionExtension([x.output(0) for x in args])
    attn.set_attribute("write_kv_cache", append)
    for key, value in dict(num_k_heads=h, num_v_heads=h, k_head_size=d, v_head_size=d).items():
        attn.set_rt_info(value, key)
    return ov.Model([attn.output(0)], params)


def paged(q, kv, kn, vn, lens, append, threads):
    b, _, h, d = q.shape
    counts = [(n + 31) // 32 for n in lens]
    starts = torch.tensor([0] + list(np.cumsum(counts)), dtype=torch.int32)
    blocks = torch.arange(sum(counts), dtype=torch.int32)
    kc = torch.zeros(sum(counts), h, 32, d, dtype=q.dtype)
    vc = torch.zeros_like(kc)
    staged = [len(k) for k, _ in kv]
    # Stage via the same operator's writer, not an assumed cache packing scheme.
    ks, vs = (torch.cat([pair[i] for pair in kv]).reshape(-1, h * d) for i in (0, 1))
    stage_inputs = [ks, ks, vs, kc, vc, torch.zeros(b, dtype=torch.int32),
                    torch.tensor([0] + list(np.cumsum(staged)), dtype=torch.int32), blocks, starts]
    model = paged_model(stage_inputs, h, d, max(lens), True)
    stage = Runner(model, stage_inputs, ks.shape, threads)
    precision = [x["runtimePrecision"] for x in stage.runtime if x["layerType"] == "PagedAttention"]
    if q.dtype == torch.bfloat16 and precision != ["bf16"]:
        raise RuntimeError(f"bf16 paged attention/cache is not native on this CPU: {precision}; "
                           "f32 fallback does not preserve the shared bf16 cache writer")
    stage.run()
    inputs = [q.reshape(b, h * d), kn.reshape(b, h * d), vn.reshape(b, h * d), kc, vc,
              torch.tensor([n - 1 for n in lens], dtype=torch.int32),
              torch.arange(b + 1, dtype=torch.int32), blocks, starts]
    if not append:
        # A reader sees past_lens + query length tokens, but never stores K/V.
        model = paged_model(inputs, h, d, max(lens), False)
    return Runner(model, inputs, q.shape, threads)
