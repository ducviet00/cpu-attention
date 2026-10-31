"""Optional checks for the OpenVINO benchmark, not a runtime dependency."""

import sys
import platform
from pathlib import Path

import pytest
import torch

pytest.importorskip("openvino")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import ov_attention as ovbench
from reference import attention
import report


def test_report_labels_openvino_without_changing_old_reports():
    run = dict(environment=dict(machine="test", cpu="CPU", threads=2, git="abc",
                                torch="test", mojo="test", vllm={}), rows=[])
    assert "vLLM" in report.summary([run])
    run["environment"]["openvino"] = "2026.4.1"
    assert "OpenVINO" in report.summary([run])
    assert "vLLM" not in report.summary([run])
    assert "OpenVINO 2026.4.1" in report.full([run])


def test_validation_rejects_wrong_output():
    import bench_openvino
    q = torch.zeros(1, 1, 1, 32)
    kv = [(torch.zeros(17, 1, 32), torch.ones(17, 1, 32))]
    with pytest.raises(AssertionError):
        bench_openvino.validate({"wrong": torch.zeros_like(q)}, q, kv)


def test_report_discloses_unsupported_and_strict_failures():
    run = dict(environment=dict(machine="test", cpu="CPU", threads=2, git="abc", torch="test",
                                mojo="test", vllm={}, openvino="2026.4.1"), rows=[dict(
        kind="paged", shape=[1, 256, 16, 64, True], dtype="bf16", timing={}, ratio=None,
        validation={"ours": dict(strict_pass=False, strict_error=0.002, strict_limit=0.001)},
        unsupported={"openvino p32": "bf16 cache unsupported"})])
    text = report.full([run])
    assert "0/1" in text
    assert "bf16 cache unsupported" in text
    assert "strict limit" in text


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
def test_openvino_flash(dtype, causal, dynamic):
    torch.manual_seed(42)
    q, k, v = (torch.randn(2, 33, 2, 32, dtype=dtype) for _ in range(3))
    runner = ovbench.flash(q, k, v, causal, threads=2, dynamic=dynamic)
    runner.run()
    ref = attention(q, k, v, 32 ** -0.5, causal)
    torch.testing.assert_close(runner.output.double(), ref, rtol=0,
                               atol=0.03 if dtype == torch.bfloat16 else 2e-4)
    q.mul_(0.5)
    runner.run()
    torch.testing.assert_close(runner.output.double(), attention(q, k, v, 32 ** -0.5, causal),
                               rtol=0, atol=0.03 if dtype == torch.bfloat16 else 2e-4)
    assert any("ScaledDotProductAttention" in x["originalLayersNames"] for x in runner.runtime)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("append", [False, True])
def test_openvino_paged_cache_and_uneven_lengths(dtype, append):
    torch.manual_seed(42)
    lens, h, d = [17, 65], 2, 32
    kv = [(torch.randn(n - int(append), h, d, dtype=dtype),
           torch.randn(n - int(append), h, d, dtype=dtype)) for n in lens]
    q, kn, vn = (torch.randn(2, 1, h, d, dtype=dtype) for _ in range(3))
    if platform.machine() == "aarch64" and dtype == torch.bfloat16:
        with pytest.raises(RuntimeError, match="bf16|only supported"):
            ovbench.paged(q, kv, kn, vn, lens, append, threads=2)
        return
    runner = ovbench.paged(q, kv, kn, vn, lens, append, threads=2)
    before = [x.clone() for x in runner.inputs[3:5]]
    for step in range(2):
        vn.add_(1)
        runner.run()
        for i, (k, v) in enumerate(kv):
            if append:
                k, v = torch.cat([k, kn[i]]), torch.cat([v, vn[i]])
            ref = attention(q[i:i + 1], k[None], v[None], d ** -0.5)
            torch.testing.assert_close(runner.output[i:i + 1].double(), ref, rtol=0,
                                       atol=0.03 if dtype == torch.bfloat16 else 2e-4)
        if not append:
            for a, b in zip(before, runner.inputs[3:5]):
                assert torch.equal(a, b)
    if append:
        assert not torch.equal(before[1], runner.inputs[4])
    assert any(x["layerType"] == "PagedAttention" for x in runner.runtime)
