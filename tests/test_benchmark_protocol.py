"""Checks for benchmark validation and independent build controls."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
import bench


class NoopPackage:
    def reshape_and_cache(self, k, v, kc, vc, slots):
        pass

    def cross_attention(self, q, kc, vc, bt, lens, scale, out=None):
        return vc[:, 0, 0].reshape_as(q)


class OneTokenPackage(NoopPackage):
    def reshape_and_cache(self, k, v, kc, vc, slots):
        kc[:, 0, 0] = k.reshape_as(kc[:, 0, 0])
        vc[:, 0, 0] = v.reshape_as(vc[:, 0, 0])


class InvalidAttentionPackage(NoopPackage):
    def cross_attention(self, q, kc, vc, bt, lens, scale, out=None):
        return torch.full_like(q, float("nan"))


class ProtocolTests(unittest.TestCase):
    def test_invalid_fast_cpp_cannot_screen_valid_baseline(self):
        samples = {"invalid": [1, 2, 3], "valid": [10, 11, 12], "slow": [40, 41, 42]}
        valid = {"invalid": (float("nan"), False), "valid": (0, True), "slow": (0, True)}
        self.assertEqual(set(bench.screen_cpp(samples, valid, 2)), {"slow"})

    def test_all_invalid_cpp_are_kept_for_reporting(self):
        self.assertEqual(bench.screen_cpp({"bad": [1]}, {"bad": (1, False)}, 2), {})

    def test_store_noop_is_invalid(self):
        arm = SimpleNamespace(name="cpp_fake", pkg=NoopPackage(), enter=lambda: None, env={})
        row = dict(b=1, h=1, hkv=1, d=32, ctx=1, entry="store", op="store", dtype="fp32")
        args = SimpleNamespace(ref_arm=arm.name, paged_rounds=1, block_calls=1)
        fns, check, _, _ = bench.paged_row(row, [arm], {}, args, 16)
        fns[arm.name]()
        self.assertFalse(check([arm.name])[arm.name][1])

    def test_invalid_bf16_peer_does_not_invalidate_good_cpp(self):
        bad = SimpleNamespace(name="cpp_bad", pkg=InvalidAttentionPackage(), enter=lambda: None)
        good = SimpleNamespace(name="cpp_scalar", pkg=OneTokenPackage(), enter=lambda: None)
        row = dict(b=1, h=1, hkv=1, d=32, ctx=1, entry="engine", op="cross", dtype="bf16")
        args = SimpleNamespace(ref_arm=bad.name, paged_rounds=1, block_calls=1)
        fns, check, _, _ = bench.paged_row(row, [bad, good], {}, args, 16)
        for fn in fns.values():
            fn()
        valid = check(list(fns))
        self.assertFalse(valid[bad.name][1])
        self.assertTrue(valid[good.name][1])
        self.assertEqual(check.peer, good.name)

    def test_previous_has_independent_duplicate(self):
        args = SimpleNamespace(arms="mojo_new,mojo_ctrl", previous="/saved/package", load_seed=0)
        with patch.object(bench.A, "current", side_effect=lambda name, backend: name), \
             patch.object(bench.A, "snapshot", side_effect=lambda name, directory: name) as snapshot:
            arms = bench.build_arms(args)
        self.assertEqual(set(arms), {"mojo_new", "mojo_ctrl", "mojo_previous"})
        self.assertEqual(snapshot.call_count, 2)
        self.assertNotEqual(snapshot.call_args_list[0].args[0],
                            snapshot.call_args_list[1].args[0])

    def test_stderr_version_is_recorded(self):
        self.assertEqual(bench._sh(sys.executable, "-c", "import sys; sys.stderr.write('version')"),
                         "version")


if __name__ == "__main__":
    unittest.main()
