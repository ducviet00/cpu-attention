import argparse
import importlib.util
import itertools
import json
import os
import random
import sys

ROOT = os.path.expanduser('~/cpu-attention')
sys.path[:0] = [ROOT, ROOT + '/audit']
import torch
import audit_common as C
import cpu_attention as A
from cpu_attention import _kernel as K

p = argparse.ArgumentParser()
p.add_argument('--old', required=True)
p.add_argument('--new', required=True)
p.add_argument('--shapes', default='128:8:64,512:8:64,1024:8:64,256:16:128,1024:16:128,1024:8:32')
p.add_argument('--dtype', default='both')
p.add_argument('--rounds', type=int, default=120)
p.add_argument('--json', required=True)
a = p.parse_args()
torch.set_num_threads(4)
torch.manual_seed(0)

def load(path):
    spec = importlib.util.spec_from_file_location('cpu_attention._C', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

mods = {'old': load(a.old), 'new': load(a.new)}
rows = []
C.burn()
for shape in a.shapes.split(','):
    s, h, d = map(int, shape.split(':'))
    for dt, tag in [(torch.bfloat16, 'bf16'), (torch.float32, 'fp32')]:
        if a.dtype not in (tag, 'both'):
            continue
        q, k, v = [torch.randn(1, s, h, d, dtype=dt) for _ in range(3)]
        outputs = {}

        def arm(name):
            def run():
                K.set_backend('cpp')
                K._BOUND[('cpp', 'dense_forward')] = mods[name].dense_forward
                outputs[name] = A.flash_attn_func(q, k, v, d ** -.5, True)
            return run

        def mojo():
            K.set_backend('mojo')
            outputs['mojo'] = A.flash_attn_func(q, k, v, d ** -.5, True)

        arms = {'old': arm('old'), 'new': arm('new'), 'ctrl': arm('new'),
                'oldctrl': arm('old'), 'mojo': mojo}
        C.time_arms(arms, 1)
        orders = list(itertools.permutations(arms))
        random.Random(0).shuffle(orders)
        samples = {n: [] for n in arms}
        for r in range(a.rounds):
            order = orders[r % len(orders)]
            got = C.time_arms({n: arms[n] for n in order}, 1, warmup=0)
            for n, xs in got.items():
                samples[n].extend(xs)
        st = {n: C.stats(x) for n, x in samples.items()}
        diff = (outputs['new'].float() - outputs['old'].float()).abs().max().item()
        row = dict(shape=shape, dtype=tag, stats=st, samples_us=samples,
                   new_old=st['new']['min'] / st['old']['min'],
                   new_old_median=st['new']['median'] / st['old']['median'],
                   ctrl_new=st['ctrl']['min'] / st['new']['min'],
                   oldctrl_old=st['oldctrl']['min'] / st['old']['min'],
                   pooled_new_old=min(samples['new'] + samples['ctrl']) /
                       min(samples['old'] + samples['oldctrl']), diff=diff)
        rows.append(row)
        with open(a.json, 'w') as f:
            json.dump(rows, f, indent=1)
        print(json.dumps({k: v for k, v in row.items() if k != 'samples_us'}), flush=True)
