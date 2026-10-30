"""The performance manifest: every row the unified Mojo backend is judged on.

Fixed before optimization started (2026-10-28) and not edited to make a score:
a losing row stays in and is reported as a loss.

* ``HISTORICAL``: the 42-row matrix of the README (11 flash_attn_func shapes,
  5 paged batch/context pairs x self/cross, both dtypes).
* ``EXTENDED``: odd lengths around tile/page boundaries, unequal query/key
  lengths, noncausal, GQA/MQA, head_dim 32/128, strided fused-QKV inputs,
  page sizes, heterogeneous contexts, store-only, the preallocated ``out=``
  engine call and the public ``flash_attn_with_kvcache`` call.

Row fields:
  flash: b, sq, sk, h, hkv, d, causal, layout ("contig" | "fused")
  paged: entry ("engine" | "engine_out" | "api" | "store"), op ("self" |
         "cross"), b, ctx (int, or a list of per-sequence contexts), h, hkv, d,
         page (None = the box default: 32 on x86, 16 on aarch64)
Every row is run in both dtypes.
"""

DTYPES = ("bf16", "fp32")


def flash(sq, h, d, sk=None, hkv=None, causal=True, layout="contig", b=1):
    return dict(kind="flash", b=b, sq=sq, sk=sk or sq, h=h, hkv=hkv or h, d=d,
                causal=causal, layout=layout)


def paged(op, b, ctx, h=16, hkv=None, d=64, page=None, entry="engine"):
    return dict(kind="paged", entry=entry, op=op, b=b, ctx=ctx, h=h,
                hkv=hkv or h, d=d, page=page)


_HIST_FLASH = [(128, 8, 64), (512, 8, 64), (1024, 8, 64), (2048, 8, 64),
               (4096, 8, 64), (256, 16, 128), (1024, 16, 128), (2048, 16, 128),
               (4096, 16, 128), (512, 32, 128), (1024, 8, 32)]
_HIST_PAGED = [(1, 256), (8, 256), (8, 1024), (32, 1024), (1, 4096)]

HISTORICAL = ([flash(s, h, d) for s, h, d in _HIST_FLASH] +
              [paged(op, b, c) for b, c in _HIST_PAGED
               for op in ("self", "cross")])

EXTENDED = (
    # Odd lengths either side of the tile/page boundaries.
    [flash(s, 8, 64) for s in (17, 65, 129, 257, 513, 1025, 2049, 4097)] +
    [flash(s, 16, 128) for s in (129, 257, 1025, 2049)] +
    [flash(1025, 8, 32)] +
    # Unequal query/key lengths (bottom-right causal), including one query.
    [flash(sq, 8, 64, sk=sk) for sq, sk in
     ((1, 129), (17, 301), (127, 129), (129, 127), (257, 513), (513, 257),
      (1025, 2049), (2049, 1025))] +
    # Noncausal.
    [flash(1024, 8, 64, causal=False), flash(1025, 8, 64, causal=False),
     flash(1024, 16, 128, causal=False)] +
    # GQA / MQA.
    [flash(1024, 32, 128, hkv=8), flash(1024, 16, 64, hkv=1)] +
    # q/k/v as strided chunks of one fused projection (the engine's layout).
    [flash(1024, 8, 64, layout="fused"), flash(257, 16, 128, layout="fused")] +
    # Paged: odd contexts around page boundaries.
    [paged(op, b, c) for b, c in ((8, 255), (8, 257), (8, 1023), (8, 1025),
                                  (1, 4097)) for op in ("self", "cross")] +
    # Page sizes (the box default is in the historical rows).
    [paged("self", 8, 1024, page=p) for p in (16, 48, 64)] +
    # GQA 4:1 and MQA decode.
    [paged("self", 8, 1024, hkv=4), paged("cross", 8, 1024, hkv=4),
     paged("cross", 8, 1024, hkv=1)] +
    # Heterogeneous contexts in one batch.
    [paged(op, 8, [17, 255, 1023, 1025, 33, 513, 2047, 64])
     for op in ("self", "cross")] +
    # head_dim 32 / 128 decode.
    [paged("cross", 8, 1024, d=32), paged("cross", 8, 1024, h=8, d=128)] +
    # Store only (reshape_and_cache): one decode step, and a 1024-token prompt.
    [paged("store", 32, 1, entry="store"), paged("store", 1, 1024, entry="store")] +
    # The engine call with a preallocated out=, and the public API.
    [paged(op, b, c, entry="engine_out") for b, c in _HIST_PAGED
     for op in ("self", "cross")] +
    [paged(op, b, c, entry="api") for b, c in ((1, 256), (8, 1024))
     for op in ("self", "cross")]
)

FULL = HISTORICAL + EXTENDED


def rows(name="full"):
    base = {"historical": HISTORICAL, "extended": EXTENDED, "full": FULL}[name]
    return [dict(r, dtype=dt) for r in base for dt in DTYPES]


def label(r):
    if r["kind"] == "flash":
        s = f"flash {r['sq']}" + (f"x{r['sk']}" if r["sk"] != r["sq"] else "")
        s += f":{r['h']}" + (f"/{r['hkv']}" if r["hkv"] != r["h"] else "")
        s += f":{r['d']}"
        s += "" if r["causal"] else " noncausal"
        s += "" if r["layout"] == "contig" else f" {r['layout']}"
    else:
        ctx = r["ctx"] if isinstance(r["ctx"], int) else "het"
        s = f"paged {r['entry']} {r['op']} b={r['b']} ctx={ctx}"
        if r["hkv"] != r["h"]:
            s += f" h={r['h']}/{r['hkv']}"
        if r["d"] != 64:
            s += f" d={r['d']}"
        if r["page"]:
            s += f" page={r['page']}"
    return f"{s} {r['dtype']}"


if __name__ == "__main__":
    for name in ("historical", "extended", "full"):
        print(name, len(rows(name)))
    for r in rows():
        print(label(r))
