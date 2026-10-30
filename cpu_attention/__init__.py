"""CPU attention kernels written in Mojo, with flash-attn's API.

The paged decode path mirrors ``flash_attn_with_kvcache``; the cache layout is
head-major ``[num_blocks, nheads_k, page_block_size, headdim]`` rather than
flash-attn's page-major one, because K is stored transposed inside the page, so
``reshape_and_cache`` (vLLM's signature) is exported to write it.
"""

from .cache import reshape_and_cache
from .flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_with_kvcache",
    "reshape_and_cache",
]
