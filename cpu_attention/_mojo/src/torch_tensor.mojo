"""Torch tensors read in place, and outputs allocated without Python.

A C++ extension reads `at::Tensor`; Mojo cannot, so the call route reads each
argument's TensorImpl fields at byte offsets that `cpu_attention._kernel`
finds at import by matching what torch reports, and verifies on probe tensors
(this module's `probe` included) before any call runs. Outputs come from
libtorch's C shim (`aoti_torch_empty_strided`) and `THPVariable_Wrap`.

The runtime config is the int64 buffer `_kernel.runtime()` builds; its slots
follow.
"""

from std.ffi import external_call
from std.memory import stack_allocation
from std.python._cpython import PyObjectPtr

from isa import ptr

comptime IMPL = 0  # THPVariable -> TensorImpl* (byte offset in the Python object)
comptime STORAGE = 1  # TensorImpl -> StorageImpl*
comptime DATA = 2  # StorageImpl -> data pointer
comptime OFFSET = 3  # TensorImpl -> storage offset (elements)
comptime DIM = 4  # TensorImpl -> dim count
comptime SIZES = 5  # TensorImpl -> inline sizes
comptime STRIDES = 6  # TensorImpl -> inline strides
comptime DTYPE = 7  # TensorImpl -> dtype index (uint16)
comptime TENSOR_TYPE = 8  # id(torch.Tensor)
comptime PARAM_TYPE = 9  # id(torch.nn.Parameter)
comptime NONE = 10  # id(None)
comptime TRUE = 11
comptime FALSE = 12
comptime F32 = 13  # dtype codes
comptime BF16 = 14
comptime I32 = 15
comptime I64 = 16
comptime CORES = 17  # physical cores
comptime LLC = 18  # last-level cache bytes
comptime AMX = 19  # 0 off, 1 bf16, 2 bf16 and fp32 (as bf16 hi/lo)


@always_inline
def _word(addr: Int) -> Int:
    return Int(ptr[Int64](addr)[0])


@always_inline
def cfg(rt: Int, slot: Int) -> Int:
    return _word(rt + 8 * slot)


@fieldwise_init
struct Tensor(ImplicitlyCopyable, Movable):
    """A plain `torch.Tensor` / `nn.Parameter` argument (`impl` 0: anything else,
    including None and subclasses, which may carry another implementation)."""

    var impl: Int
    var rt: Int

    @staticmethod
    def of(rt: Int, obj: Int) -> Tensor:
        var tp = _word(obj + 8)
        var plain = tp == cfg(rt, TENSOR_TYPE) or tp == cfg(rt, PARAM_TYPE)
        return Tensor(_word(obj + cfg(rt, IMPL)) if plain else 0, rt)

    def ok(self) -> Bool:
        return self.impl != 0

    def dim(self) -> Int:
        return _word(self.impl + cfg(self.rt, DIM))

    def size(self, i: Int) -> Int:
        return _word(self.impl + cfg(self.rt, SIZES) + 8 * i)

    def stride(self, i: Int) -> Int:
        return _word(self.impl + cfg(self.rt, STRIDES) + 8 * i)

    def dtype(self) -> Int:
        return Int(ptr[UInt16](self.impl + cfg(self.rt, DTYPE))[0])

    def base(self) -> Int:
        """The storage's data address."""
        return _word(_word(self.impl + cfg(self.rt, STORAGE)) + cfg(self.rt, DATA))

    def offset(self) -> Int:
        """The storage offset, in elements."""
        return _word(self.impl + cfg(self.rt, OFFSET))

    def data(self, itemsize: Int) -> Int:
        return self.base() + self.offset() * itemsize

    def contiguous(self) -> Bool:
        var expect = 1
        for i in range(self.dim() - 1, -1, -1):
            if self.size(i) != 1 and self.stride(i) != expect:
                return False
            expect *= self.size(i)
        return True


def dims4(d0: Int, d1: Int, d2: Int = 0, d3: Int = 0) -> Array[Int, 4]:
    var a = Array[Int, 4](fill=0)
    a[0] = d0
    a[1] = d1
    a[2] = d2
    a[3] = d3
    return a^


def empty(n: Int, sizes: Array[Int, 4], bf16: Bool) -> PyObjectPtr:
    """A new contiguous CPU tensor of `n` dims, as a Python object."""
    var buf = stack_allocation[9, Int64]()
    var s = 1
    for i in range(n - 1, -1, -1):
        buf[i] = Int64(sizes[i])
        buf[n + i] = Int64(s)
        s *= sizes[i]
    var handle = stack_allocation[1, Int]()
    _ = external_call["aoti_torch_empty_strided", Int32](
        Int64(n), Int(buf), Int(buf) + 8 * n, Int32(15 if bf16 else 6), Int32(0), Int32(0), Int(handle)
    )
    var obj = external_call["_Z16THPVariable_WrapON2at10TensorBaseE", PyObjectPtr](handle[0])
    _ = external_call["aoti_torch_delete_tensor_object", Int32](handle[0])
    return obj
