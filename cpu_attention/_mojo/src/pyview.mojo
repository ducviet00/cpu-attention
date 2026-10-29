"""Read a torch tensor's data pointer, sizes and strides without calling Python.

`tensor.data_ptr()` and `.shape` cost ~50 ns each from Python, and a decode call
needs a dozen of them: at batch 1 that is a quarter of the call. The C++ backend
reads `at::Tensor` fields directly; this does the same through field offsets
that `cpu_attention._kernel._calibrate_layout` finds at import by matching the
values torch reports (`tensor._cdata` is the TensorImpl address), and verifies
on a set of probe tensors before the fast path is enabled.

Layout of the `hdr` int64 slots (filled by the calibration):
  1 storage ptr offset in TensorImpl   2 data ptr offset in StorageImpl
  3 storage offset (elements)           4 dim count
  5 inline sizes                        6 inline strides
  7 dtype byte offset (uint16)          8, 9 accepted Python types
  10 Python None
"""

from isa import ptr
from std.ffi import external_call
from std.memory import stack_allocation
from std.python._cpython import PyObjectPtr

comptime IMPL = 16  # THPVariable: PyObject_HEAD, then the at::Tensor (TensorImpl*)


@always_inline
def word(addr: Int) -> Int:
    return Int(ptr[Int64](addr).unsafe_load(0))


@always_inline
def hdr_at(hdr: Int, i: Int) -> Int:
    return word(hdr + 8 * i)


@always_inline
def is_tensor(hdr: Int, obj: Int) -> Bool:
    """Exactly torch.Tensor or nn.Parameter (a subclass may carry another impl)."""
    var tp = word(obj + 8)
    return tp == hdr_at(hdr, 8) or tp == hdr_at(hdr, 9)


@always_inline
def impl(obj: Int) -> Int:
    return word(obj + IMPL)


@always_inline
def data_ptr(hdr: Int, im: Int, itemsize: Int) -> Int:
    var storage = word(im + hdr_at(hdr, 1))
    return word(storage + hdr_at(hdr, 2)) + word(im + hdr_at(hdr, 3)) * itemsize


@always_inline
def dims(hdr: Int, im: Int) -> Int:
    return word(im + hdr_at(hdr, 4))


@always_inline
def size(hdr: Int, im: Int, i: Int) -> Int:
    return word(im + hdr_at(hdr, 5) + 8 * i)


@always_inline
def stride(hdr: Int, im: Int, i: Int) -> Int:
    return word(im + hdr_at(hdr, 6) + 8 * i)


@always_inline
def dtype_code(hdr: Int, im: Int) -> Int:
    return Int(ptr[UInt16](im + hdr_at(hdr, 7)).unsafe_load(0))


@always_inline
def alloc_out_3d(d0: Int, d1: Int, d2: Int, job_dtype: Int) -> PyObjectPtr:
    """Allocate a fresh contiguous 3D CPU tensor via libtorch's C/C++ ABI."""
    var buf = stack_allocation[7, Int64]()
    buf.unsafe_store(0, Int64(d0))
    buf.unsafe_store(1, Int64(d1))
    buf.unsafe_store(2, Int64(d2))
    buf.unsafe_store(3, Int64(d1 * d2))
    buf.unsafe_store(4, Int64(d2))
    buf.unsafe_store(5, Int64(1))
    buf.unsafe_store(6, Int64(0))
    var stype = Int32(15 if job_dtype == 1 else 6)
    var base = Int(buf)
    _ = external_call["aoti_torch_empty_strided", Int32](
        Int64(3), base, base + 24, stype, Int32(0), Int32(0), base + 48
    )
    var handle = Int(buf.unsafe_load(6))
    var py_ptr = external_call[
        "_Z16THPVariable_WrapON2at10TensorBaseE", PyObjectPtr
    ](handle)
    _ = external_call["aoti_torch_delete_tensor_object", Int32](handle)
    return py_ptr


@always_inline
def alloc_out_4d(
    d0: Int, d1: Int, d2: Int, d3: Int, job_dtype: Int
) -> PyObjectPtr:
    """Allocate a fresh contiguous 4D CPU tensor via libtorch's C/C++ ABI."""
    var buf = stack_allocation[9, Int64]()
    var s2 = d2 * d3
    var s1 = d1 * s2
    buf.unsafe_store(0, Int64(d0))
    buf.unsafe_store(1, Int64(d1))
    buf.unsafe_store(2, Int64(d2))
    buf.unsafe_store(3, Int64(d3))
    buf.unsafe_store(4, Int64(s1))
    buf.unsafe_store(5, Int64(s2))
    buf.unsafe_store(6, Int64(d3))
    buf.unsafe_store(7, Int64(1))
    buf.unsafe_store(8, Int64(0))
    var stype = Int32(15 if job_dtype == 1 else 6)
    var base = Int(buf)
    _ = external_call["aoti_torch_empty_strided", Int32](
        Int64(4), base, base + 32, stype, Int32(0), Int32(0), base + 64
    )
    var handle = Int(buf.unsafe_load(8))
    var py_ptr = external_call[
        "_Z16THPVariable_WrapON2at10TensorBaseE", PyObjectPtr
    ](handle)
    _ = external_call["aoti_torch_delete_tensor_object", Int32](handle)
    return py_ptr
