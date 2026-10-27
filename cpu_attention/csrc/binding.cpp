// Python binding: the plain CPython C API, no torch and no pybind.
//
// Tensors never cross the boundary -- the Python side owns them and passes
// `data_ptr()` integers, so this file has no dependency on libtorch's ABI and
// the module loads in any process.
//
// Two entries exist for the dense path:
//   dense_make_plan + dense_forward   five arguments per call, the geometry
//                                     riding in one int64 buffer (the call path
//                                     is a tenth of a 26 us short-row call);
//   dense_attention                   the long form, for tests.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstring>

#include "cpu_attn.hpp"
#include "cpu_isa.hpp"

namespace {

using attn::DenseJob;
using attn::PagedJob;

constexpr int64_t kPlanMagic = 0x4350504154544e31ll;
enum {
  P_MAGIC = 0,
  P_BATCH,
  P_SEQ_Q,
  P_SEQ_K,
  P_HEADS,
  P_KV_HEADS,
  P_HD,
  P_DTYPE,
  P_CAUSAL,
  P_SCALE,
  P_QB,
  P_QT,
  P_QH,
  P_KB,
  P_KT,
  P_KH,
  P_VB,
  P_VT,
  P_VH,
  P_OB,
  P_OT,
  P_OH,
  P_WORKERS,
  P_N
};

inline int64_t f2i(float f) {
  int32_t i;
  std::memcpy(&i, &f, 4);
  return i;
}
inline float i2f(int64_t i) {
  const int32_t v = static_cast<int32_t>(i);
  float f;
  std::memcpy(&f, &v, 4);
  return f;
}

PyObject* py_attention(PyObject*, PyObject* args) {
  unsigned long long q, k, v, kc, vc, dst, slots, bt, lens;
  long long nseq, nheads, nkv, hd, page, btstride, qstride, kvstride, dststride;
  long long dtype, workers, do_store, do_attend;
  double scale;
  if (!PyArg_ParseTuple(args, "KKKKKKKKKLLLLLLLLLLdLLL", &q, &k, &v, &kc, &vc,
                        &dst, &slots, &bt, &lens, &nseq, &nheads, &nkv, &hd,
                        &page, &btstride, &qstride, &kvstride, &dststride,
                        &dtype, &scale, &workers, &do_store, &do_attend))
    return nullptr;
  PagedJob job;
  job.q = reinterpret_cast<const void*>(q);
  job.k = reinterpret_cast<const void*>(k);
  job.v = reinterpret_cast<const void*>(v);
  job.k_cache = reinterpret_cast<void*>(kc);
  job.v_cache = reinterpret_cast<void*>(vc);
  job.out = reinterpret_cast<void*>(dst);
  job.slots = reinterpret_cast<const int64_t*>(slots);
  job.block_table = reinterpret_cast<const int32_t*>(bt);
  job.lens = reinterpret_cast<const int32_t*>(lens);
  job.num_seqs = nseq;
  job.num_heads = nheads;
  job.num_kv_heads = nkv;
  job.head_dim = hd;
  job.page = page;
  job.bt_stride = btstride;
  job.q_stride = qstride;
  job.kv_stride = kvstride;
  job.out_stride = dststride;
  job.scale = static_cast<float>(scale);
  job.dtype = static_cast<int>(dtype);
  attn::attention_entry(job, static_cast<int>(workers), do_store != 0,
                        do_attend != 0);
  Py_RETURN_NONE;
}

// Paged plan: the 14 constants of a geometry, so a call marshals only the data
// pointers and the two token strides. At batch 1 the whole call is 7 us, and
// the 23-argument long form costs 0.68 us of it in `PyArg_ParseTuple` alone.
enum {
  Q_MAGIC = 0,
  Q_NSEQ,
  Q_NHEADS,
  Q_NKV,
  Q_HD,
  Q_PAGE,
  Q_BTSTRIDE,
  Q_OUTSTRIDE,
  Q_DTYPE,
  Q_SCALE,
  Q_WORKERS,
  Q_DOSTORE,
  Q_N
};

PyObject* py_paged_make_plan(PyObject*, PyObject* args) {
  unsigned long long addr;
  long long nseq, nheads, nkv, hd, page, btstride, outstride, dtype, workers,
      do_store;
  double scale;
  if (!PyArg_ParseTuple(args, "KLLLLLLLLdLL", &addr, &nseq, &nheads, &nkv, &hd,
                        &page, &btstride, &outstride, &dtype, &scale, &workers,
                        &do_store))
    return nullptr;
  int64_t* p = reinterpret_cast<int64_t*>(addr);
  p[Q_MAGIC] = kPlanMagic;
  p[Q_NSEQ] = nseq;
  p[Q_NHEADS] = nheads;
  p[Q_NKV] = nkv;
  p[Q_HD] = hd;
  p[Q_PAGE] = page;
  p[Q_BTSTRIDE] = btstride;
  p[Q_OUTSTRIDE] = outstride;
  p[Q_DTYPE] = dtype;
  p[Q_SCALE] = f2i(static_cast<float>(scale));
  p[Q_WORKERS] = workers;
  p[Q_DOSTORE] = do_store;
  Py_RETURN_NONE;
}

PyObject* py_paged_forward(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs != 13) {
    PyErr_Format(PyExc_TypeError, "paged_forward expected 13 arguments, got %zd",
                 nargs);
    return nullptr;
  }
  // FASTCALL avoids building a tuple for every decoder layer. Keep conversions
  // checked: a bad address argument must fail before the plan is dereferenced.
  unsigned long long ptrs[10];
  for (int i = 0; i < 10; ++i) ptrs[i] = PyLong_AsUnsignedLongLong(args[i]);
  const auto [addr, q, k, v, kc, vc, out, slots, bt, lens] = ptrs;
  const long long qstride = PyLong_AsLongLong(args[10]);
  const long long kvstride = PyLong_AsLongLong(args[11]);
  const double scale = PyFloat_AsDouble(args[12]);
  if (PyErr_Occurred()) return nullptr;
  const int64_t* p = reinterpret_cast<const int64_t*>(addr);
  if (p[Q_MAGIC] != kPlanMagic) {
    PyErr_SetString(PyExc_RuntimeError, "paged_forward: stale plan buffer");
    return nullptr;
  }
  PagedJob job;
  job.q = reinterpret_cast<const void*>(q);
  job.k = reinterpret_cast<const void*>(k);
  job.v = reinterpret_cast<const void*>(v);
  job.k_cache = reinterpret_cast<void*>(kc);
  job.v_cache = reinterpret_cast<void*>(vc);
  job.out = reinterpret_cast<void*>(out);
  job.slots = reinterpret_cast<const int64_t*>(slots);
  job.block_table = reinterpret_cast<const int32_t*>(bt);
  job.lens = reinterpret_cast<const int32_t*>(lens);
  job.num_seqs = p[Q_NSEQ];
  job.num_heads = p[Q_NHEADS];
  job.num_kv_heads = p[Q_NKV];
  job.head_dim = p[Q_HD];
  job.page = p[Q_PAGE];
  job.bt_stride = p[Q_BTSTRIDE];
  job.out_stride = p[Q_OUTSTRIDE];
  job.q_stride = qstride;
  job.kv_stride = kvstride;
  job.scale = static_cast<float>(scale);
  job.dtype = static_cast<int>(p[Q_DTYPE]);
  attn::attention_entry(job, static_cast<int>(p[Q_WORKERS]),
                        p[Q_DOSTORE] != 0, true);
  Py_RETURN_NONE;
}

PyObject* py_dense_make_plan(PyObject*, PyObject* args) {
  unsigned long long addr;
  long long batch, seq_q, seq_k, heads, kv_heads, hd, dtype, causal;
  long long qb, qt, qh, kb, kt, kh, vb, vt, vh, ob, ot, oh, workers;
  double scale;
  if (!PyArg_ParseTuple(args, "KLLLLLLLLdLLLLLLLLLLLLL", &addr, &batch, &seq_q,
                        &seq_k, &heads, &kv_heads, &hd, &dtype, &causal, &scale,
                        &qb, &qt, &qh, &kb, &kt, &kh, &vb, &vt, &vh, &ob, &ot,
                        &oh, &workers))
    return nullptr;
  int64_t* p = reinterpret_cast<int64_t*>(addr);
  p[P_MAGIC] = kPlanMagic;
  p[P_BATCH] = batch;
  p[P_SEQ_Q] = seq_q;
  p[P_SEQ_K] = seq_k;
  p[P_HEADS] = heads;
  p[P_KV_HEADS] = kv_heads;
  p[P_HD] = hd;
  p[P_DTYPE] = dtype;
  p[P_CAUSAL] = causal;
  p[P_SCALE] = f2i(static_cast<float>(scale));
  p[P_QB] = qb;
  p[P_QT] = qt;
  p[P_QH] = qh;
  p[P_KB] = kb;
  p[P_KT] = kt;
  p[P_KH] = kh;
  p[P_VB] = vb;
  p[P_VT] = vt;
  p[P_VH] = vh;
  p[P_OB] = ob;
  p[P_OT] = ot;
  p[P_OH] = oh;
  p[P_WORKERS] = workers;
  Py_RETURN_NONE;
}

PyObject* py_dense_forward(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs != 5) {
    PyErr_Format(PyExc_TypeError, "dense_forward expected 5 arguments, got %zd",
                 nargs);
    return nullptr;
  }
  unsigned long long ptrs[5];
  for (int i = 0; i < 5; ++i) ptrs[i] = PyLong_AsUnsignedLongLong(args[i]);
  const auto [addr, q, k, v, out] = ptrs;
  if (PyErr_Occurred()) return nullptr;
  const int64_t* p = reinterpret_cast<const int64_t*>(addr);
  if (p[P_MAGIC] != kPlanMagic) {
    PyErr_SetString(PyExc_RuntimeError, "dense_forward: stale plan buffer");
    return nullptr;
  }
  DenseJob job;
  job.batch = p[P_BATCH];
  job.seq_q = p[P_SEQ_Q];
  job.seq_k = p[P_SEQ_K];
  job.heads = p[P_HEADS];
  job.kv_heads = p[P_KV_HEADS];
  job.head_dim = p[P_HD];
  job.dtype = static_cast<int>(p[P_DTYPE]);
  job.causal = p[P_CAUSAL] != 0;
  job.scale = i2f(p[P_SCALE]);
  job.qb = p[P_QB];
  job.qt = p[P_QT];
  job.qh = p[P_QH];
  job.kb = p[P_KB];
  job.kt = p[P_KT];
  job.kh = p[P_KH];
  job.vb = p[P_VB];
  job.vt = p[P_VT];
  job.vh = p[P_VH];
  job.ob = p[P_OB];
  job.ot = p[P_OT];
  job.oh = p[P_OH];
  job.q = reinterpret_cast<const void*>(q);
  job.k = reinterpret_cast<const void*>(k);
  job.v = reinterpret_cast<const void*>(v);
  job.out = reinterpret_cast<void*>(out);
  attn::dense_entry(job, static_cast<int>(p[P_WORKERS]));
  Py_RETURN_NONE;
}

PyObject* py_dense_attention(PyObject*, PyObject* args) {
  unsigned long long q, k, v, out;
  long long batch, seq_q, seq_k, heads, kv_heads, hd;
  long long qb, qt, qh, kb, kt, kh, vb, vt, vh, ob, ot, oh;
  long long dtype, causal, workers;
  double scale;
  if (!PyArg_ParseTuple(args, "KKKKLLLLLLLLLLLLLLLLLLdLLL", &q, &k, &v, &out,
                        &batch, &seq_q, &seq_k, &heads, &kv_heads, &hd, &qb,
                        &qt, &qh, &kb, &kt, &kh, &vb, &vt, &vh, &ob, &ot, &oh,
                        &scale, &causal, &dtype, &workers))
    return nullptr;
  DenseJob job;
  job.q = reinterpret_cast<const void*>(q);
  job.k = reinterpret_cast<const void*>(k);
  job.v = reinterpret_cast<const void*>(v);
  job.out = reinterpret_cast<void*>(out);
  job.batch = batch;
  job.seq_q = seq_q;
  job.seq_k = seq_k;
  job.heads = heads;
  job.kv_heads = kv_heads;
  job.head_dim = hd;
  job.qb = qb;
  job.qt = qt;
  job.qh = qh;
  job.kb = kb;
  job.kt = kt;
  job.kh = kh;
  job.vb = vb;
  job.vt = vt;
  job.vh = vh;
  job.ob = ob;
  job.ot = ot;
  job.oh = oh;
  job.scale = static_cast<float>(scale);
  job.dtype = static_cast<int>(dtype);
  job.causal = causal != 0;
  attn::dense_entry(job, static_cast<int>(workers));
  Py_RETURN_NONE;
}

PyObject* py_backend(PyObject*, PyObject*) {
  return PyUnicode_FromString("cpp");
}

PyObject* py_isa(PyObject*, PyObject*) {
  return PyUnicode_FromString(attn::isa_name(attn::dense_isa_name()));
}

PyObject* py_workspace_bytes(PyObject*, PyObject*) {
  return PyLong_FromSize_t(attn::workspace_bytes());
}

PyObject* py_workspace_clear(PyObject*, PyObject*) {
  attn::workspace_clear();
  Py_RETURN_NONE;
}

PyMethodDef kMethods[] = {
    {"attention", py_attention, METH_VARARGS,
     "paged attention (store and/or attend), raw pointers"},
    {"dense_attention", py_dense_attention, METH_VARARGS,
     "dense attention, long form"},
    {"dense_make_plan", py_dense_make_plan, METH_VARARGS,
     "fill a per-geometry plan buffer"},
    {"paged_make_plan", py_paged_make_plan, METH_VARARGS,
     "fill a per-geometry paged plan buffer"},
    {"paged_forward", reinterpret_cast<PyCFunction>(
                          reinterpret_cast<void (*)()>(py_paged_forward)),
     METH_FASTCALL,
     "paged attention from a plan buffer"},
    {"dense_forward", reinterpret_cast<PyCFunction>(
                          reinterpret_cast<void (*)()>(py_dense_forward)),
     METH_FASTCALL,
     "dense attention from a plan buffer"},
    {"backend", py_backend, METH_NOARGS, "backend name"},
    {"isa", py_isa, METH_NOARGS, "the ISA the dense path selects"},
    {"workspace_bytes", py_workspace_bytes, METH_NOARGS, "held workspace"},
    {"workspace_clear", py_workspace_clear, METH_NOARGS, "drop workspace"},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {PyModuleDef_HEAD_INIT, "_C", "cpu_attention (C++)", -1,
                       kMethods, nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__C(void) { return PyModule_Create(&kModule); }
