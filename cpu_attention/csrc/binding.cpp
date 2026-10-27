// Python binding: the CPython C API, no pybind.
//
// Most entries take `data_ptr()` integers. The paged decode entry
// (`paged_call`) takes the tensors themselves, because at batch 1 reading them
// from Python was 40% of the call; that one entry ties the module to the torch
// it was built against (the build links that torch's libraries).
//
// Two entries exist for the dense path:
//   dense_make_plan + dense_forward   five arguments per call, the geometry
//                                     riding in one int64 buffer (the call path
//                                     is a tenth of a 26 us short-row call);
//   dense_attention                   the long form, for tests.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <ATen/Parallel.h>
#include <ATen/ops/empty.h>
#include <torch/csrc/autograd/python_variable.h>

#include <cstring>
#include <vector>

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

// The engine's per-layer paged entry, reading the tensors themselves.
//
// At batch 1 a decode call is ~4.6 us and 1.8 us of it was Python: a plan key
// built from three `.shape` reads and hashed, six `data_ptr()`s, a stride and
// `torch.empty_like`. Here those are field loads, the geometry cache is a
// per-thread linear scan, and the output comes from `at::empty`.
//
// The worker count stays a Python policy (`_kernel._workers`): a geometry this
// thread has not seen returns NotImplemented, Python registers it with
// `paged_register`, and calls again.
struct PagedGeom {
  int64_t nseq, nheads, nkv, hd, page, npages;
  int dtype, threads;
  bool store;
  int workers;
  bool same(const PagedGeom& o) const {
    return nseq == o.nseq && nheads == o.nheads && nkv == o.nkv &&
           hd == o.hd && page == o.page && npages == o.npages &&
           dtype == o.dtype && threads == o.threads && store == o.store;
  }
};
thread_local std::vector<PagedGeom> t_paged;

int dtype_code(const at::Tensor& t) {
  switch (t.scalar_type()) {
    case at::kFloat: return attn::kF32;
    case at::kBFloat16: return attn::kBF16;
    default: return -1;
  }
}

const at::Tensor* tensor_arg(PyObject* o, const char* name) {
  if (!THPVariable_Check(o)) {
    PyErr_Format(PyExc_TypeError, "paged_call: %s must be a tensor", name);
    return nullptr;
  }
  return &THPVariable_Unpack(o);
}

// Geometry of (q, k_cache, block_table) plus the calling thread's torch thread
// count, so `torch.set_num_threads` re-runs the worker policy.
bool paged_geom(PyObject* q, PyObject* kc, PyObject* bt, bool store,
                PagedGeom* g) {
  const at::Tensor* tq = tensor_arg(q, "query");
  const at::Tensor* tk = tq ? tensor_arg(kc, "k_cache") : nullptr;
  const at::Tensor* tb = tk ? tensor_arg(bt, "block_table") : nullptr;
  if (!tb) return false;
  if (tq->dim() != 3 || tk->dim() != 4 || tb->dim() != 2) {
    PyErr_SetString(PyExc_ValueError,
                    "paged_call: query [seqs, heads, dim], cache [pages, kv "
                    "heads, page, dim], block_table [seqs, pages]");
    return false;
  }
  g->nseq = tq->size(0);
  g->nheads = tq->size(1);
  g->hd = tq->size(2);
  g->nkv = tk->size(1);
  g->page = tk->size(2);
  g->npages = tb->size(1);
  g->dtype = dtype_code(*tq);
  g->threads = at::get_num_threads();
  g->store = store;
  return true;
}

PyObject* py_paged_register(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs != 5) {
    PyErr_Format(PyExc_TypeError, "paged_register expected 5 arguments, got %zd",
                 nargs);
    return nullptr;
  }
  PagedGeom g;
  if (!paged_geom(args[0], args[1], args[2], PyObject_IsTrue(args[3]) == 1, &g))
    return nullptr;
  g.workers = static_cast<int>(PyLong_AsLong(args[4]));
  if (PyErr_Occurred()) return nullptr;
  if (t_paged.size() >= 64) t_paged.clear();
  t_paged.push_back(g);
  Py_RETURN_NONE;
}

// paged_call(q, k, v, k_cache, v_cache, slots, block_table, lens, scale, out)
// k/v/slots are None for cross attention; out is None to allocate. Layouts are
// checked, not assumed: heads and dims contiguous, any token stride (the
// engine's q/k/v are chunks of one fused projection).
PyObject* py_paged_call(PyObject*, PyObject* const* args, Py_ssize_t nargs) {
  if (nargs != 10) {
    PyErr_Format(PyExc_TypeError, "paged_call expected 10 arguments, got %zd",
                 nargs);
    return nullptr;
  }
  const bool store = args[1] != Py_None;
  PagedGeom g;
  if (!paged_geom(args[0], args[3], args[6], store, &g)) return nullptr;
  const PagedGeom* plan = nullptr;
  for (const PagedGeom& e : t_paged)
    if (e.same(g)) {
      plan = &e;
      break;
    }
  if (!plan) Py_RETURN_NOTIMPLEMENTED;

  const at::Tensor& q = THPVariable_Unpack(args[0]);
  const at::Tensor* kc = tensor_arg(args[3], "k_cache");
  const at::Tensor* vc = kc ? tensor_arg(args[4], "v_cache") : nullptr;
  const at::Tensor* lens = vc ? tensor_arg(args[7], "context_lens") : nullptr;
  if (!lens) return nullptr;
  const at::Tensor& bt = THPVariable_Unpack(args[6]);
  const char* bad = nullptr;
  if (g.dtype < 0 || dtype_code(*kc) != g.dtype || dtype_code(*vc) != g.dtype)
    bad = "query and caches must share one dtype, float32 or bfloat16";
  else if (q.stride(2) != 1 || q.stride(1) != g.hd)
    bad = "query heads/dims must be contiguous";
  else if (!kc->is_contiguous() || !vc->is_contiguous() ||
           !vc->sizes().equals(kc->sizes()))
    bad = "k_cache and v_cache must be contiguous and the same shape";
  else if (bt.scalar_type() != at::kInt || bt.stride(1) != 1 ||
           lens->scalar_type() != at::kInt || !lens->is_contiguous() ||
           lens->numel() < g.nseq)
    bad = "block_table and context_lens must be int32";
  if (bad) {
    PyErr_Format(PyExc_ValueError, "paged_call: %s", bad);
    return nullptr;
  }

  PagedJob job;
  if (store) {
    const at::Tensor* k = tensor_arg(args[1], "key");
    const at::Tensor* v = k ? tensor_arg(args[2], "value") : nullptr;
    const at::Tensor* slots = v ? tensor_arg(args[5], "slot_mapping") : nullptr;
    if (!slots) return nullptr;
    // One token stride serves both: they are chunks of the same projection.
    if (dtype_code(*k) != g.dtype || dtype_code(*v) != g.dtype ||
        k->dim() != 3 || !k->sizes().equals(v->sizes()) ||
        k->size(0) != g.nseq || k->size(1) != g.nkv || k->size(2) != g.hd ||
        k->stride(2) != 1 || k->stride(1) != g.hd ||
        v->stride(2) != 1 || v->stride(1) != g.hd ||
        v->stride(0) != k->stride(0) || slots->scalar_type() != at::kLong ||
        !slots->is_contiguous() || slots->numel() < g.nseq) {
      PyErr_SetString(PyExc_ValueError,
                      "paged_call: key/value must be [seqs, kv heads, dim] "
                      "with one token stride, slot_mapping int64");
      return nullptr;
    }
    job.k = k->data_ptr();
    job.v = v->data_ptr();
    job.slots = slots->data_ptr<int64_t>();
    job.kv_stride = k->stride(0);
  }

  PyObject* out_obj = args[9];
  at::Tensor fresh;
  const at::Tensor* out;
  if (out_obj == Py_None) {
    fresh = at::empty({g.nseq, g.nheads, g.hd}, q.options());
    out = &fresh;
  } else {
    out = tensor_arg(out_obj, "out");
    if (!out) return nullptr;
    if (dtype_code(*out) != g.dtype || !out->is_contiguous() ||
        !out->sizes().equals(q.sizes())) {
      PyErr_SetString(PyExc_ValueError,
                      "paged_call: out must be a contiguous tensor shaped and "
                      "typed like query");
      return nullptr;
    }
  }
  const double scale = PyFloat_AsDouble(args[8]);
  if (PyErr_Occurred()) return nullptr;

  job.q = q.data_ptr();
  job.k_cache = kc->data_ptr();
  job.v_cache = vc->data_ptr();
  job.out = out->data_ptr();
  job.block_table = bt.data_ptr<int32_t>();
  job.lens = lens->data_ptr<int32_t>();
  job.num_seqs = g.nseq;
  job.num_heads = g.nheads;
  job.num_kv_heads = g.nkv;
  job.head_dim = g.hd;
  job.page = g.page;
  job.bt_stride = bt.stride(0);
  job.q_stride = q.stride(0);
  job.out_stride = g.nheads * g.hd;
  job.scale = static_cast<float>(scale);
  job.dtype = g.dtype;
  attn::attention_entry(job, plan->workers, store, true);
  if (out_obj != Py_None) {
    Py_INCREF(out_obj);
    return out_obj;
  }
  return THPVariable_Wrap(std::move(fresh));
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
    {"paged_call", reinterpret_cast<PyCFunction>(
                       reinterpret_cast<void (*)()>(py_paged_call)),
     METH_FASTCALL,
     "paged attention from tensors; NotImplemented for an unregistered geometry"},
    {"paged_register", reinterpret_cast<PyCFunction>(
                           reinterpret_cast<void (*)()>(py_paged_register)),
     METH_FASTCALL,
     "register a paged geometry's worker count for this thread"},
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
