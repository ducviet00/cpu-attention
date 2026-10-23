// Shared core: the job descriptions, the workspace, and the loop skeletons.
//
// Every ISA provides a *policy* (see the comments on `dense_run`, `paged_run`,
// `store_run`) and gets the scheduling, the online softmax bookkeeping, the
// mask and the workspace for free. The policy owns every layout decision: what
// a packed K/V slab looks like, how a Q@K^T tile is computed and how
// probabilities are folded into V.
//
// The loop shapes here are the ones a wide-vector CPU wants, measured on the
// three benchmark boxes:
//
//  * dense: a work item is (batch, q head, RQ consecutive query rows). It walks
//    the key range in tiles of KT and, inside a tile, the query rows in chunks
//    of RQT. K/V are therefore read once per RQ rows rather than once per 8,
//    which is what the long D=128 rows are bound by (the same traffic argument
//    that makes vLLM's k-tile/Q-tile split pay).
//  * paged: a work item is (sequence, q head) end to end -- one scan of that
//    head's cache, no split, no reduction.
#pragma once

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "utils.hpp"

namespace attn {

// ---------------------------------------------------------------------------
// jobs
// ---------------------------------------------------------------------------
struct PagedJob {
  const void* q = nullptr;
  const void* k = nullptr;  // store only
  const void* v = nullptr;
  void* k_cache = nullptr;
  void* v_cache = nullptr;
  void* out = nullptr;
  const int64_t* slots = nullptr;      // store only, int64
  const int32_t* block_table = nullptr;
  const int32_t* lens = nullptr;       // attend only, int32
  int64_t num_seqs = 0;
  int64_t num_heads = 0;
  int64_t num_kv_heads = 0;
  int64_t head_dim = 0;
  int64_t page = 0;       // block_size, tokens per page
  int64_t bt_stride = 0;
  int64_t q_stride = 0;   // element stride between sequences (the fused qkv one)
  int64_t kv_stride = 0;  // ... for the new K/V
  int64_t out_stride = 0;
  float scale = 1.0f;
  int dtype = kF32;
};

struct DenseJob {
  const void* q = nullptr;
  const void* k = nullptr;
  const void* v = nullptr;
  void* out = nullptr;
  int64_t batch = 0, seq_q = 0, seq_k = 0, heads = 0, kv_heads = 0,
          head_dim = 0;
  int64_t qb = 0, qt = 0, qh = 0;
  int64_t kb = 0, kt = 0, kh = 0;
  int64_t vb = 0, vt = 0, vh = 0;
  int64_t ob = 0, ot = 0, oh = 0;
  float scale = 1.0f;
  int dtype = kF32;
  bool causal = false;
};

// ---------------------------------------------------------------------------
// workspace
// ---------------------------------------------------------------------------
// Per-thread, per-geometry, aligned, and never returned to the allocator: a
// 33 MB `torch.empty` is served by mmap, so freeing it unmaps and the next call
// faults pages in one at a time (that was 1.4-1.7x on the dense path before the
// Mojo tree started reusing its workspace). Keyed on a geometry hash, so two
// live geometries cannot alias each other's bytes.
char* workspace(uint64_t key, size_t bytes);
void workspace_clear();
size_t workspace_bytes();

// ---------------------------------------------------------------------------
// dense driver
// ---------------------------------------------------------------------------
// Policy interface (all static, all `P::`):
//
//   static constexpr int RQT;   // query rows per register tile (1..16)
//   static constexpr int KT;    // keys per kv tile == logits width
//   static int rq(int64_t seq_q, int hd);  // rows per work item
//   static size_t slab_bytes(int64_t seq_k, int hd);
//   static size_t aux_bytes(int rq, int hd);   // acc/m/l + logits, one thread
//   static void pack(const DenseJob&, int64_t b, int64_t kvh, char* slab);
//   static void qk(const DenseJob&, const char* slab, int64_t b, int64_t kvh,
//                  const char* q_base, int nrows, int64_t k0, int nkeys,
//                  int64_t row_limit0, float* logits);
//   static void softmax(float* logits, int nrows, int nkeys, float c,
//                       float* m, float* l, float* alpha);
//   static void pv(const DenseJob&, const char* slab, int64_t b, int64_t kvh,
//                  int64_t k0, int nkeys, const float* p, int nrows,
//                  float* acc);
//
// `q_base` addresses query row 0 of this head; row r is `q_base + r*qt`
// elements. `row_limit0` is the last key row 0 may see, and row r sees
// `row_limit0 + r` (bottom-right-aligned causal).
template <class P>
void dense_run(const DenseJob& job, int workers) {
  const int hd = static_cast<int>(job.head_dim);
  const int64_t group = job.heads / job.kv_heads;
  const int rq = P::rq(job.seq_q, hd);
  const int64_t nqb = (job.seq_q + rq - 1) / rq;
  const int64_t nbh = job.batch * job.kv_heads;
  const int64_t heads_total = job.batch * job.heads;
  const size_t slab = P::slab_bytes(job.seq_k, hd);
  const size_t aux = P::aux_bytes(rq, hd);

  // One slab per thread is enough when the thread packs the head it then
  // attends; otherwise every (batch, kv head) needs its own.
  const bool fuse =
      (group == 1) && (nbh >= workers) && (nbh % workers == 0);
  const size_t cells = fuse ? static_cast<size_t>(workers) : nbh;
  const uint64_t key = 0x9E3779B97F4A7C15ull ^
                       (static_cast<uint64_t>(job.seq_k) * 0x100000001B3ull) ^
                       (static_cast<uint64_t>(hd) << 40) ^
                       (static_cast<uint64_t>(job.dtype) << 56) ^
                       (static_cast<uint64_t>(job.heads) << 24) ^
                       (static_cast<uint64_t>(job.batch) << 8) ^
                       static_cast<uint64_t>(rq);
  char* ws = workspace(key, cells * slab + static_cast<size_t>(workers) * aux);
  char* aux_base = ws + cells * slab;
  const int64_t diag = job.seq_k - job.seq_q;
  const float c = job.scale * kLog2e;
  const int esz = elem_size(job.dtype);

#pragma omp parallel num_threads(workers)
  {
    const int tid = omp_get_thread_num();
    char* my_aux = aux_base + static_cast<size_t>(tid) * aux;
    float* logits = reinterpret_cast<float*>(my_aux);
    float* acc = logits + static_cast<size_t>(P::RQT) * P::KT;
    float* m = acc + static_cast<size_t>(rq) * hd;
    float* l = m + rq;
    float* alpha = l + rq;

    auto attend = [&](int64_t b, int64_t h, int64_t qb, const char* sl) {
      const int64_t q0 = qb * rq;
      const int nrows = static_cast<int>(std::min<int64_t>(rq, job.seq_q - q0));
      const char* qp = static_cast<const char*>(job.q) +
                       (b * job.qb + q0 * job.qt + h * job.qh) * esz;
      // "No mask" is a limit no key index can reach, but it still has to fit in
      // an int32: the mask is applied with 32-bit lane compares.
      const int64_t row_limit0 = job.causal ? (q0 + diag) : (int64_t(1) << 30);
      const bool empty = job.causal && (row_limit0 + nrows - 1) < 0;
      std::memset(acc, 0, sizeof(float) * nrows * hd);
      for (int r = 0; r < nrows; ++r) {
        m[r] = -std::numeric_limits<float>::infinity();
        l[r] = 0.0f;
      }
      if (!empty) {
        for (int64_t k0 = 0; k0 < job.seq_k; k0 += P::KT) {
          const int nkeys =
              static_cast<int>(std::min<int64_t>(P::KT, job.seq_k - k0));
          for (int r0 = 0; r0 < nrows; r0 += P::RQT) {
            const int sub = std::min<int>(P::RQT, nrows - r0);
            P::qk(job, sl, b, h / group, qp + static_cast<int64_t>(r0) * job.qt * esz,
                  sub, k0, nkeys, row_limit0 + r0, logits);
            P::softmax(logits, sub, nkeys, c, m + r0, l + r0, alpha);
            for (int r = 0; r < sub; ++r) {
              const float a = alpha[r];
              if (a == 1.0f) continue;
              float* dst = acc + (r0 + r) * hd;
              for (int d = 0; d < hd; ++d) dst[d] *= a;
            }
            P::pv(job, sl, b, h / group, k0, nkeys, logits, sub,
                  acc + r0 * hd);
          }
        }
      }
      for (int r = 0; r < nrows; ++r) {
        char* op = static_cast<char*>(job.out) +
                   (b * job.ob + (q0 + r) * job.ot + h * job.oh) * esz;
        const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        const float* src = acc + r * hd;
        if (job.dtype == kF32) {
          float* o = reinterpret_cast<float*>(op);
          for (int d = 0; d < hd; ++d) o[d] = src[d] * inv;
        } else {
          bf16* o = reinterpret_cast<bf16*>(op);
          for (int d = 0; d < hd; ++d) o[d] = f32_to_bf16(src[d] * inv);
        }
      }
    };

    if (fuse) {
      // Work item == (batch, kv head): pack it, then attend every query block
      // of its query head. No barrier, and the packed head is what was just
      // written, so it is still in L1/L2.
#pragma omp for schedule(static)
      for (int64_t i = 0; i < nbh; ++i) {
        // One slab per thread: with `nbh > workers` a thread takes several
        // heads in a row, and each pack overwrites the previous one only after
        // its attention is done.
        char* sl = ws + static_cast<size_t>(tid) * slab;
        const int64_t b = i / job.kv_heads, kvh = i % job.kv_heads;
        P::pack(job, b, kvh, sl);
        for (int64_t qb = 0; qb < nqb; ++qb) attend(b, kvh, qb, sl);
      }
    } else {
#pragma omp for schedule(static)
      for (int64_t i = 0; i < nbh; ++i)
        P::pack(job, i / job.kv_heads, i % job.kv_heads,
                ws + static_cast<size_t>(i) * slab);
      // The `omp for` above ends in an implicit barrier: every slab is complete
      // before any thread reads one.
#pragma omp for schedule(dynamic, 1)
      for (int64_t t = 0; t < heads_total * nqb; ++t) {
        // Boustrophedon over the query blocks of one head: causal work grows
        // with the block index, so pair the cheapest with the most expensive
        // instead of giving thread 0 the first contiguous slice (which idles
        // half the cores at the end of a prefill).
        const int64_t hp = t / nqb, qb_raw = t % nqb;
        const int64_t qb = (qb_raw & 1) ? (nqb - 1 - (qb_raw >> 1))
                                        : (qb_raw >> 1);
        const int64_t b = hp / job.heads, h = hp % job.heads;
        attend(b, h, qb, ws + static_cast<size_t>(b * job.kv_heads +
                                                 h / group) *
                                    slab);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// paged driver
// ---------------------------------------------------------------------------
// Policy interface:
//   static void store_head(const PagedJob&, int64_t seq, int64_t head);
//   static void attend_head(const PagedJob&, int64_t seq, int64_t head,
//                           char* ws);
//   static size_t aux_bytes(int hd);   // per thread
//
// The store and the attend share the cache layout the policy defines; nothing
// else in the tree knows it.
template <class P>
void store_run(const PagedJob& job, int workers) {
  const int64_t tasks = job.num_seqs * job.num_heads;
#pragma omp parallel for num_threads(workers) schedule(static)
  for (int64_t t = 0; t < tasks; ++t) {
    if (job.slots[t / job.num_heads] < 0) continue;
    P::store_head(job, t / job.num_heads, t % job.num_heads);
  }
}

template <class P>
void paged_run(const PagedJob& job, int workers) {
  const int64_t tasks = job.num_seqs * job.num_heads;
  const size_t aux = P::aux_bytes(static_cast<int>(job.head_dim));
  const uint64_t key = 0x2545F4914F6CDD1Dull ^
                       (static_cast<uint64_t>(job.head_dim) << 32) ^
                       static_cast<uint64_t>(job.dtype);
  char* ws = workspace(key, static_cast<size_t>(workers) * aux);
#pragma omp parallel for num_threads(workers) schedule(static)
  for (int64_t t = 0; t < tasks; ++t) {
    P::attend_head(job, t / job.num_heads, t % job.num_heads,
                   ws + static_cast<size_t>(omp_get_thread_num()) * aux);
  }
}

}  // namespace attn
