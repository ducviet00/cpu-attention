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
  bool do_store = false;  // attend entry: also scatter k/v first (one region)
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
//   static int kt_();           // keys per kv tile == logits width
//   static int rq(int64_t seq_q, int hd);  // rows per work item
//   static int acc_rows(int rq);           // rows the accumulator is sized for
//                                          // (>= roundup(rq, RQT): `pv` runs its
//                                          // row loop over the whole tile)
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
//
// Optional policy members, all detected rather than declared by the other
// policies:
//   static constexpr bool zeroes_acc = true;
//       `pv` initialises the accumulator itself on its first key tile, so the
//       driver's memset is dead work.
//   static size_t logit_aux_floats(int kt);
//       floats that follow the logits tile before the accumulator starts (the
//       vec bf16 policy keeps one packed probability pair per key pair per row).
//   static void store_bf16(char* dst, const float* src, int hd, float inv);
//       a vectorised output epilogue; the default scalar loop is 1.6 us of a
//       27 us call at 128:8:64 because `f32_to_bf16`'s NaN branch stops GCC
//       from vectorising it.
template <class P>
constexpr size_t logit_aux_floats(int kt) {
  if constexpr (requires { P::logit_aux_floats(kt); })
    return P::logit_aux_floats(kt);
  else
    return 0;
}

template <class P>
inline void store_bf16(char* dst, const float* src, int hd, float inv) {
  if constexpr (requires { P::store_bf16(dst, src, hd, inv); }) {
    P::store_bf16(dst, src, hd, inv);
  } else {
    bf16* o = reinterpret_cast<bf16*>(dst);
    for (int d = 0; d < hd; ++d) o[d] = f32_to_bf16(src[d] * inv);
  }
}

template <class P>
constexpr bool zeroes_acc() {
  if constexpr (requires { P::zeroes_acc; })
    return P::zeroes_acc;
  else
    return false;
}

template <class P>
void dense_run(const DenseJob& job, int workers) {
  const int hd = static_cast<int>(job.head_dim);
  const int64_t group = job.heads / job.kv_heads;
  const int rq = P::rq(job.seq_q, hd);
  const int arows = P::acc_rows(rq);
  const int64_t nqb = (job.seq_q + rq - 1) / rq;
  const int64_t nbh = job.batch * job.kv_heads;
  const int64_t heads_total = job.batch * job.heads;
  const size_t slab = P::slab_bytes(job.seq_k, hd);
  const size_t aux = P::aux_bytes(rq, hd);

  // One slab per thread is enough when the thread packs the head it then
  // attends; otherwise every (batch, kv head) needs its own.
  //
  // Which one wins is a cache question, not a code question. Fused keeps a
  // thread's K/V in its own L2 (and skips the barrier), but the threads then
  // work on *different* heads, so the working set is `workers` heads' worth of
  // K/V: past the LLC that is DRAM-bound (4096:16:128 fp32 is 4 MB per head,
  // 32 MB across 8 threads, against a 16 MB L3 -- 1.6x slower than sharing).
  // Split packs every head once and the task order (consecutive query blocks of
  // one head, dynamic scheduling) makes the concurrent threads share one head's
  // slab through L3.
  const size_t per_thread_set = slab * static_cast<size_t>(workers);
  const int fuse_env = env_int("CPU_ATTENTION_FUSE", -1, -1, 1);
  const bool fuse_ok =
      (group == 1) && (nbh >= workers) && (nbh % workers == 0) &&
      per_thread_set <= static_cast<size_t>(last_level_cache_bytes()) * 3 / 5;
  const bool fuse = fuse_env < 0 ? fuse_ok : (fuse_env != 0);
  const size_t cells = fuse ? static_cast<size_t>(workers) : nbh;
  uint64_t key = key_mix(static_cast<uint64_t>(job.seq_k),
                         static_cast<uint64_t>(hd) + 977 * job.dtype);
  key = key_mix(key, static_cast<uint64_t>(job.batch + 31 * job.heads +
                                          131 * job.kv_heads + 7919 * rq));
  char* ws = workspace(key, cells * slab + static_cast<size_t>(workers) * aux);
  char* aux_base = ws + cells * slab;
  const int64_t diag = job.seq_k - job.seq_q;
  const float c = job.scale * kLog2e;
  const int esz = elem_size(job.dtype);

#pragma omp parallel num_threads(workers)
  {
    const int tid = omp_get_thread_num();
    char* my_aux = aux_base + static_cast<size_t>(tid) * aux;
    const int kt = P::kt_();
    float* logits = reinterpret_cast<float*>(my_aux);
    float* acc = logits + static_cast<size_t>(P::RQT) * kt +
                 logit_aux_floats<P>(kt);
    float* m = acc + static_cast<size_t>(arows) * hd;
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
      // A policy whose `pv` zeroes the accumulator on its first key tile (the
      // AMX kernel's tile_zero) does not need this, and at head_dim 128 it is
      // 16 KB of stores per work item. Detected rather than declared so the
      // other policies are untouched.
      if constexpr (!zeroes_acc<P>())
        std::memset(acc, 0, sizeof(float) * static_cast<size_t>(arows) * hd);
      for (int r = 0; r < nrows; ++r) {
        m[r] = -std::numeric_limits<float>::infinity();
        l[r] = 0.0f;
      }
      if (!empty) {
        // Stop at the last row's causal limit, and skip a query chunk whose own
        // limit is behind the tile: without this a causal call computes every
        // key of every tile for every row, up to 5x the useful work at rq=32.
        const int64_t kend =
            job.causal ? std::min(job.seq_k, row_limit0 + nrows) : job.seq_k;
        for (int64_t k0 = 0; k0 < kend; k0 += kt) {
          const int nkeys =
              static_cast<int>(std::min<int64_t>(kt, job.seq_k - k0));
          for (int r0 = 0; r0 < nrows; r0 += P::RQT) {
            const int sub = std::min<int>(P::RQT, nrows - r0);
            // Skip only when the chunk's *last* row cannot see this tile; an
            // earlier row of the block may still have keys inside it.
            if (job.causal && row_limit0 + r0 + sub - 1 < k0) continue;
            // Exact per-chunk key bound: the chunk's last row can see key
            // `row_limit0 + r0 + sub - 1`, so the boundary tile shrinks instead
            // of running the full tile and masking it.
            const int nk = job.causal
                               ? static_cast<int>(std::min<int64_t>(
                                     nkeys, row_limit0 + r0 + sub - k0))
                               : nkeys;
            if (nk <= 0) continue;
            P::qk(job, sl, b, h / group,
                  qp + static_cast<int64_t>(r0) * job.qt * esz, sub, k0, nk,
                  row_limit0 + r0, logits);
            P::softmax(logits, sub, nk, c, m + r0, l + r0, alpha);
            for (int r = 0; r < sub; ++r) {
              const float a = alpha[r];
              // a == 1 is a no-op; a == 0 only happens on the first tile that
              // contributes (the running max was -inf), where the accumulator
              // is still exactly zero -- scaling it is 16 KB of stores for
              // nothing, and 1.10x at 256:16:128 when it was skipped.
              if (a == 1.0f || a == 0.0f) continue;
              float* dst = acc + (r0 + r) * hd;
              for (int d = 0; d < hd; ++d) dst[d] *= a;
            }
            P::pv(job, sl, b, h / group, k0, nk, logits, sub, acc + r0 * hd);
          }
        }
      }
      for (int r = 0; r < nrows; ++r) {
        char* op = static_cast<char*>(job.out) +
                   (b * job.ob + (q0 + r) * job.ot + h * job.oh) * esz;
        // A row whose softmax saw no key at all is exactly zero, and it must be
        // written without touching the accumulator: with `zeroes_acc` (or a
        // fully masked item, e.g. seq_q > seq_k) `pv` never ran, so the
        // accumulator holds the previous work item's bytes and `NaN * 0` is
        // NaN. Found by tests/fuzz_dense.py case 46 on the AMX box.
        if (l[r] <= 0.0f) {
          std::memset(op, 0, static_cast<size_t>(esz) * hd);
          continue;
        }
        const float inv = 1.0f / l[r];
        const float* src = acc + r * hd;
        if (job.dtype == kF32) {
          float* o = reinterpret_cast<float*>(op);
          for (int d = 0; d < hd; ++d) o[d] = src[d] * inv;
        } else {
          store_bf16<P>(op, src, hd, inv);
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
  const bool do_store = job.do_store;
  const int64_t tasks = job.num_seqs * job.num_heads;
  const size_t aux = P::aux_bytes(static_cast<int>(job.head_dim));
  const uint64_t key = key_mix(static_cast<uint64_t>(job.head_dim),
                               31 + 977 * job.dtype);
  char* ws = workspace(key, static_cast<size_t>(workers) * aux);
  // Store and attend share one region: a second fork-join costs ~1.1 us, which
  // is a sixth of the whole call at batch 1. MHA needs no barrier (a task
  // stores and then reads its own head's slice); GQA/MQA does, because the
  // query heads of one group are different tasks and one of them may read a
  // slice another is still writing.
  const bool barrier = job.num_heads != job.num_kv_heads;
#pragma omp parallel num_threads(workers)
  {
    if (do_store) {
#pragma omp for schedule(static)
      for (int64_t t = 0; t < tasks; ++t) {
        if (job.slots[t / job.num_heads] < 0) continue;
        P::store_head(job, t / job.num_heads, t % job.num_heads);
      }
      if (barrier) {
#pragma omp barrier
      }
    }
#pragma omp for schedule(static)
    for (int64_t t = 0; t < tasks; ++t)
      P::attend_head(job, t / job.num_heads, t % job.num_heads,
                     ws + static_cast<size_t>(omp_get_thread_num()) * aux);
  }
}

}  // namespace attn
