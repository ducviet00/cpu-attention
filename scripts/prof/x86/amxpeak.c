#define _GNU_SOURCE
// AMX / AVX-512 peak-throughput probe for the attention-shaped inner loops.
//
//   gcc -O3 -march=native -mamx-tile -mamx-bf16 -mamx-fp16 -fopenmp \
//       -o /tmp/amxpeak scripts/prof/x86/amxpeak.c
//   OMP_NUM_THREADS=1 /tmp/amxpeak
//
// Reports GFLOP/s (2 flops per MAC) per loop, best of N. gcc's AMX intrinsics
// take *literal* tile numbers (an "i" constraint), so the tile ids below are
// hard-coded.
//
// Why this exists: the dense kernel's bf16 path is 1.3-1.9x behind vLLM's VEC
// kernel and 1.7-6.7x behind torch SDPA, and SDPA is AMX-backed on this box.
// Before building an AMX path, measure what AMX actually delivers here.
#include <immintrin.h>
#include <signal.h>
#include <stdint.h>
#include <ucontext.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define ARCH_REQ_XCOMP_PERM 0x1023
#define XFEATURE_XTILEDATA 18

static void on_sig(int sig, siginfo_t *si, void *uc) {
  ucontext_t *u = (ucontext_t *)uc;
  fprintf(stderr, "signal %d rip=%p\n", sig,
          (void *)u->uc_mcontext.gregs[REG_RIP]);
  _exit(3);
}

static double now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int amx_init(void) {
  long r = syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
  if (r != 0) {
    fprintf(stderr, "arch_prctl(XTILEDATA) failed: AMX unusable\n");
    return -1;
  }
  return 0;
}

// 8 tiles: 16 rows x 64 bytes (32 bf16 or 16 fp32).
// tilecfg layout: palette(1), start_row(1), rsvd(14), colsb[8] as **u16** at
// offset 16, rsvd(16), rows[8] as u8 at offset 48.
static void tilecfg_bf16(void) {
  uint8_t cfg[64] __attribute__((aligned(64)));
  memset(cfg, 0, sizeof cfg);
  cfg[0] = 1;  // palette 1
  uint16_t *colsb = (uint16_t *)(cfg + 16);
  for (int i = 0; i < 8; i++) {
    colsb[i] = 64;     // 32 bf16 / 16 fp32 per row
    cfg[48 + i] = 16;  // rows
  }
  _tile_loadconfig(cfg);
}

// D[16x16] += A[16x32] @ B[32x16]: 512 MACs per instruction. Tiles 0,1 hold the
// operands, 2..7 are six independent accumulator chains.
#define BODY_BF16                                          \
  _tile_dpbf16ps(2, 0, 1);                                 \
  _tile_dpbf16ps(3, 0, 1);                                 \
  _tile_dpbf16ps(4, 0, 1);                                 \
  _tile_dpbf16ps(5, 0, 1);                                 \
  _tile_dpbf16ps(6, 0, 1);                                 \
  _tile_dpbf16ps(7, 0, 1);

__attribute__((noinline)) static double amx_bf16_peak(long iters) {
  amx_init();
  tilecfg_bf16();
  int8_t *a = aligned_alloc(64, 16 * 64);
  int8_t *b = aligned_alloc(64, 16 * 64);
  memset(a, 1, 16 * 64);
  memset(b, 1, 16 * 64);
  double tstart = now();
  for (long it = 0; it < iters; it++) {
    _tile_loadd(0, a, 64);
    _tile_loadd(1, b, 64);
    for (int i = 0; i < 64; i++) {
      BODY_BF16
    }
  }
  double t = now() - tstart;
  _tile_release();
  free(a);
  free(b);
  return 64.0 * 6 * 512 * 2 * iters / t / 1e9;
}

#ifdef __AMX_FP16__
__attribute__((noinline)) static double amx_fp16_peak(long iters) {
  _Float16 *a = aligned_alloc(64, 16 * 64);
  _Float16 *b = aligned_alloc(64, 16 * 64);
  for (int i = 0; i < 16 * 32; i++) { a[i] = 1; b[i] = 1; }
  double tstart = now();
  for (long it = 0; it < iters; it++) {
    _tile_loadd(0, a, 64);
    _tile_loadd(1, b, 64);
    for (int i = 0; i < 64; i++) {
      _tile_dpfp16ps(2, 0, 1);
      _tile_dpfp16ps(3, 0, 1);
      _tile_dpfp16ps(4, 0, 1);
      _tile_dpfp16ps(5, 0, 1);
      _tile_dpfp16ps(6, 0, 1);
      _tile_dpfp16ps(7, 0, 1);
    }
  }
  double t = now() - tstart;
  _tile_release();
  free(a);
  free(b);
  return 64.0 * 6 * 512 * 2 * iters / t / 1e9;
}
#endif

__attribute__((noinline)) static double fma_fp32_peak(long iters) {
  __m512 a = _mm512_set1_ps(1.f), b = _mm512_set1_ps(1.f);
  __m512 acc[8];
  for (int i = 0; i < 8; i++) acc[i] = _mm512_setzero_ps();
  double tstart = now();
  for (long it = 0; it < iters; it++) {
    for (int i = 0; i < 64; i++)
      for (int j = 0; j < 8; j++) acc[j] = _mm512_fmadd_ps(a, b, acc[j]);
  }
  double t = now() - tstart;
  double g = 64.0 * 8 * 16 * 2 * iters / t / 1e9;
  double sink = 0;
  for (int i = 0; i < 8; i++) sink += _mm512_reduce_add_ps(acc[i]);
  return g + 0.0 * sink;
}

__attribute__((noinline)) static double dpbf16_peak(long iters) {
  __m512bh a = (__m512bh)_mm512_set1_ps(1.f), b = (__m512bh)_mm512_set1_ps(1.f);
  __m512 acc[8];
  for (int i = 0; i < 8; i++) acc[i] = _mm512_setzero_ps();
  double tstart = now();
  for (long it = 0; it < iters; it++) {
    for (int i = 0; i < 64; i++)
      for (int j = 0; j < 8; j++) acc[j] = _mm512_dpbf16_ps(acc[j], a, b);
  }
  double t = now() - tstart;
  double g = 64.0 * 8 * 32 * 2 * iters / t / 1e9;
  double sink = 0;
  for (int i = 0; i < 8; i++) sink += _mm512_reduce_add_ps(acc[i]);
  return g + 0.0 * sink;
}

// A memory-fed Q@K^T panel: 16 query rows x `keys` keys, contraction over
// head_dim, K read 16 keys at a time from a [keys][head_dim] buffer (the packed
// layout the softmax wants). This is amx_bf16_peak with real operand loads.
__attribute__((noinline)) static double amx_qk(int head_dim, int keys, long reps) {
  int8_t *q = aligned_alloc(64, 16 * head_dim * 2 + 64);
  int8_t *k = aligned_alloc(64, (size_t)keys * head_dim * 2 + 64);
  memset(q, 1, (size_t)16 * head_dim * 2);
  memset(k, 1, (size_t)keys * head_dim * 2);
  static float out[16 * 16] __attribute__((aligned(64)));
  int nd = head_dim / 32;  // 32 bf16 = one contraction step
  double tstart = now();
  for (long r = 0; r < reps; r++) {
    for (int t = 0; t < keys; t += 16) {
      _tile_zero(2);
      _tile_loadd(1, k + (size_t)t * head_dim * 2, head_dim * 2);
      // Q panels: tile i holds dim slice i (32 bf16) of all 16 rows; they live
      // in tiles 0 and 1 on the d64 path, so reload per key panel (pessimistic).
      for (int i = 0; i < nd; i++) {
        _tile_loadd(0, q + (size_t)i * 64, head_dim * 2);
        _tile_dpbf16ps(2, 0, 1);
      }
      _tile_stored(2, out, 64);
    }
  }
  double tm = now() - tstart;
  _tile_release();
  free(q);
  free(k);
  return (double)16 * keys * head_dim * 2 * reps / tm / 1e9;
}

// P@V-shaped: 16 query rows x head_dim outputs, contraction over `keys`.
// P is [16 rows][keys] bf16, V is [keys][head_dim] bf16.
__attribute__((noinline)) static double amx_pv(int head_dim, int keys, long reps) {
  int8_t *p = aligned_alloc(64, 16 * keys * 2 + 64);
  int8_t *v = aligned_alloc(64, (size_t)keys * head_dim * 2 + 64);
  memset(p, 1, (size_t)16 * keys * 2);
  memset(v, 1, (size_t)keys * head_dim * 2);
  static float out[16 * 16] __attribute__((aligned(64)));
  int nd = head_dim / 16;  // 16 fp32 outputs = one tile
  double tstart = now();
  for (long r = 0; r < reps; r++) {
    for (int t = 0; t < keys; t += 32) {
      _tile_loadd(1, p + (size_t)t * 2, keys * 2);  // P: 16 x 32 bf16
      for (int d = 0; d < nd; d++) {
        _tile_zero(2);
        // V panel for output tile d: 32 keys x 16 dims, transposed so the rows
        // are keys -- use stride 32 bytes (head_dim*2) as if pre-packed.
        _tile_loadd(0, v + (size_t)t * head_dim * 2 + (size_t)d * 32, head_dim * 2);
        _tile_dpbf16ps(2, 1, 0);
        _tile_stored(2, out, 64);
      }
    }
  }
  double tm = now() - tstart;
  _tile_release();
  free(p);
  free(v);
  return (double)16 * head_dim * keys * 2 * reps / tm / 1e9;
}

int main(void) {
  double best;
  setvbuf(stdout, NULL, _IONBF, 0);
  struct sigaction sa = {0};
  sa.sa_sigaction = on_sig;
  sa.sa_flags = SA_SIGINFO;
  sigaction(SIGILL, &sa, NULL);
  sigaction(SIGSEGV, &sa, NULL);
  fprintf(stderr, "amx_bf16_peak at %p\n", (void *)amx_bf16_peak);
  fprintf(stderr, "init...\n");
  if (amx_init() != 0) return 1;
  fprintf(stderr, "amx ok\n");
  tilecfg_bf16();
#ifdef _OPENMP
  printf("threads=%d\n", omp_get_max_threads());
#endif

#define RUN(name, call, w, t)                              \
  do {                                                     \
    printf("%-22s ...\n", name);                            \
    tilecfg_bf16();                                          \
    for (int i = 0; i < (w); i++) call;                     \
    best = 0;                                              \
    for (int i = 0; i < (t); i++) {                         \
      double g = call;                                      \
      if (g > best) best = g;                               \
    }                                                       \
    printf("%-22s %8.1f GFLOP/s\n", name, best);            \
    fflush(stdout);                                         \
  } while (0)

  RUN("fma fp32 (avx512)", fma_fp32_peak(2000), 3, 5);
  RUN("dpbf16ps (avx512)", dpbf16_peak(2000), 3, 5);
  RUN("amx tdpbf16ps", amx_bf16_peak(300), 3, 5);
#ifdef __AMX_FP16__
  RUN("amx tdpfp16ps", amx_fp16_peak(300), 3, 5);
#endif
  RUN("amx QK 16x512 d64", amx_qk(64, 512, 500), 3, 5);
  RUN("amx QK 16x512 d128", amx_qk(128, 512, 500), 3, 5);
  RUN("amx PV 16x512 d64", amx_pv(64, 512, 500), 3, 5);
  RUN("amx PV 16x512 d128", amx_pv(128, 512, 500), 3, 5);
  return 0;
}
