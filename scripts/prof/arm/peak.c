// Peak-throughput probes for the m9g (Graviton 5 / Neoverse-V3).
//   gcc -O3 -march=armv8.2-a+bf16+fp16 -fopenmp -o peak peak.c && ./peak
//
// Intrinsics plus an empty asm barrier on the operands each iteration: without
// the barrier GCC hoists the loop-invariant matmul and reports nonsense (1e8
// GFLOP/s), which is what the first version of this probe did.
#include <arm_neon.h>
#include <omp.h>
#include <stdio.h>
#include <time.h>

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

#define ITER 2000000

// 4 dependent integer adds (1 cycle each) per iteration -> the clock.
static double clock_ghz(void) {
  long x = 1, n = ITER;
  double t0 = now();
  __asm__ volatile("1:\n\t"
                   "add %0, %0, #1\n\t"
                   "add %0, %0, #1\n\t"
                   "add %0, %0, #1\n\t"
                   "add %0, %0, #1\n\t"
                   "subs %1, %1, #1\n\t"
                   "b.ne 1b\n\t"
                   : "+r"(x), "+r"(n)
                   :
                   : "cc");
  return (double)ITER * 4 / (now() - t0) / 1e9;
}

static float sink;

static double fma_probe(void) {
  float32x4_t a0 = vdupq_n_f32(1.f), a1 = vdupq_n_f32(1.f);
  float32x4_t a2 = vdupq_n_f32(1.f), a3 = vdupq_n_f32(1.f);
  float32x4_t a4 = vdupq_n_f32(1.f), a5 = vdupq_n_f32(1.f);
  float32x4_t a6 = vdupq_n_f32(1.f), a7 = vdupq_n_f32(1.f);
  float32x4_t b = vdupq_n_f32(1.f), c = vdupq_n_f32(1e-7f);
  double t0 = now();
  for (long i = 0; i < ITER; i++) {
    __asm__ volatile("" : "+w"(b), "+w"(c));
    a0 = vfmaq_f32(a0, b, c); a1 = vfmaq_f32(a1, b, c);
    a2 = vfmaq_f32(a2, b, c); a3 = vfmaq_f32(a3, b, c);
    a4 = vfmaq_f32(a4, b, c); a5 = vfmaq_f32(a5, b, c);
    a6 = vfmaq_f32(a6, b, c); a7 = vfmaq_f32(a7, b, c);
  }
  double dt = now() - t0;
  sink = vgetq_lane_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)), 0) +
         vgetq_lane_f32(vaddq_f32(vaddq_f32(a4, a5), vaddq_f32(a6, a7)), 0);
  return (double)ITER * 8 * 4 * 2 / dt / 1e9;
}

static double bfmmla_probe(void) {
  bfloat16x8_t x = vdupq_n_bf16(1.f), y = vdupq_n_bf16(1.f);
  float32x4_t a0 = vdupq_n_f32(1.f), a1 = vdupq_n_f32(1.f);
  float32x4_t a2 = vdupq_n_f32(1.f), a3 = vdupq_n_f32(1.f);
  float32x4_t a4 = vdupq_n_f32(1.f), a5 = vdupq_n_f32(1.f);
  float32x4_t a6 = vdupq_n_f32(1.f), a7 = vdupq_n_f32(1.f);
  double t0 = now();
  for (long i = 0; i < ITER; i++) {
    __asm__ volatile("" : "+w"(x), "+w"(y), "+w"(a0));
    a0 = vbfmmlaq_f32(a0, x, y); a1 = vbfmmlaq_f32(a1, x, y);
    a2 = vbfmmlaq_f32(a2, x, y); a3 = vbfmmlaq_f32(a3, x, y);
    a4 = vbfmmlaq_f32(a4, x, y); a5 = vbfmmlaq_f32(a5, x, y);
    a6 = vbfmmlaq_f32(a6, x, y); a7 = vbfmmlaq_f32(a7, x, y);
  }
  double dt = now() - t0;
  sink = vgetq_lane_f32(a0, 0) + vgetq_lane_f32(a7, 0);
  return (double)ITER * 8 * 32 / dt / 1e9;
}

static double bfdot_probe(void) {
  bfloat16x8_t x = vdupq_n_bf16(1.f), y = vdupq_n_bf16(1.f);
  float32x4_t a0 = vdupq_n_f32(1.f), a1 = vdupq_n_f32(1.f);
  float32x4_t a2 = vdupq_n_f32(1.f), a3 = vdupq_n_f32(1.f);
  float32x4_t a4 = vdupq_n_f32(1.f), a5 = vdupq_n_f32(1.f);
  float32x4_t a6 = vdupq_n_f32(1.f), a7 = vdupq_n_f32(1.f);
  double t0 = now();
  for (long i = 0; i < ITER; i++) {
    __asm__ volatile("" : "+w"(x), "+w"(y), "+w"(a0));
    a0 = vbfdotq_f32(a0, x, y); a1 = vbfdotq_f32(a1, x, y);
    a2 = vbfdotq_f32(a2, x, y); a3 = vbfdotq_f32(a3, x, y);
    a4 = vbfdotq_f32(a4, x, y); a5 = vbfdotq_f32(a5, x, y);
    a6 = vbfdotq_f32(a6, x, y); a7 = vbfdotq_f32(a7, x, y);
  }
  double dt = now() - t0;
  sink = vgetq_lane_f32(a0, 0) + vgetq_lane_f32(a7, 0);
  return (double)ITER * 8 * 16 / dt / 1e9;
}

int main(void) {
  int nt = omp_get_max_threads();
  printf("threads=%d  clock=%.2f GHz\n", nt, clock_ghz());
  for (int t = 1; t <= nt; t *= 2) {
    omp_set_num_threads(t);
    double f = 0, b = 0, d = 0;
#pragma omp parallel
    {
#pragma omp barrier
      f = fma_probe();
      b = bfmmla_probe();
      d = bfdot_probe();
    }
    printf("  t=%d  fp32 FMA %6.1f   bfmmla %6.1f   bfdot %6.1f  GFLOP/s/core\n",
           t, f, b, d);
  }
  return sink == 0.f;
}
